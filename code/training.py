from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .augmentations import gaussian_noise, make_li_paper_view, random_scale
from .datasets import EEGArrayDataset, EEGUnlabeledDataset, load_motor_imagery_array, load_source_pool, split_loso
from .models import EEGClassifier, EEGNetEncoder, LiSENetEncoder, MaskedLatentModel, ProjectionMLP, ShallowConvNetEncoder
from .objectives import NTXentLoss, VICRegLoss
from .preprocessing import apply_channelwise_zscore, fit_channelwise_zscore
from .utils import ensure_dir, get_device, save_json, set_seed


@dataclass
class RunResult:
    held_out_subject: int
    accuracy: float
    kappa: float
    checkpoint: str


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool = False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


def random_view(x: torch.Tensor, mode: str = "amp_add_amp_scale") -> torch.Tensor:
    if mode == "amp_add_amp_scale":
        return gaussian_noise(random_scale(x), std=0.02)
    if mode == "li_paper_mix":
        return make_li_paper_view(x)
    noise = 0.01 * torch.randn_like(x)
    scale = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(0.9, 1.1)
    shift = torch.randint(low=0, high=max(2, x.size(-1) // 20), size=(1,), device=x.device).item()
    return torch.roll(x * scale + noise, shifts=shift, dims=-1)


def build_encoder(cfg: dict, n_channels: int, n_times: int):
    model_name = str(cfg["model"].get("name", "eegnet")).lower()
    if model_name == "li_cnn_senet":
        return LiSENetEncoder(
            n_channels=n_channels,
            n_times=n_times,
            dropout=float(cfg["model"].get("dropout", 0.2)),
            se_reduction=int(cfg["model"].get("se_reduction", 2)),
        )
    if model_name in {"shallow_convnet", "shallowconvnet", "shallow"}:
        return ShallowConvNetEncoder(
            n_channels=n_channels,
            n_times=n_times,
            dropout=float(cfg["model"].get("dropout", 0.5)),
            n_filters=int(cfg["model"].get("n_filters", 40)),
            temporal_kernel=int(cfg["model"].get("temporal_kernel", 25)),
            pool_kernel=int(cfg["model"].get("pool_kernel", 75)),
            pool_stride=int(cfg["model"].get("pool_stride", 15)),
        )
    return EEGNetEncoder(
        n_channels=n_channels,
        embedding_dim=int(cfg["model"]["embedding_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    )


def _split_train_validation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if validation_fraction <= 0.0:
        return x, y, x, y
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=y,
    )
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def _model_selection_mode(cfg: dict) -> str:
    mode = str(cfg["training"].get("model_selection", "validation")).lower()
    if mode not in {"validation", "final"}:
        raise ValueError(f"Unsupported training.model_selection: {mode}")
    return mode


def _apply_train_fitted_zscore_if_needed(
    cfg: dict,
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mode = str(cfg.get("preprocessing", {}).get("standardize", "per_trial_zscore")).lower()
    if mode != "zscore_train":
        return train_x, val_x, test_x
    mean, std = fit_channelwise_zscore(train_x)
    train_x = apply_channelwise_zscore(train_x, mean, std).astype(np.float32)
    val_x = apply_channelwise_zscore(val_x, mean, std).astype(np.float32)
    test_x = apply_channelwise_zscore(test_x, mean, std).astype(np.float32)
    return train_x, val_x, test_x


def train_supervised_one_split(cfg: dict, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, held_out_subject: int) -> RunResult:
    device = get_device(cfg["training"]["device"])
    run_root = Path(cfg["run_root"])
    selection_mode = _model_selection_mode(cfg)
    validation_fraction = float(cfg["training"].get("validation_fraction", 0.2))
    if selection_mode == "validation":
        train_x_fit, train_y_fit, val_x, val_y = _split_train_validation(
            train_x,
            train_y,
            validation_fraction=validation_fraction,
            seed=int(cfg["training"]["seed"]) + held_out_subject,
        )
    else:
        train_x_fit, train_y_fit = train_x, train_y
        val_x, val_y = train_x, train_y
    train_x_fit, val_x, test_x = _apply_train_fitted_zscore_if_needed(cfg, train_x_fit, val_x, test_x)
    train_dataset = EEGArrayDataset(train_x_fit, train_y_fit)
    val_dataset = EEGArrayDataset(val_x, val_y)
    test_dataset = EEGArrayDataset(test_x, test_y)
    supervised_batch_size = int(cfg["training"].get("supervised_batch_size", cfg["training"]["batch_size"]))
    train_loader = make_loader(train_dataset, supervised_batch_size, True, int(cfg["training"]["num_workers"]), drop_last=False)
    val_loader = make_loader(val_dataset, supervised_batch_size, False, int(cfg["training"]["num_workers"]))
    test_loader = make_loader(test_dataset, supervised_batch_size, False, int(cfg["training"]["num_workers"]))

    hidden_dim = cfg["model"].get("classifier_hidden_dim")
    model = EEGClassifier(
        build_encoder(cfg, train_x.shape[1], train_x.shape[-1]),
        n_classes=4,
        hidden_dim=int(hidden_dim) if hidden_dim is not None else None,
        dropout=float(cfg["model"].get("dropout", 0.2)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    epochs = int(cfg["training"]["supervised_epochs"])
    model.train()
    best_state = copy.deepcopy(model.state_dict())
    best_val_accuracy = float("-inf")
    for _ in tqdm(range(epochs), desc=f"supervised-s{held_out_subject}", leave=False):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        if selection_mode == "validation":
            val_accuracy, _ = evaluate_classifier(model, val_loader, device)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_state = copy.deepcopy(model.state_dict())
        else:
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    accuracy, kappa = evaluate_classifier(model, test_loader, device)
    checkpoint_dir = ensure_dir(run_root / "checkpoints" / "supervised")
    checkpoint_path = checkpoint_dir / f"subject_{held_out_subject}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    return RunResult(held_out_subject=held_out_subject, accuracy=accuracy, kappa=kappa, checkpoint=str(checkpoint_path))


def evaluate_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(batch_y.tolist())
    model.train()
    return float(accuracy_score(targets, predictions)), float(cohen_kappa_score(targets, predictions))


def pretrain_ssl(cfg: dict, objective: str, *, held_out_subject: int | None = None) -> str:
    device = get_device(cfg["training"]["device"])
    run_root = Path(cfg["run_root"])
    source_x = load_source_pool(cfg, held_out_subject=held_out_subject)
    dataset = EEGUnlabeledDataset(source_x)
    objective = objective.lower()
    ssl_batch_size = int(cfg["training"].get("ssl_batch_size", cfg["training"]["batch_size"]))
    ssl_drop_last = objective in {"simclr", "vicreg"}
    loader = make_loader(dataset, ssl_batch_size, True, int(cfg["training"]["num_workers"]), drop_last=ssl_drop_last)
    encoder = build_encoder(cfg, source_x.shape[1], source_x.shape[-1]).to(device)

    epochs = int(cfg["training"]["ssl_epochs"])
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        betas=(float(cfg["training"].get("adam_beta1", 0.9)), float(cfg["training"].get("adam_beta2", 0.999))),
    )

    checkpoint_dir = ensure_dir(run_root / "checkpoints" / objective)
    checkpoint_name = f"global.pt" if held_out_subject is None else f"subject_{held_out_subject}.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name

    if objective == "masked":
        model = MaskedLatentModel(encoder, mask_ratio=float(cfg["training"]["mask_ratio"])).to(device)
        optimizer = torch.optim.AdamW(
            list(model.student.parameters()) + list(model.predictor.parameters()) + [model.mask_token],
            lr=float(cfg["training"]["lr"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
            betas=(float(cfg["training"].get("adam_beta1", 0.9)), float(cfg["training"].get("adam_beta2", 0.999))),
        )
        for _ in tqdm(range(epochs), desc=f"masked-pretrain-{held_out_subject or 'global'}", leave=False):
            for batch_x in loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = model(batch_x)
                loss.backward()
                optimizer.step()
                model.update_teacher()
        torch.save(
            {
                "objective": objective,
                "encoder": model.student.state_dict(),
                "teacher": model.teacher.state_dict(),
                "predictor": model.predictor.state_dict(),
            },
            checkpoint_path,
        )
        return str(checkpoint_path)

    projector = ProjectionMLP(encoder.feature_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        betas=(float(cfg["training"].get("adam_beta1", 0.9)), float(cfg["training"].get("adam_beta2", 0.999))),
    )

    if objective == "simclr":
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = NTXentLoss(
            temperature=float(cfg["training"]["temperature"])
        )
    elif objective == "vicreg":
        loss_fn = VICRegLoss(
            sim_coeff=float(cfg["training"]["vicreg_sim_coeff"]),
            var_coeff=float(cfg["training"]["vicreg_var_coeff"]),
            cov_coeff=float(cfg["training"]["vicreg_cov_coeff"]),
        )
    else:
        raise ValueError(f"Unsupported objective: {objective}")

    augmentation_mode = cfg["training"].get("augmentation_mode", "amp_add_amp_scale")
    for _ in tqdm(range(epochs), desc=f"{objective}-pretrain-{held_out_subject or 'global'}", leave=False):
        for batch_x in loader:
            batch_x = batch_x.to(device)
            view_a = random_view(batch_x, augmentation_mode)
            view_b = random_view(batch_x, augmentation_mode)
            optimizer.zero_grad(set_to_none=True)
            z1 = projector(encoder(view_a))
            z2 = projector(encoder(view_b))
            loss = loss_fn(z1, z2)
            loss.backward()
            optimizer.step()

    torch.save({"objective": objective, "encoder": encoder.state_dict(), "projector": projector.state_dict()}, checkpoint_path)
    return str(checkpoint_path)


def finetune_ssl_one_split(
    cfg: dict,
    objective: str,
    checkpoint_path: str | Path,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    held_out_subject: int,
) -> RunResult:
    device = get_device(cfg["training"]["device"])
    run_root = Path(cfg["run_root"])
    encoder = build_encoder(cfg, train_x.shape[1], train_x.shape[-1])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    encoder.load_state_dict(checkpoint["encoder"])
    hidden_dim = cfg["model"].get("classifier_hidden_dim")
    model = EEGClassifier(
        encoder,
        n_classes=4,
        hidden_dim=int(hidden_dim) if hidden_dim is not None else None,
        dropout=float(cfg["model"].get("dropout", 0.2)),
    ).to(device)
    finetune_batch_size = int(cfg["training"].get("finetune_batch_size", cfg["training"]["batch_size"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        betas=(float(cfg["training"].get("adam_beta1", 0.9)), float(cfg["training"].get("adam_beta2", 0.999))),
    )
    criterion = nn.CrossEntropyLoss()
    selection_mode = _model_selection_mode(cfg)
    validation_fraction = float(cfg["training"].get("validation_fraction", 0.2))
    if selection_mode == "validation":
        train_x_fit, train_y_fit, val_x, val_y = _split_train_validation(
            train_x,
            train_y,
            validation_fraction=validation_fraction,
            seed=int(cfg["training"]["seed"]) + held_out_subject,
        )
    else:
        train_x_fit, train_y_fit = train_x, train_y
        val_x, val_y = train_x, train_y
    train_x_fit, val_x, test_x = _apply_train_fitted_zscore_if_needed(cfg, train_x_fit, val_x, test_x)
    train_loader = make_loader(
        EEGArrayDataset(train_x_fit, train_y_fit),
        finetune_batch_size,
        True,
        int(cfg["training"]["num_workers"]),
        drop_last=False,
    )
    val_loader = make_loader(
        EEGArrayDataset(val_x, val_y),
        finetune_batch_size,
        False,
        int(cfg["training"]["num_workers"]),
    )
    test_loader = make_loader(
        EEGArrayDataset(test_x, test_y),
        finetune_batch_size,
        False,
        int(cfg["training"]["num_workers"]),
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_accuracy = float("-inf")
    for _ in tqdm(range(int(cfg["training"]["finetune_epochs"])), desc=f"{objective}-ft-s{held_out_subject}", leave=False):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        if selection_mode == "validation":
            val_accuracy, _ = evaluate_classifier(model, val_loader, device)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_state = copy.deepcopy(model.state_dict())
        else:
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    accuracy, kappa = evaluate_classifier(model, test_loader, device)
    checkpoint_dir = ensure_dir(run_root / "checkpoints" / f"{objective}_finetuned")
    out_path = checkpoint_dir / f"subject_{held_out_subject}.pt"
    torch.save(model.state_dict(), out_path)
    return RunResult(held_out_subject=held_out_subject, accuracy=accuracy, kappa=kappa, checkpoint=str(out_path))


def run_supervised_benchmark(cfg: dict) -> list[RunResult]:
    set_seed(int(cfg["training"]["seed"]))
    bundle = load_motor_imagery_array(cfg["target_dataset"]["name"], cfg, subjects=cfg["target_dataset"].get("subjects"))
    results = []
    for split in split_loso(bundle, cfg):
        results.append(
            train_supervised_one_split(
                cfg,
                split["train_x"],
                split["train_y"],
                split["test_x"],
                split["test_y"],
                split["held_out_subject"],
            )
        )
    return results


def run_ssl_benchmark(cfg: dict, objective: str) -> list[RunResult]:
    set_seed(int(cfg["training"]["seed"]))
    bundle = load_motor_imagery_array(cfg["target_dataset"]["name"], cfg, subjects=cfg["target_dataset"].get("subjects"))
    results = []
    for split in split_loso(bundle, cfg):
        checkpoint_path = pretrain_ssl(cfg, objective, held_out_subject=split["held_out_subject"])
        results.append(
            finetune_ssl_one_split(
                cfg,
                objective,
                checkpoint_path,
                split["train_x"],
                split["train_y"],
                split["test_x"],
                split["test_y"],
                split["held_out_subject"],
            )
        )
    return results


def summarize_results(results: list[RunResult], output_path: str | Path) -> dict:
    accuracy = np.array([result.accuracy for result in results], dtype=np.float64)
    kappa = np.array([result.kappa for result in results], dtype=np.float64)
    payload = {
        "subjects": [result.held_out_subject for result in results],
        "mean_accuracy": float(accuracy.mean()),
        "std_accuracy": float(accuracy.std(ddof=0)),
        "mean_kappa": float(kappa.mean()),
        "std_kappa": float(kappa.std(ddof=0)),
        "per_subject": [
            {
                "held_out_subject": result.held_out_subject,
                "accuracy": result.accuracy,
                "kappa": result.kappa,
                "checkpoint": result.checkpoint,
            }
            for result in results
        ],
    }
    save_json(output_path, payload)
    return payload
