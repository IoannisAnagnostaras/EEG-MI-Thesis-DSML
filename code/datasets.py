from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import moabb
import numpy as np
import pandas as pd
import torch
from moabb.datasets import BNCI2014_001, Stieger2021
from moabb.paradigms import MotorImagery
from torch.utils.data import Dataset

from .preprocessing import apply_channelwise_zscore, fit_channelwise_zscore
from .utils import per_trial_zscore


BNCI_22_CHANNELS = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
]

DATASET_REGISTRY = {
    "BNCI2014_001": BNCI2014_001,
    "Stieger2021": Stieger2021,
}

DATASET_LABELS = {
    "BNCI2014_001": ["left_hand", "right_hand", "feet", "tongue"],
    "Stieger2021": ["left_hand", "right_hand", "both_hand", "rest"],
}

_SOURCE_BUNDLE_CACHE: dict[tuple, "ArrayBundle"] = {}


@dataclass
class ArrayBundle:
    x: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame


class EEGArrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class EEGUnlabeledDataset(Dataset[torch.Tensor]):
    def __init__(self, x: np.ndarray) -> None:
        self.x = torch.from_numpy(x).float()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.x[index]


def _matrix_inv_sqrt(cov: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-6, None)
    return (eigvecs * (eigvals ** -0.5)) @ eigvecs.T


def _euclidean_align_by_subject(x: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    aligned = np.empty_like(x)
    for subject in np.unique(subjects):
        mask = subjects == subject
        subject_x = x[mask]
        cov = np.mean([trial @ trial.T / trial.shape[-1] for trial in subject_x], axis=0)
        whitening = _matrix_inv_sqrt(cov).astype(np.float32)
        aligned[mask] = np.einsum("ij,bjt->bit", whitening, subject_x)
    return aligned


def _apply_standardization(x: np.ndarray, standardize: str | None) -> np.ndarray:
    if standardize in {None, "none"}:
        return x
    if standardize == "per_trial_zscore":
        return per_trial_zscore(x)
    if standardize == "zscore_train":
        # Train-fitted z-scoring is split-dependent and is applied in training code.
        return x
    raise ValueError(f"Unsupported standardization mode: {standardize}")


def _build_paradigm(cfg: dict, *, mode: str) -> MotorImagery:
    pp = cfg["preprocessing"]
    moabb.set_log_level("ERROR")
    tmin_key = "target_tmin" if mode == "target" else "source_tmin"
    tmax_key = "target_tmax" if mode == "target" else "source_tmax"
    return MotorImagery(
        n_classes=len(pp["target_labels"]),
        channels=BNCI_22_CHANNELS,
        fmin=float(pp["fmin"]),
        fmax=float(pp["fmax"]),
        tmin=float(pp[tmin_key]),
        tmax=float(pp[tmax_key]),
        resample=pp["resample"],
    )


def _dataset_instance(name: str):
    factory = DATASET_REGISTRY[name]
    return factory()


def _label_map(target_labels: Iterable[str]) -> dict[str, int]:
    return {label: index for index, label in enumerate(target_labels)}


def _tuple_or_none(values: Iterable | None) -> tuple | None:
    if values is None:
        return None
    return tuple(values)


def _clone_bundle(bundle: ArrayBundle) -> ArrayBundle:
    return ArrayBundle(
        x=bundle.x.copy(),
        y=bundle.y.copy(),
        meta=bundle.meta.copy(deep=True),
    )


def _source_bundle_cache_key(cfg: dict, dataset_name: str, subjects: list[int] | None) -> tuple:
    pp = cfg["preprocessing"]
    source_labels = cfg.get("source_dataset_labels", {}).get(dataset_name)
    return (
        dataset_name,
        _tuple_or_none(subjects),
        float(pp["fmin"]),
        float(pp["fmax"]),
        float(pp["source_tmin"]),
        float(pp["source_tmax"]),
        float(pp["resample"]),
        tuple(pp.get("target_labels", ())),
        _tuple_or_none(source_labels),
        str(pp.get("source_label_mode", "dataset_specific")),
        str(pp.get("standardize", "per_trial_zscore")),
    )


def _get_source_subjects(cfg: dict, dataset_name: str, held_out_subject: int | None) -> list[int] | None:
    configured = cfg.get("source_dataset_subjects", {}).get(dataset_name)
    if configured is None:
        subjects = None
    else:
        subjects = [int(subject) for subject in configured]

    if dataset_name != "BNCI2014_001" or held_out_subject is None:
        return subjects

    if subjects is None:
        subjects = list(_dataset_instance(dataset_name).subject_list)
    return [subject for subject in subjects if subject != held_out_subject]


def _filter_and_encode_labels(
    labels: np.ndarray,
    *,
    dataset_name: str,
    cfg: dict,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "target":
        dataset_labels = cfg["preprocessing"]["target_labels"]
    else:
        source_label_mode = cfg.get("preprocessing", {}).get("source_label_mode", "dataset_specific")
        if source_label_mode == "unfiltered":
            categories = pd.Categorical(labels)
            return np.ones(len(labels), dtype=bool), categories.codes.astype(np.int64)
        if source_label_mode == "target_only":
            dataset_labels = cfg["preprocessing"]["target_labels"]
        else:
            dataset_labels = cfg.get("source_dataset_labels", {}).get(dataset_name, DATASET_LABELS.get(dataset_name))

    if dataset_labels is None:
        categories = pd.Categorical(labels)
        return np.ones(len(labels), dtype=bool), categories.codes.astype(np.int64)

    label_to_index = _label_map(dataset_labels)
    keep = np.array([label in label_to_index for label in labels], dtype=bool)
    y = np.array([label_to_index[label] for label in labels[keep]], dtype=np.int64)
    return keep, y


def load_motor_imagery_array(
    dataset_name: str,
    cfg: dict,
    *,
    subjects: list[int] | None = None,
    mode: str = "target",
) -> ArrayBundle:
    dataset = _dataset_instance(dataset_name)
    if subjects is not None:
        dataset.subject_list = subjects
    paradigm = _build_paradigm(cfg, mode=mode)
    x, labels, meta = paradigm.get_data(dataset=dataset, subjects=subjects)
    labels = np.asarray(labels)
    keep, y = _filter_and_encode_labels(labels, dataset_name=dataset_name, cfg=cfg, mode=mode)
    x = x[keep].astype(np.float32)
    meta = meta.loc[keep].reset_index(drop=True).copy()
    x = _apply_standardization(x, cfg["preprocessing"].get("standardize", "per_trial_zscore"))
    meta["dataset"] = dataset_name
    return ArrayBundle(x=x, y=y, meta=meta)


def _apply_session_filter(bundle: ArrayBundle, session_name: str | None) -> ArrayBundle:
    if session_name is None or "session" not in bundle.meta.columns:
        return bundle
    mask = bundle.meta["session"].to_numpy() == session_name
    return ArrayBundle(
        x=bundle.x[mask],
        y=bundle.y[mask],
        meta=bundle.meta.loc[mask].reset_index(drop=True),
    )


def _harmonize_time_lengths(arrays: list[np.ndarray]) -> list[np.ndarray]:
    if not arrays:
        return arrays
    time_lengths = [int(array.shape[-1]) for array in arrays]
    if len(set(time_lengths)) == 1:
        return arrays
    target_length = min(time_lengths)
    return [array[..., :target_length] for array in arrays]


def load_source_pool(cfg: dict, *, held_out_subject: int | None = None) -> np.ndarray:
    arrays: list[np.ndarray] = []
    source_session = cfg.get("split", {}).get("source_session")
    apply_ea = bool(cfg.get("preprocessing", {}).get("apply_euclidean_alignment", False))
    for dataset_name in cfg["source_datasets"]:
        subjects = _get_source_subjects(cfg, dataset_name, held_out_subject)
        cache_key = _source_bundle_cache_key(cfg, dataset_name, subjects)
        if cache_key in _SOURCE_BUNDLE_CACHE:
            bundle = _clone_bundle(_SOURCE_BUNDLE_CACHE[cache_key])
        else:
            bundle = load_motor_imagery_array(dataset_name, cfg, subjects=subjects, mode="source")
            _SOURCE_BUNDLE_CACHE[cache_key] = _clone_bundle(bundle)
        bundle = _apply_session_filter(bundle, source_session if dataset_name == "BNCI2014_001" else None)
        if apply_ea and len(bundle.meta) > 0:
            bundle.x = _euclidean_align_by_subject(bundle.x, bundle.meta["subject"].to_numpy())
        arrays.append(bundle.x)
    if not arrays:
        raise ValueError("No source datasets configured.")
    arrays = _harmonize_time_lengths(arrays)
    x = np.concatenate(arrays, axis=0)
    max_trials = cfg["preprocessing"].get("max_source_trials")
    if max_trials:
        x = x[: int(max_trials)]
    if cfg["preprocessing"].get("standardize", "per_trial_zscore") == "zscore_train":
        mean, std = fit_channelwise_zscore(x)
        x = apply_channelwise_zscore(x, mean, std).astype(np.float32)
    return x


def split_loso(bundle: ArrayBundle, cfg: dict) -> list[dict]:
    splits: list[dict] = []
    split_cfg = cfg.get("split", {})
    source_session = split_cfg.get("source_session")
    target_session = split_cfg.get("target_session")
    apply_ea = bool(cfg.get("preprocessing", {}).get("apply_euclidean_alignment", False))
    subjects = sorted(int(subject) for subject in bundle.meta["subject"].unique().tolist())
    for subject in subjects:
        subject_arr = bundle.meta["subject"].to_numpy()
        train_mask = subject_arr != subject
        test_mask = ~train_mask
        if source_session is not None and "session" in bundle.meta.columns:
            train_mask = train_mask & (bundle.meta["session"].to_numpy() == source_session)
        if target_session is not None and "session" in bundle.meta.columns:
            test_mask = test_mask & (bundle.meta["session"].to_numpy() == target_session)
        train_x = bundle.x[train_mask]
        test_x = bundle.x[test_mask]
        if apply_ea:
            train_subjects = bundle.meta.loc[train_mask, "subject"].to_numpy()
            test_subjects = bundle.meta.loc[test_mask, "subject"].to_numpy()
            if len(train_subjects) > 0:
                train_x = _euclidean_align_by_subject(train_x, train_subjects)
            if len(test_subjects) > 0:
                test_x = _euclidean_align_by_subject(test_x, test_subjects)
        splits.append(
            {
                "held_out_subject": subject,
                "train_x": train_x,
                "train_y": bundle.y[train_mask],
                "test_x": test_x,
                "test_y": bundle.y[test_mask],
            }
        )
    return splits


def summarize_datasets(dataset_names: list[str], data_root: Path) -> list[dict]:
    del data_root  # Download/cache path is managed by MOABB/MNE settings.
    summaries = []
    for dataset_name in dataset_names:
        dataset = _dataset_instance(dataset_name)
        subject = dataset.subject_list[:1]
        data = dataset.get_data(subjects=subject)
        subject_key = subject[0]
        first_session = next(iter(data[subject_key].values()))
        first_run = next(iter(first_session.values()))
        channel_names = list(first_run.info["ch_names"])
        event_id = getattr(dataset, "event_id", {})
        summaries.append(
            {
                "dataset": dataset_name,
                "subjects_sampled": subject,
                "channel_count": len(channel_names),
                "eeg_channels_sample": channel_names[:16],
                "event_id": event_id,
            }
        )
    return summaries
