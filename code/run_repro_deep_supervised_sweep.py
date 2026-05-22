from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
from pathlib import Path

from ssl_bci_rebuild.training import run_supervised_benchmark, summarize_results
from ssl_bci_rebuild.utils import ensure_dir, save_json


BASE_CFG: dict = {
    "benchmark_policy": {"name": "li_bciiv2a_train_session_loso"},
    "split": {"source_session": "0train", "target_session": "0train"},
    "target_dataset": {"name": "BNCI2014_001", "subjects": None},
    "source_datasets": ["BNCI2014_001"],
    "source_dataset_labels": {
        "BNCI2014_001": ["left_hand", "right_hand", "feet", "tongue"],
    },
    "preprocessing": {
        "target_labels": ["left_hand", "right_hand", "feet", "tongue"],
        "target_tmin": 0.0,
        "target_tmax": 4.0,
        "source_tmin": 0.0,
        "source_tmax": 4.0,
        "fmin": 3.0,
        "fmax": 40.0,
        "resample": 250,
        "source_label_mode": "dataset_specific",
        "standardize": "none",
        "apply_euclidean_alignment": True,
        "max_source_trials": None,
    },
    "model": {
        "name": "eegnet",
        "embedding_dim": 256,
        "dropout": 0.25,
    },
    "training": {
        "batch_size": 64,
        "ssl_batch_size": 128,
        "supervised_batch_size": 64,
        "finetune_batch_size": 32,
        "supervised_epochs": 300,
        "ssl_epochs": 100,
        "finetune_epochs": 300,
        "lr": 0.001,
        "weight_decay": 0.001,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "temperature": 0.5,
        "augmentation_mode": "amp_add_amp_scale",
        "vicreg_sim_coeff": 25.0,
        "vicreg_var_coeff": 25.0,
        "vicreg_cov_coeff": 1.0,
        "mask_ratio": 0.35,
        "seed": 42,
        "num_workers": 0,
        "validation_fraction": 0.2,
        "model_selection": "final",
        "device": "cuda",
    },
}

WINDOWS = [
    ("abs2_6", 0.0, 4.0),  # absolute 2–6 s on BNCI2014_001
    ("abs3_6", 1.0, 4.0),  # absolute 3–6 s on BNCI2014_001
]

MODELS = [
    (
        "eegnet",
        {
            "name": "eegnet",
            "embedding_dim": 256,
            "dropout": 0.25,
        },
    ),
    (
        "shallow_convnet",
        {
            "name": "shallow_convnet",
            "dropout": 0.5,
            "n_filters": 40,
            "temporal_kernel": 25,
            "pool_kernel": 75,
            "pool_stride": 15,
        },
    ),
]


def _lock_reference(output_root: Path) -> None:
    comparison = Path("outputs") / "official_comparison_latest.csv"
    payload = {
        "locked_reference": "official_bnci_plus_stieger_s1_ssl_r2 + simclr",
        "mean_accuracy": 0.36367245849567127,
        "source_file": str(comparison),
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if comparison.exists():
        payload["source_file_exists"] = True
    else:
        payload["source_file_exists"] = False
    save_json(output_root / "reference_lock.json", payload)


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = ensure_dir(Path("outputs") / f"repro_deep_supervised_sweep_{stamp}")
    _lock_reference(output_root)

    rows: list[dict] = []
    for model_name, model_cfg in MODELS:
        for window_name, tmin, tmax in WINDOWS:
            run_name = f"{model_name}__{window_name}__f3_40__loso_train_session_final300"
            cfg = copy.deepcopy(BASE_CFG)
            cfg["run_root"] = str(output_root / run_name)
            cfg["model"] = copy.deepcopy(model_cfg)
            cfg["preprocessing"]["target_tmin"] = tmin
            cfg["preprocessing"]["target_tmax"] = tmax
            cfg["preprocessing"]["source_tmin"] = tmin
            cfg["preprocessing"]["source_tmax"] = tmax

            save_json(Path(cfg["run_root"]) / "config_used.json", cfg)
            print(f"[RUN] {run_name}", flush=True)
            results = run_supervised_benchmark(cfg)
            summary = summarize_results(results, Path(cfg["run_root"]) / "supervised_summary.json")
            print(
                f"[DONE] {run_name} mean_acc={summary['mean_accuracy']:.6f} mean_kappa={summary['mean_kappa']:.6f}",
                flush=True,
            )
            rows.append(
                {
                    "run_name": run_name,
                    "model_name": model_name,
                    "window_name": window_name,
                    "target_tmin_rel": tmin,
                    "target_tmax_rel": tmax,
                    "fmin": 3.0,
                    "fmax": 40.0,
                    "epochs": 300,
                    "selection": "final",
                    "mean_accuracy": summary["mean_accuracy"],
                    "std_accuracy": summary["std_accuracy"],
                    "mean_kappa": summary["mean_kappa"],
                    "std_kappa": summary["std_kappa"],
                    "run_root": cfg["run_root"],
                }
            )

    rows.sort(key=lambda row: row["mean_accuracy"], reverse=True)
    summary_csv = output_root / "deep_supervised_leaderboard.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    save_json(output_root / "best_config_summary.json", best)
    print(f"[BEST] {best['run_name']} mean_acc={best['mean_accuracy']:.6f}", flush=True)
    print(f"[OUT] {summary_csv}", flush=True)


if __name__ == "__main__":
    main()

