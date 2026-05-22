from __future__ import annotations

import argparse
import copy
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from ssl_bci_rebuild.training import run_ssl_benchmark, run_supervised_benchmark, summarize_results
from ssl_bci_rebuild.utils import ensure_dir, save_json


def _load_best_run_root(best_summary_path: Path) -> Path:
    payload = json.loads(best_summary_path.read_text(encoding="utf-8"))
    return Path(payload["run_root"])


def _load_base_cfg(best_run_root: Path) -> dict:
    cfg_path = best_run_root / "config_used.json"
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _variant_cfg(base_cfg: dict, output_root: Path, variant: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["training"]["ssl_batch_size"] = 128
    cfg["training"]["finetune_batch_size"] = 32
    cfg["training"]["batch_size"] = 32
    cfg["training"]["model_selection"] = "final"
    cfg["training"]["augmentation_mode"] = "li_paper_mix"
    cfg["training"]["num_workers"] = 0

    if variant == "bnci_only":
        cfg["run_root"] = str(output_root / "ssl_bnci_only_li_aug")
        cfg["source_datasets"] = ["BNCI2014_001"]
        cfg["source_dataset_labels"] = {
            "BNCI2014_001": ["left_hand", "right_hand", "feet", "tongue"],
        }
        cfg.pop("source_dataset_subjects", None)
        return cfg

    if variant == "bnci_plus_stieger_s1":
        cfg["run_root"] = str(output_root / "ssl_bnci_plus_stieger_s1_li_aug")
        cfg["source_datasets"] = ["BNCI2014_001", "Stieger2021"]
        cfg["source_dataset_subjects"] = {"Stieger2021": [1]}
        cfg["source_dataset_labels"] = {
            "BNCI2014_001": ["left_hand", "right_hand", "feet", "tongue"],
            "Stieger2021": ["left_hand", "right_hand", "both_hand", "rest"],
        }
        return cfg

    raise ValueError(f"Unknown variant: {variant}")


def _run_selected_objectives(cfg: dict, variant: str, objectives: list[str]) -> list[dict]:
    rows: list[dict] = []
    run_root = Path(cfg["run_root"])
    ensure_dir(run_root)
    save_json(run_root / "config_used.json", cfg)

    if "supervised" in objectives:
        print(f"[RUN] {variant} objective=supervised", flush=True)
        sup_summary = summarize_results(
            run_supervised_benchmark(cfg),
            run_root / "supervised_summary.json",
        )
        rows.append(
            {
                "variant": variant,
                "objective": "supervised",
                "mean_accuracy": sup_summary["mean_accuracy"],
                "std_accuracy": sup_summary["std_accuracy"],
                "mean_kappa": sup_summary["mean_kappa"],
                "std_kappa": sup_summary["std_kappa"],
                "run_root": str(run_root),
            }
        )
        print(f"[DONE] {variant} objective=supervised mean_acc={sup_summary['mean_accuracy']:.6f}", flush=True)

    for objective in ("simclr", "vicreg", "masked"):
        if objective not in objectives:
            continue
        print(f"[RUN] {variant} objective={objective}", flush=True)
        summary = summarize_results(
            run_ssl_benchmark(cfg, objective),
            run_root / f"{objective}_summary.json",
        )
        rows.append(
            {
                "variant": variant,
                "objective": objective,
                "mean_accuracy": summary["mean_accuracy"],
                "std_accuracy": summary["std_accuracy"],
                "mean_kappa": summary["mean_kappa"],
                "std_kappa": summary["std_kappa"],
                "run_root": str(run_root),
            }
        )
        print(f"[DONE] {variant} objective={objective} mean_acc={summary['mean_accuracy']:.6f}", flush=True)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-summary", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=["bnci_only", "bnci_plus_stieger_s1", "both"],
        default="both",
    )
    parser.add_argument(
        "--objectives",
        default="supervised,simclr,vicreg,masked",
        help="Comma-separated objectives from: supervised,simclr,vicreg,masked",
    )
    args = parser.parse_args()

    base_run_root = _load_best_run_root(args.best_summary)
    base_cfg = _load_base_cfg(base_run_root)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = ensure_dir(Path("outputs") / f"repro_ssl_from_best_{stamp}")
    save_json(
        output_root / "base_supervised_source.json",
        {
            "best_summary": str(args.best_summary),
            "best_run_root": str(base_run_root),
        },
    )

    objective_list = [item.strip() for item in args.objectives.split(",") if item.strip()]
    allowed = {"supervised", "simclr", "vicreg", "masked"}
    unknown = [item for item in objective_list if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown objectives: {unknown}")

    variants = ["bnci_only", "bnci_plus_stieger_s1"] if args.variant == "both" else [args.variant]

    rows: list[dict] = []
    for variant in variants:
        cfg = _variant_cfg(base_cfg, output_root, variant)
        rows.extend(_run_selected_objectives(cfg, variant, objective_list))

    rows.sort(key=lambda row: (row["variant"], row["objective"]))
    out_csv = output_root / "ssl_comparison.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_json(output_root / "ssl_comparison.json", {"rows": rows})

    print(f"[OUT] {out_csv}", flush=True)


if __name__ == "__main__":
    main()
