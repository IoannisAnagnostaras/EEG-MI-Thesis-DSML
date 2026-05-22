from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _summary_row(label: str, objective: str, summary_path: Path) -> dict:
    payload = _read_json(summary_path)
    return {
        "label": label,
        "objective": objective,
        "mean_accuracy": float(payload["mean_accuracy"]),
        "std_accuracy": float(payload["std_accuracy"]),
        "mean_kappa": float(payload["mean_kappa"]),
        "std_kappa": float(payload["std_kappa"]),
        "n_subjects": len(payload.get("subjects", [])),
        "source_path": str(summary_path),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    outputs = root / "outputs"

    legacy_csv = outputs / "official_comparison_latest.csv"
    deep_csv = outputs / "repro_deep_supervised_sweep_20260404T205754Z" / "deep_supervised_leaderboard.csv"
    bnci_only_dir = outputs / "repro_ssl_from_best_20260404T220815Z" / "ssl_bnci_only_li_aug"
    bnci_plus_stieger_dir = outputs / "repro_ssl_from_best_20260404T220815Z" / "ssl_bnci_plus_stieger_s1_li_aug"

    rows: list[dict] = []

    legacy_rows = _read_csv(legacy_csv)
    legacy_target = None
    for row in legacy_rows:
        if row.get("run_dir") == "official_bnci_plus_stieger_s1_ssl_r2" and row.get("objective") == "simclr":
            legacy_target = row
            break
    if legacy_target is not None:
        rows.append(
            {
                "label": "legacy_stable_reference",
                "objective": "simclr",
                "mean_accuracy": float(legacy_target["mean_accuracy"]),
                "std_accuracy": "",
                "mean_kappa": float(legacy_target["mean_kappa"]),
                "std_kappa": "",
                "n_subjects": int(legacy_target["n_subjects"]),
                "source_path": legacy_target["path"],
            }
        )

    for row in _read_csv(deep_csv):
        run_root = Path(row["run_root"])
        if not run_root.is_absolute():
            run_root = (root / run_root).resolve()
        rows.append(
            {
                "label": f"deep_supervised::{row['run_name']}",
                "objective": "supervised",
                "mean_accuracy": float(row["mean_accuracy"]),
                "std_accuracy": float(row["std_accuracy"]),
                "mean_kappa": float(row["mean_kappa"]),
                "std_kappa": float(row["std_kappa"]),
                "n_subjects": 9,
                "source_path": str(run_root),
            }
        )

    for objective in ("supervised", "simclr", "vicreg", "masked"):
        rows.append(_summary_row("ssl_bnci_only_li_aug", objective, bnci_only_dir / f"{objective}_summary.json"))
        rows.append(
            _summary_row(
                "ssl_bnci_plus_stieger_s1_li_aug",
                objective,
                bnci_plus_stieger_dir / f"{objective}_summary.json",
            )
        )

    rows.sort(key=lambda item: (item["label"], item["objective"]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_csv = outputs / f"official_comparison_repro_locked_{stamp}.csv"
    out_json = outputs / f"official_comparison_repro_locked_{stamp}.json"

    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"[OUT] {out_csv}")
    print(f"[OUT] {out_json}")


if __name__ == "__main__":
    main()
