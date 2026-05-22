from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


BENCHMARK_POLICIES: dict[str, dict[str, str]] = {
    "li_bciiv2a_train_session_loso": {
        "source_session": "0train",
        "target_session": "0train",
    },
    "bnci_cross_session_loso": {
        "source_session": "0train",
        "target_session": "1test",
    },
}


def _resolve_benchmark_policy(cfg: dict[str, Any]) -> dict[str, Any]:
    policy_cfg = cfg.get("benchmark_policy")
    if policy_cfg is None:
        return cfg

    if isinstance(policy_cfg, str):
        policy_name = policy_cfg
        policy_overrides: dict[str, Any] = {}
    else:
        policy_name = policy_cfg["name"]
        policy_overrides = {key: value for key, value in policy_cfg.items() if key != "name"}

    if policy_name not in BENCHMARK_POLICIES:
        raise ValueError(f"Unknown benchmark policy: {policy_name}")

    resolved_split = {**BENCHMARK_POLICIES[policy_name], **policy_overrides}
    split_cfg = dict(cfg.get("split", {}))
    for key, value in resolved_split.items():
        current = split_cfg.get(key)
        if current is not None and current != value:
            raise ValueError(
                f"Config split.{key}={current!r} conflicts with benchmark policy {policy_name} ({value!r})."
            )
        split_cfg[key] = value

    cfg["split"] = split_cfg
    cfg["benchmark_policy"] = {"name": policy_name, **resolved_split}
    return cfg


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return _resolve_benchmark_policy(cfg)
