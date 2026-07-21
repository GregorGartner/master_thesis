from __future__ import annotations

import os
from pathlib import Path
from typing import Any

if os.environ.get("FRICTION_STOPPING_V15_SMOKE", "0").lower() in {"1", "true", "yes"}:
    os.environ.setdefault("FRICTION_STOPPING_SWEEP_SMOKE", "1")

import yaml

import run_friction_stopping_rma_v10_v11_v12 as base


V15_REPEATS = 3
V15_SIGMA_V = 0.35


def _v15_variant(version: str) -> base.Variant:
    return base.Variant(
        version=version,
        action_cost_weight=0.15,
        safety_cost_weight=12.0,
        crash_penalty=250.0,
        crash_remaining_penalty=250.0,
        calibration_noise_std=[0.15, 0.60],
    )


def _variants() -> list[base.Variant]:
    count = 1 if base.smoke_enabled() else V15_REPEATS
    return [_v15_variant(f"v15_{idx}") for idx in range(1, count + 1)]


def _run_variant_with_tighter_speed_reward(
    base_cfg: dict[str, Any],
    original_config: str,
    variant: base.Variant,
) -> Path:
    old_env_overrides = base._env_overrides

    def _env_overrides_with_sigma_v(active_variant: base.Variant) -> dict[str, Any]:
        overrides = old_env_overrides(active_variant)
        overrides["sigma_v"] = float(V15_SIGMA_V)
        return overrides

    base._env_overrides = _env_overrides_with_sigma_v
    old_ignore = os.environ.get("FRICTION_STOPPING_SWEEP_IGNORE_GATE")
    os.environ["FRICTION_STOPPING_SWEEP_IGNORE_GATE"] = "1"
    try:
        return base._run_variant(base_cfg, original_config, variant)
    finally:
        base._env_overrides = old_env_overrides
        if old_ignore is None:
            os.environ.pop("FRICTION_STOPPING_SWEEP_IGNORE_GATE", None)
        else:
            os.environ["FRICTION_STOPPING_SWEEP_IGNORE_GATE"] = old_ignore


def main() -> None:
    original_config = base.CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_config)
    completed: list[str] = []
    try:
        for variant in _variants():
            run_dir = _run_variant_with_tighter_speed_reward(base_cfg, original_config, variant)
            completed.append(str(run_dir.resolve()))
    finally:
        base.CONFIG_PATH.write_text(original_config)

    print("Friction-stopping v15 repeat runner finished.", flush=True)
    for run_dir in completed:
        print(f"  {run_dir}", flush=True)


if __name__ == "__main__":
    main()
