from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

if os.environ.get("FRICTION_STOPPING_V13_V14_SMOKE", "0").lower() in {"1", "true", "yes"}:
    os.environ.setdefault("FRICTION_STOPPING_SWEEP_SMOKE", "1")

import yaml

import run_friction_stopping_rma_v10_v11_v12 as base


V13_REPEATS = 5
V14_REPEATS = 3
V13_FINAL_DISTANCE = (0.9, 1.2)
V14_FINAL_DISTANCE = (0.8, 1.1)


def _run_v14_enabled() -> bool:
    return os.environ.get("FRICTION_STOPPING_RUN_V14", "0").lower() in {"1", "true", "yes"}


def _v10_like_variant(version: str) -> base.Variant:
    return base.Variant(
        version=version,
        action_cost_weight=0.15,
        safety_cost_weight=12.0,
        crash_penalty=250.0,
        crash_remaining_penalty=250.0,
        calibration_noise_std=[0.15, 0.60],
    )


def _variants() -> list[tuple[base.Variant, tuple[float, float]]]:
    v13_count = 1 if base.smoke_enabled() else V13_REPEATS
    out: list[tuple[base.Variant, tuple[float, float]]] = [
        (_v10_like_variant(f"v13_{idx}"), V13_FINAL_DISTANCE) for idx in range(1, v13_count + 1)
    ]
    if _run_v14_enabled():
        v14_count = 1 if base.smoke_enabled() else V14_REPEATS
        out.extend((_v10_like_variant(f"v14_{idx}"), V14_FINAL_DISTANCE) for idx in range(1, v14_count + 1))
    return out


def _patch_manifest_final_distance(run_dir: Path, final_distance: tuple[float, float]) -> None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return
    with manifest_path.open("r") as f:
        payload: dict[str, Any] = json.load(f)
    payload["final_visible_distance"] = [float(final_distance[0]), float(final_distance[1])]
    payload["phase_distance_ranges"] = dict(payload.get("phase_distance_ranges", {}))
    payload["phase_distance_ranges"]["final"] = [float(final_distance[0]), float(final_distance[1])]
    with manifest_path.open("w") as f:
        json.dump(payload, f, indent=2)


def _run_with_final_distance(
    base_cfg: dict[str, Any],
    original_config: str,
    variant: base.Variant,
    final_distance: tuple[float, float],
) -> Path:
    old_final = tuple(base.PHASE_DISTANCE_RANGES["final"])
    old_env_overrides = base._env_overrides

    def _env_overrides_with_final_distance(active_variant: base.Variant) -> dict[str, Any]:
        overrides = old_env_overrides(active_variant)
        overrides["visible_distance_low"] = float(final_distance[0])
        overrides["visible_distance_high"] = float(final_distance[1])
        return overrides

    base.PHASE_DISTANCE_RANGES["final"] = tuple(final_distance)
    base._env_overrides = _env_overrides_with_final_distance
    try:
        run_dir = base._run_variant(base_cfg, original_config, variant)
        _patch_manifest_final_distance(run_dir, final_distance)
        return run_dir
    finally:
        base.PHASE_DISTANCE_RANGES["final"] = old_final
        base._env_overrides = old_env_overrides


def main() -> None:
    original_config = base.CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_config)
    completed: list[str] = []
    try:
        for variant, final_distance in _variants():
            run_dir = _run_with_final_distance(base_cfg, original_config, variant, final_distance)
            completed.append(str(run_dir.resolve()))
    finally:
        base.CONFIG_PATH.write_text(original_config)

    print("Friction-stopping v13/v14 repeat runner finished.", flush=True)
    for run_dir in completed:
        print(f"  {run_dir}", flush=True)


if __name__ == "__main__":
    main()
