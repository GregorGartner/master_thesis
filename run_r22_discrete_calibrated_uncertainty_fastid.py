from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import run_r22_discrete_calibrated_uncertainty as base


FAST_ID_INTERVAL = 1
FAST_RUN_FAMILY_ROOT = (
    base.ROOT / "experiments" / "r22_discrete_calibrated_uncertainty_fastid"
)
FAST_POINTER = FAST_RUN_FAMILY_ROOT / "latest_r22_discrete_calibrated_uncertainty_fastid.txt"


def _patch_base_for_fast_id() -> None:
    old_apply = base._apply_standard_discrete_r22_settings
    old_write_manifest = base._write_manifest

    def _apply_fast_id_settings(cfg: dict[str, Any], system: dict[str, Any]) -> None:
        old_apply(cfg, system)
        cfg["model"]["params"]["id_update_interval"] = FAST_ID_INTERVAL

    def _write_fast_id_manifest(run_root: Path, payload: dict[str, Any]) -> None:
        setup = payload.setdefault("main_setup", {})
        setup["id_update_interval"] = FAST_ID_INTERVAL
        payload["variant"] = "fast_id_update"
        payload["base_runner"] = "run_r22_discrete_calibrated_uncertainty.py"
        old_write_manifest(run_root, payload)

    base._apply_standard_discrete_r22_settings = _apply_fast_id_settings
    base._write_manifest = _write_fast_id_manifest
    base.base._resolve_exp_root = lambda _cfg: (base.ROOT / "experiments").resolve()
    base.RUN_FAMILY_ROOT = FAST_RUN_FAMILY_ROOT
    base.POINTER = FAST_POINTER


def main() -> None:
    if os.environ.get("OVERNIGHT_SMOKE", "0").lower() in {"1", "true", "yes"}:
        os.environ.setdefault("R22_CALIBRATED_UNCERTAINTY_SMOKE", "1")
    _patch_base_for_fast_id()
    base.main()


if __name__ == "__main__":
    main()
