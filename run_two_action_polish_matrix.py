from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from run_two_action_pipeline import (
    CONFIG_PATH,
    TWO_ACTION_LQR,
    _base_stage_cfg,
    _resolve_exp_root,
    _run_with_config,
    _set_unc_policy,
)


SOURCE_RUN = "s_05-16__14-44_two_action_unc_cont_from_P05_pen0p15_ent0p002"
SOURCE_WEIGHTS = "weights"

BRANCHES = [
    ("pen0p15_ent0p002_lr2e4", 0.15, 0.002, 2e-4, 8_000_000),
    ("pen0p175_ent0p001_lr2e4", 0.175, 0.001, 2e-4, 8_000_000),
    ("pen0p20_ent0p001_lr2e4", 0.20, 0.001, 2e-4, 8_000_000),
    ("pen0p175_ent0_lr3e4", 0.175, 0.0, 3e-4, 6_000_000),
    ("pen0p20_ent0_lr3e4", 0.20, 0.0, 3e-4, 6_000_000),
]


def _write_schedule(schedule_dir: Path, payload: dict) -> None:
    schedule_dir.mkdir(parents=True, exist_ok=True)
    with open(schedule_dir / "matrix_schedule.yaml", "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main() -> None:
    original_text = CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_text)
    exp_root = _resolve_exp_root(base_cfg)
    source_dir = exp_root / SOURCE_RUN
    stamp = datetime.now().strftime("%m-%d__%H-%M")
    schedule_dir = exp_root / f"s_{stamp}_two_action_polish_matrix"

    payload = {
        "timestamp": stamp,
        "two_action_lqr": TWO_ACTION_LQR,
        "source_run": SOURCE_RUN,
        "source_weights": SOURCE_WEIGHTS,
        "purpose": "Continue from the best two-action uncertainty checkpoint with non-small learning rates and keep uncertainty pressure >= 0.15.",
        "branches": [],
    }

    try:
        if not (source_dir / f"{SOURCE_WEIGHTS}.zip").exists():
            raise FileNotFoundError(f"Missing source checkpoint: {source_dir / f'{SOURCE_WEIGHTS}.zip'}")

        for label, penalty, ent, lr, steps in BRANCHES:
            exp_name = f"s_{stamp}_two_action_polish_{label}"
            cfg = _base_stage_cfg(base_cfg, exp_root, exp_name, steps)
            _set_unc_policy(cfg, penalty=penalty, ent=ent, lr=lr)
            cfg["training"]["load_weights"] = True
            cfg["training"]["load_weights_from"] = str(source_dir.resolve())
            cfg["training"]["load_weights_name"] = SOURCE_WEIGHTS
            cfg["training"]["load_encoder_only"] = False

            payload["branches"].append({
                "experiment": exp_name,
                "source_run": SOURCE_RUN,
                "source_weights": SOURCE_WEIGHTS,
                "penalty": penalty,
                "ent_coef": ent,
                "learning_rate": lr,
                "timesteps": steps,
            })
            _write_schedule(schedule_dir, payload)
            print(f"START {exp_name}: penalty={penalty} ent={ent} lr={lr} steps={steps}", flush=True)
            _run_with_config(cfg)
            print(f"END   {exp_name}", flush=True)

    finally:
        _write_schedule(schedule_dir, payload)
        CONFIG_PATH.write_text(original_text)


if __name__ == "__main__":
    main()
