from __future__ import annotations

import copy
import os
from datetime import datetime
from pathlib import Path

import yaml

from run_two_action_pipeline import (
    CONFIG_PATH,
    TWO_ACTION_LQR,
    _base_stage_cfg,
    _resolve_exp_root,
    _run_with_config,
    _snapshot_phase_weights,
)


RUN_SUFFIX = "two_action_recurrent64_policy"
ARCH = [64, 64]
LSTM_HIDDEN_SIZE = 64

PHASES = [
    ("P01", 0.02, 3e-4, 2_000_000),
    ("P02", 0.01, 3e-4, 3_000_000),
    ("P03", 0.005, 2e-4, 3_000_000),
    ("P04", 0.001, 1e-4, 2_000_000),
]


def _smoke_enabled() -> bool:
    return os.environ.get("TWO_ACTION_RECURRENT_SMOKE", "0").lower() in {"1", "true", "yes"}


def _selected_phases() -> list[tuple[str, float, float, int]]:
    if _smoke_enabled():
        return [
            ("P01", 0.05, 3e-4, 512),
            ("P02", 0.02, 3e-4, 512),
        ]
    return PHASES


def _set_recurrent_policy(cfg: dict, *, ent: float, lr: float) -> None:
    n_steps = 128 if _smoke_enabled() else 4096
    batch_size = 64 if _smoke_enabled() else 1024
    cfg["model"]["name"] = "RecurrentPPO"
    cfg["model"]["params"] = {
        "policy": "MlpLstmPolicy",
        "learning_rate": float(lr),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": 8,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "normalize_advantage": True,
        "ent_coef": float(ent),
        "vf_coef": 0.5,
        "max_grad_norm": 1.0,
        "target_kl": None,
        "verbose": 1,
        "seed": 0,
        "device": "cpu",
        "policy_kwargs": {
            "lstm_hidden_size": LSTM_HIDDEN_SIZE,
            "n_lstm_layers": 1,
            "net_arch": {
                "pi": list(ARCH),
                "vf": list(ARCH),
            },
        },
    }


def _set_previous_action_observation(cfg: dict) -> None:
    wrappers = []
    for wrapper in cfg.get("wrappers", []):
        wrappers.append(copy.deepcopy(wrapper))
    wrappers.append(
        {
            "name": "PreviousActionObservationWrapper",
            "enabled": True,
            "params": {},
        }
    )
    cfg["wrappers"] = wrappers


def _set_callbacks(cfg: dict) -> None:
    for cb in cfg.get("callbacks", []):
        if cb.get("name") == "LivePlotCallback":
            cb["enabled"] = True
        if cb.get("name") == "SaveModelCallback":
            cb.setdefault("params", {})
            cb["params"]["best_metric"] = "episode_reward"
            cb["params"]["save_on_training_end"] = True


def _write_pipeline_file(exp_dir: Path, payload: dict) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "two_action_recurrent_policy.yaml", "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _run_stage(label: str, cfg: dict, exp_dir: Path, payload: dict) -> None:
    print(
        f"START {label}: {cfg['training']['experiment_name']} "
        f"steps={cfg['total_timesteps']}",
        flush=True,
    )
    _write_pipeline_file(exp_dir, payload)
    _run_with_config(cfg)
    print(f"END   {label}: {cfg['training']['experiment_name']}", flush=True)


def main() -> None:
    original_text = CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_text)
    exp_root = _resolve_exp_root(base_cfg)
    stamp = datetime.now().strftime("%m-%d__%H-%M")
    phases = _selected_phases()
    suffix = f"{RUN_SUFFIX}_smoke" if _smoke_enabled() else RUN_SUFFIX

    exp_name = f"s_{stamp}_{suffix}"
    exp_dir = exp_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": stamp,
        "experiment": exp_name,
        "two_action_lqr": TWO_ACTION_LQR,
        "observation": "state plus previous action",
        "model": "RecurrentPPO/MlpLstmPolicy",
        "arch": ARCH,
        "lstm_hidden_size": LSTM_HIDDEN_SIZE,
        "phases": [
            {"phase": phase, "ent_coef": ent, "learning_rate": lr, "timesteps": steps}
            for phase, ent, lr, steps in phases
        ],
    }

    try:
        for phase_idx, (phase, ent, lr, steps) in enumerate(phases):
            cfg = _base_stage_cfg(base_cfg, exp_root, exp_name, steps)
            if _smoke_enabled():
                cfg["lqr_env"]["max_episode_steps"] = 64
            _set_recurrent_policy(cfg, ent=ent, lr=lr)
            _set_previous_action_observation(cfg)
            _set_callbacks(cfg)
            cfg["training"]["load_weights"] = phase_idx > 0
            cfg["training"]["load_weights_from"] = str(exp_dir.resolve()) if phase_idx > 0 else None
            cfg["training"]["load_weights_name"] = "weights"
            cfg["training"]["load_encoder_only"] = False

            _run_stage(f"recurrent/{phase}", cfg, exp_dir, payload)
            _snapshot_phase_weights(exp_dir, phase)

    finally:
        _write_pipeline_file(exp_dir, payload)
        CONFIG_PATH.write_text(original_text)


if __name__ == "__main__":
    main()
