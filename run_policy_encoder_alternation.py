from __future__ import annotations

import copy
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


PHASES = [
    ("P01", "policy", 0.50, 0.03, 3e-4, 3_000_000, None),
    ("E01", "encoder_mixed", 0.0, 0.0, 1e-5, 500_000, [0.0, 0.5]),
    ("P02", "policy", 0.35, 0.03, 3e-4, 3_000_000, None),
    ("E02", "encoder_mixed", 0.0, 0.0, 1e-5, 500_000, [0.0, 0.5]),
    ("P03", "policy", 0.25, 0.02, 2e-4, 3_000_000, None),
    ("E03", "encoder_mixed", 0.0, 0.0, 1e-5, 500_000, [0.0, 0.5]),
    ("P04", "policy", 0.20, 0.02, 2e-4, 3_000_000, None),
    ("E04", "encoder_mixed", 0.0, 0.0, 1e-5, 500_000, [0.0, 0.5]),
    ("P05", "policy", 0.175, 0.02, 2e-4, 3_000_000, None),
    ("E05", "encoder_mixed", 0.0, 0.0, 1e-5, 500_000, [0.0, 0.5]),
    ("P06", "policy", 0.15, 0.01, 1e-4, 4_000_000, None),
    ("E06", "encoder_mixed", 0.0, 0.0, 1e-5, 500_000, [0.0, 0.5]),
    ("P07", "policy", 0.125, 0.01, 1e-4, 4_000_000, None),
]


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
TRAIN_CMD = ["python3", str(ROOT / "cartpole_ppo_sb3_training.py")]


def _set_policy_phase(cfg: dict, penalty: float, ent: float, lr: float) -> None:
    params = cfg["model"]["params"]
    _set_normal_training_params(params)
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["uncertainty_regularization_coef"] = 0.0
    params["condition_on_uncertainty"] = True
    params["privileged_uncertainty_mode"] = "predicted"
    params["naive_action_noise_std"] = 0.0
    params["naive_action_noise_dist"] = "gaussian"
    params["deterministic_actions"] = False
    params["uncertainty_reward_penalty_coef"] = float(penalty)
    params["uncertainty_penalty_metric"] = "std"
    params["ent_coef"] = float(ent)
    params["learning_rate"] = float(lr)


def _set_encoder_phase(cfg: dict, noise_std: list[float] | None, lr: float) -> None:
    params = cfg["model"]["params"]
    _set_normal_training_params(params)
    params["freeze_ppo"] = True
    params["regression_coef"] = 1.0
    params["uncertainty_regularization_coef"] = 0.0
    params["condition_on_uncertainty"] = True
    params["privileged_uncertainty_mode"] = "predicted"
    params["naive_action_noise_std"] = 0.0 if noise_std is None else noise_std
    params["naive_action_noise_dist"] = "gaussian" if noise_std is None else ["gaussian", "uniform"]
    params["deterministic_actions"] = False
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["uncertainty_penalty_metric"] = "std"
    params["ent_coef"] = 0.0
    params["learning_rate"] = float(lr)


def _set_normal_training_params(params: dict) -> None:
    params["n_steps"] = 4096
    params["batch_size"] = 1024
    params["n_epochs"] = 8
    params["clip_range"] = 0.2
    params["max_grad_norm"] = 1.0
    params["verbose"] = 1


def _set_common_callbacks(cfg: dict) -> None:
    for cb in cfg.get("callbacks", []):
        if cb.get("name") == "LivePlotCallback":
            cb["enabled"] = True
        if cb.get("name") == "SaveModelCallback":
            cb.setdefault("params", {})
            cb["params"]["best_metric"] = "episode_reward"


def _run_with_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")
    subprocess.run(TRAIN_CMD, cwd=ROOT, env=env, check=True)


def _snapshot_phase_weights(exp_dir: Path, phase_name: str) -> None:
    for name in ["weights", "weights_best"]:
        src = exp_dir / f"{name}.zip"
        if src.exists():
            shutil.copy2(src, exp_dir / f"{name}_{phase_name}.zip")
    metric = exp_dir / "weights_best.metric"
    if metric.exists():
        shutil.copy2(metric, exp_dir / f"weights_best_{phase_name}.metric")


def _write_schedule_file(exp_dir: Path, base_load_from: str, phases: list[tuple]) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schedule_name": "policy_encoder_alternation_std",
        "base_load_from": base_load_from,
        "phase_handoff": "P01 loads encoder only from base_load_from; all later phases load weights.zip from this experiment.",
        "phases": [
            {
                "phase": name,
                "kind": kind,
                "timesteps": int(steps),
                "learning_rate": float(lr),
                "ent_coef": float(ent),
                "uncertainty_reward_penalty_coef": float(penalty),
                "uncertainty_penalty_metric": "std",
                "naive_action_noise_std": noise_std,
            }
            for name, kind, penalty, ent, lr, steps, noise_std in phases
        ],
    }
    with open(exp_dir / "matrix_schedule.yaml", "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main() -> None:
    original_text = CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_text)
    training = base_cfg["training"]
    base_load_from = training.get("load_weights_from")
    if not base_load_from:
        raise ValueError("config.training.load_weights_from must point to your pretrained encoder run.")
    base_load_from = str((ROOT / base_load_from).resolve()) if not str(base_load_from).startswith("/") else str(base_load_from)
    exp_root = str((ROOT / training["experiment_root"]).resolve())
    smoke = os.environ.get("ALT_SMOKE_TEST", "0").lower() in {"1", "true", "yes"}
    phases = PHASES
    if smoke:
        smoke_steps = int(os.environ.get("ALT_SMOKE_STEPS", "64"))
        phases = [
            (name, kind, penalty, ent, lr, smoke_steps, noise_std)
            for name, kind, penalty, ent, lr, _steps, noise_std in PHASES[:4]
        ]
    exp_name = (
        f"smoke_{datetime.now().strftime('%m-%d__%H-%M-%S')}_alt_encoder_std"
        if smoke
        else f"s_{datetime.now().strftime('%m-%d__%H-%M')}_random_curr_alt_encoder_std"
    )
    exp_dir = Path(exp_root) / exp_name
    _write_schedule_file(exp_dir, base_load_from, phases)

    try:
        for phase_idx, (name, kind, penalty, ent, lr, steps, noise_std) in enumerate(phases):
            cfg = copy.deepcopy(base_cfg)
            if kind == "policy":
                _set_policy_phase(cfg, penalty=penalty, ent=ent, lr=lr)
            else:
                _set_encoder_phase(cfg, noise_std=noise_std, lr=lr)
            _set_common_callbacks(cfg)
            cfg["training"]["experiment_root"] = exp_root
            cfg["training"]["experiment_name"] = exp_name
            cfg["training"]["load_weights"] = True
            cfg["training"]["load_weights_from"] = base_load_from if phase_idx == 0 else str(exp_dir.resolve())
            cfg["training"]["load_weights_name"] = "weights_best" if phase_idx == 0 else "weights"
            cfg["training"]["load_encoder_only"] = phase_idx == 0
            cfg["total_timesteps"] = int(steps)
            if smoke:
                params = cfg["model"]["params"]
                params["n_steps"] = int(steps)
                params["batch_size"] = int(steps)
                params["n_epochs"] = 1
                for cb in cfg.get("callbacks", []):
                    if cb.get("name") == "LivePlotCallback":
                        cb["enabled"] = False
            print(
                f"[{phase_idx + 1}/{len(phases)}] START {name} kind={kind} "
                f"pen={penalty:g} lr={lr:g} ent={ent:g} steps={steps} noise={noise_std}",
                flush=True,
            )
            _run_with_config(cfg)
            print(f"[{phase_idx + 1}/{len(phases)}] END   {name}", flush=True)
            _snapshot_phase_weights(exp_dir, name)
    finally:
        CONFIG_PATH.write_text(original_text)


if __name__ == "__main__":
    main()
