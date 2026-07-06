from __future__ import annotations

import copy
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


SCHEDULES = [
    ("dense_strong_id", [2.0, 1.0, 0.5, 0.25, 0.225, 0.2, 0.175, 0.15, 0.125, 0.1]),
]
PHASES = [
    ("P01", 0.2, 8e-4, 1_500_000),
    ("P02", 0.1, 6e-4, 1_500_000),
    ("P03", 0.05, 4e-4, 1_500_000),
    ("P04", 0.02, 3e-4, 2_000_000),
    ("P05", 0.01, 3e-4, 2_000_000),
    ("P06", 0.01, 2e-4, 2_000_000),
    ("P07", 0.005, 2e-4, 2_000_000),
    ("P08", 0.005, 1e-4, 2_000_000),
    ("P09", 0.0, 1e-4, 2_000_000),
    ("P10", 0.0, 1e-4, 2_000_000),
]


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
TRAIN_CMD = ["python3", str(ROOT / "cartpole_ppo_sb3_training.py")]


def _set_common_policy_phase(cfg: dict, penalty: float) -> None:
    params = cfg["model"]["params"]
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["uncertainty_regularization_coef"] = 0.0
    params["condition_on_uncertainty"] = True
    params["privileged_uncertainty_mode"] = "predicted"
    params["naive_action_noise_std"] = 0.0
    params["naive_action_noise_dist"] = "gaussian"
    params["deterministic_actions"] = False
    params["uncertainty_reward_penalty_coef"] = float(penalty)
    for cb in cfg.get("callbacks", []):
        if cb.get("name") == "SaveModelCallback":
            cb.setdefault("params", {})
            cb["params"]["best_metric"] = "episode_reward"


def _run_with_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    subprocess.run(TRAIN_CMD, cwd=ROOT, check=True)


def _write_schedule_file(exp_dir: Path, schedule_name: str, penalties: list[float], base_load_from: str) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schedule_name": schedule_name,
        "base_load_from": base_load_from,
        "phase_handoff": "P01 loads encoder only from base_load_from; later phases load weights.zip from this experiment.",
        "phases": [
            {
                "phase": phase_name,
                "timesteps": int(steps),
                "learning_rate": float(lr),
                "ent_coef": float(ent),
                "uncertainty_reward_penalty_coef": float(penalties[idx]),
            }
            for idx, (phase_name, ent, lr, steps) in enumerate(PHASES)
        ],
    }
    with open(exp_dir / "matrix_schedule.yaml", "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _snapshot_phase_weights(exp_dir: Path, phase_name: str) -> None:
    for name in ["weights", "weights_best"]:
        src = exp_dir / f"{name}.zip"
        if src.exists():
            shutil.copy2(src, exp_dir / f"{name}_{phase_name}.zip")
    metric = exp_dir / "weights_best.metric"
    if metric.exists():
        shutil.copy2(metric, exp_dir / f"weights_best_{phase_name}.metric")


def main() -> None:
    original_text = CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_text)
    training = base_cfg["training"]
    base_load_from = training.get("load_weights_from")
    if not base_load_from:
        raise ValueError("config.training.load_weights_from must point to your pretrained encoder/policy run.")
    base_load_from = str((ROOT / base_load_from).resolve()) if not str(base_load_from).startswith("/") else str(base_load_from)
    exp_root = str((ROOT / training["experiment_root"]).resolve())
    stamp = datetime.now().strftime("%m-%d__%H-%M")
    total_jobs = len(SCHEDULES) * len(PHASES)
    job_idx = 0

    def run_job(label: str, cfg: dict) -> None:
        nonlocal job_idx
        job_idx += 1
        print(f"[{job_idx}/{total_jobs}] START {label}", flush=True)
        _run_with_config(cfg)
        print(f"[{job_idx}/{total_jobs}] END   {label}", flush=True)

    try:
        for schedule_name, penalties in SCHEDULES:
            exp_name = f"s_{stamp}_random_curr_{schedule_name}"
            exp_dir = Path(exp_root) / exp_name
            _write_schedule_file(exp_dir, schedule_name, penalties, base_load_from)

            for phase_idx, (phase_name, ent, lr, steps) in enumerate(PHASES):
                penalty = penalties[phase_idx]
                cfg = copy.deepcopy(base_cfg)
                _set_common_policy_phase(cfg, penalty)
                cfg["training"]["experiment_root"] = exp_root
                cfg["training"]["experiment_name"] = exp_name
                cfg["training"]["load_weights"] = True
                cfg["training"]["load_weights_from"] = base_load_from if phase_idx == 0 else str((Path(exp_root) / exp_name).resolve())
                cfg["training"]["load_weights_name"] = "weights_best" if phase_idx == 0 else "weights"
                cfg["training"]["load_encoder_only"] = phase_idx == 0
                cfg["total_timesteps"] = int(steps)
                cfg["model"]["params"]["ent_coef"] = float(ent)
                cfg["model"]["params"]["learning_rate"] = float(lr)
                run_job(
                    f"{schedule_name}/{phase_name} pen={penalty:g} "
                    f"lr={lr:g} ent={ent:g} steps={steps}",
                    cfg,
                )
                _snapshot_phase_weights(exp_dir, phase_name)
    finally:
        CONFIG_PATH.write_text(original_text)


if __name__ == "__main__":
    main()
