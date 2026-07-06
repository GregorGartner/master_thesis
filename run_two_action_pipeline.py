from __future__ import annotations

import copy
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
TRAIN_CMD = ["python3", str(ROOT / "cartpole_ppo_sb3_training.py")]

RUN_PRIVILEGED = False
RUN_ENCODER = True
RUN_NO_UNCERTAINTY_BASELINE = True
RUN_UNCERTAINTY_POLICY = True

RESUME_PRIVILEGED_FROM = "s_05-13__11-43_two_action_privileged"

TWO_ACTION_LQR = {
    "A": [[0.98, 0.0], [0.0, 0.95]],
    "B": [[0.1, 0.03], [1.0, 0.2]],
    "delta_B": [[0.5, 0.02], [0.15, 1.0]],
    "Q": [[2.0, 0.0], [0.0, 1.0]],
    "R": [[5.0, 0.0], [0.0, 0.2]],
}

PRIVILEGED_STEPS = 8_000_000
ENCODER_STEPS = 6_000_000

NO_UNC_PHASES = [
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

UNC_PHASES = [
    ("P01", 0.50, 0.03, 3e-4, 3_000_000),
    ("P02", 0.35, 0.03, 3e-4, 3_000_000),
    ("P03", 0.25, 0.02, 2e-4, 3_000_000),
    ("P04", 0.20, 0.02, 2e-4, 3_000_000),
    ("P05", 0.175, 0.02, 2e-4, 3_000_000),
    ("P06", 0.15, 0.01, 1e-4, 4_000_000),
    ("P07", 0.125, 0.01, 1e-4, 4_000_000),
]


def _normal_training_params(params: dict) -> None:
    params["n_steps"] = 4096
    params["batch_size"] = 1024
    params["n_epochs"] = 8
    params["clip_range"] = 0.2
    params["max_grad_norm"] = 1.0
    params["verbose"] = 1


def _set_two_action_lqr(cfg: dict) -> None:
    lqr = cfg.setdefault("lqr_env", {})
    lqr.update(copy.deepcopy(TWO_ACTION_LQR))
    lqr.setdefault("process_noise_std", 0.05)
    lqr.setdefault("initial_state_low", -0.3)
    lqr.setdefault("initial_state_high", 0.3)
    lqr.setdefault("max_episode_steps", 512)


def _set_common_model_params(cfg: dict) -> None:
    params = cfg["model"]["params"]
    _normal_training_params(params)
    params["policy"] = "MlpPolicy"
    params["regression_param_names"] = ["theta"]
    params["latent_dim"] = 1
    params["window_length"] = 50
    params["id_update_interval"] = 10
    params["nominal_warmup_steps"] = 49
    params["z_scale"] = 10.0
    params["use_transition_features"] = True
    params["transition_type"] = "delta"
    params["encoder_type"] = "mlp"
    params["encoder_net_arch"] = [64, 64]
    params["actor_net_arch"] = [64, 64]
    params["critic_net_arch"] = [64, 64]
    params["detach_context_for_rl"] = True
    params["deterministic_actions"] = False
    params["naive_action_noise_std"] = 0.0
    params["naive_action_noise_dist"] = "gaussian"


def _set_callbacks(cfg: dict, best_metric: str) -> None:
    for cb in cfg.get("callbacks", []):
        if cb.get("name") == "LivePlotCallback":
            cb["enabled"] = True
        if cb.get("name") == "SaveModelCallback":
            cb.setdefault("params", {})
            cb["params"]["best_metric"] = best_metric
            cb["params"]["save_on_training_end"] = True


def _set_privileged(cfg: dict) -> None:
    params = cfg["model"]["params"]
    _set_common_model_params(cfg)
    params["context_mode"] = "privileged"
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["condition_on_uncertainty"] = True
    params["privileged_uncertainty_mode"] = "zeros"
    params["privileged_uncertainty_value"] = 0.0
    params["uncertainty_regularization_coef"] = 0.0
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["uncertainty_penalty_metric"] = "std"
    params["learning_rate"] = 3e-4
    params["ent_coef"] = 0.0
    _set_callbacks(cfg, best_metric="episode_reward")


def _set_encoder(cfg: dict) -> None:
    params = cfg["model"]["params"]
    _set_common_model_params(cfg)
    params["context_mode"] = "encoder_nll"
    params["freeze_ppo"] = True
    params["regression_coef"] = 1.0
    params["condition_on_uncertainty"] = True
    params["privileged_uncertainty_mode"] = "predicted"
    params["uncertainty_regularization_coef"] = 0.0
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["uncertainty_penalty_metric"] = "std"
    params["naive_action_noise_std"] = [0.0, 0.5]
    params["naive_action_noise_dist"] = ["gaussian", "uniform"]
    params["learning_rate"] = 1e-5
    params["ent_coef"] = 0.0
    _set_callbacks(cfg, best_metric="regression_mse")


def _set_no_unc_policy(cfg: dict, ent: float, lr: float) -> None:
    params = cfg["model"]["params"]
    _set_common_model_params(cfg)
    params["context_mode"] = "encoder_nll"
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["condition_on_uncertainty"] = False
    params["privileged_uncertainty_mode"] = "zeros"
    params["uncertainty_regularization_coef"] = 0.0
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["uncertainty_penalty_metric"] = "std"
    params["learning_rate"] = float(lr)
    params["ent_coef"] = float(ent)
    _set_callbacks(cfg, best_metric="episode_reward")


def _set_unc_policy(cfg: dict, penalty: float, ent: float, lr: float) -> None:
    params = cfg["model"]["params"]
    _set_common_model_params(cfg)
    params["context_mode"] = "encoder_nll"
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["condition_on_uncertainty"] = True
    params["privileged_uncertainty_mode"] = "predicted"
    params["uncertainty_regularization_coef"] = 0.0
    params["uncertainty_reward_penalty_coef"] = float(penalty)
    params["uncertainty_penalty_metric"] = "std"
    params["learning_rate"] = float(lr)
    params["ent_coef"] = float(ent)
    _set_callbacks(cfg, best_metric="episode_reward")


def _resolve_exp_root(base_cfg: dict) -> Path:
    exp_root = Path(str(base_cfg["training"]["experiment_root"]))
    return exp_root if exp_root.is_absolute() else (ROOT / exp_root).resolve()


def _run_with_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
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


def _base_stage_cfg(base_cfg: dict, exp_root: Path, exp_name: str, steps: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["environment"] = "lqr"
    cfg["total_timesteps"] = int(steps)
    cfg["training"]["experiment_root"] = str(exp_root)
    cfg["training"]["experiment_name"] = exp_name
    cfg["model"]["name"] = "UnifiedContextPPO"
    _set_two_action_lqr(cfg)
    return cfg


def _write_pipeline_file(root_dir: Path, payload: dict) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)
    with open(root_dir / "two_action_pipeline.yaml", "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main() -> None:
    original_text = CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_text)
    exp_root = _resolve_exp_root(base_cfg)
    stamp = datetime.now().strftime("%m-%d__%H-%M")

    names = {
        "privileged": f"s_{stamp}_two_action_privileged",
        "encoder": f"s_{stamp}_two_action_nll_encoder_rand_exp",
        "no_unc": f"s_{stamp}_two_action_no_uncertainty_baseline",
        "unc": f"s_{stamp}_two_action_uncertainty_policy",
    }
    if RESUME_PRIVILEGED_FROM:
        names["privileged"] = RESUME_PRIVILEGED_FROM
    dirs = {key: exp_root / value for key, value in names.items()}

    payload = {
        "timestamp": stamp,
        "two_action_lqr": TWO_ACTION_LQR,
        "stages": [],
    }
    pipeline_dir = exp_root / f"s_{stamp}_two_action_pipeline"

    def run_stage(label: str, cfg: dict) -> None:
        print(f"START {label}: {cfg['training']['experiment_name']} steps={cfg['total_timesteps']}", flush=True)
        _write_pipeline_file(pipeline_dir, payload)
        _run_with_config(cfg)
        print(f"END   {label}: {cfg['training']['experiment_name']}", flush=True)

    try:
        if RUN_PRIVILEGED:
            cfg = _base_stage_cfg(base_cfg, exp_root, names["privileged"], PRIVILEGED_STEPS)
            _set_privileged(cfg)
            if RESUME_PRIVILEGED_FROM:
                if not (dirs["privileged"] / "weights.zip").exists():
                    raise FileNotFoundError(f"Cannot resume privileged run, missing weights.zip in {dirs['privileged']}")
                cfg["training"]["load_weights"] = True
                cfg["training"]["load_weights_from"] = str(dirs["privileged"].resolve())
                cfg["training"]["load_weights_name"] = "weights"
            else:
                cfg["training"]["load_weights"] = False
                cfg["training"]["load_weights_from"] = None
            cfg["training"]["load_encoder_only"] = False
            payload["stages"].append({"name": "privileged", "experiment": names["privileged"], "timesteps": PRIVILEGED_STEPS})
            run_stage("privileged", cfg)

        privileged_dir = dirs["privileged"]
        if RUN_ENCODER:
            cfg = _base_stage_cfg(base_cfg, exp_root, names["encoder"], ENCODER_STEPS)
            _set_encoder(cfg)
            cfg["training"]["load_weights"] = True
            cfg["training"]["load_weights_from"] = str(privileged_dir.resolve())
            cfg["training"]["load_weights_name"] = "weights"
            cfg["training"]["load_encoder_only"] = False
            payload["stages"].append({
                "name": "encoder_nll",
                "experiment": names["encoder"],
                "load_weights_from": str(privileged_dir.resolve()),
                "timesteps": ENCODER_STEPS,
                "naive_action_noise_std": [0.0, 0.5],
                "naive_action_noise_dist": ["gaussian", "uniform"],
            })
            run_stage("encoder_nll", cfg)

        encoder_dir = dirs["encoder"]
        if RUN_NO_UNCERTAINTY_BASELINE:
            exp_dir = dirs["no_unc"]
            exp_dir.mkdir(parents=True, exist_ok=True)
            payload["stages"].append({
                "name": "no_uncertainty_baseline",
                "experiment": names["no_unc"],
                "load_encoder_from": str(encoder_dir.resolve()),
                "phases": [
                    {"phase": p, "ent_coef": ent, "learning_rate": lr, "timesteps": steps}
                    for p, ent, lr, steps in NO_UNC_PHASES
                ],
            })
            for phase_idx, (phase, ent, lr, steps) in enumerate(NO_UNC_PHASES):
                cfg = _base_stage_cfg(base_cfg, exp_root, names["no_unc"], steps)
                _set_no_unc_policy(cfg, ent=ent, lr=lr)
                cfg["training"]["load_weights"] = True
                cfg["training"]["load_weights_from"] = str(encoder_dir.resolve()) if phase_idx == 0 else str(exp_dir.resolve())
                cfg["training"]["load_weights_name"] = "weights_best" if phase_idx == 0 else "weights"
                cfg["training"]["load_encoder_only"] = phase_idx == 0
                run_stage(f"no_uncertainty/{phase}", cfg)
                _snapshot_phase_weights(exp_dir, phase)

        if RUN_UNCERTAINTY_POLICY:
            exp_dir = dirs["unc"]
            exp_dir.mkdir(parents=True, exist_ok=True)
            payload["stages"].append({
                "name": "uncertainty_policy",
                "experiment": names["unc"],
                "load_encoder_from": str(encoder_dir.resolve()),
                "uncertainty_penalty_metric": "std",
                "phases": [
                    {"phase": p, "penalty": pen, "ent_coef": ent, "learning_rate": lr, "timesteps": steps}
                    for p, pen, ent, lr, steps in UNC_PHASES
                ],
            })
            for phase_idx, (phase, penalty, ent, lr, steps) in enumerate(UNC_PHASES):
                cfg = _base_stage_cfg(base_cfg, exp_root, names["unc"], steps)
                _set_unc_policy(cfg, penalty=penalty, ent=ent, lr=lr)
                cfg["training"]["load_weights"] = True
                cfg["training"]["load_weights_from"] = str(encoder_dir.resolve()) if phase_idx == 0 else str(exp_dir.resolve())
                cfg["training"]["load_weights_name"] = "weights_best" if phase_idx == 0 else "weights"
                cfg["training"]["load_encoder_only"] = phase_idx == 0
                run_stage(f"uncertainty/{phase}", cfg)
                _snapshot_phase_weights(exp_dir, phase)

    finally:
        _write_pipeline_file(pipeline_dir, payload)
        CONFIG_PATH.write_text(original_text)


if __name__ == "__main__":
    main()
