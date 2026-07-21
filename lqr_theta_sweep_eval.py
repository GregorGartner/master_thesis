from __future__ import annotations

import csv
import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from lqr_model import EmergencyBrakeEnv, LinearQuadraticEnv, StoppingCarEnv, install_lqr_adapter_for_domain_randomization
from scipy.linalg import solve_discrete_are
import torch as th
import yaml
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from tqdm import tqdm

from unified_context_ppo import UnifiedContextPPO, _flatten_obs
from wrappers import ForceMujocoFixedCamera, PreviousActionObservationWrapper, RewardWrapper


exp_dict = {
    1: "./experiments/s_05-01__16-24_nll_50_async_10_rand_exp",
    # 2: "./experiments/s_04-21__21-30_gls_50_async_10_no_unc_warmup",
    # 3: "./experiments/s_04-20__14-07_gls_50_async_10",
}


### ---------- Sweep configuration ---------- ###
TARGET_EXP_ID = 1
TARGET_EXPERIMENT = os.environ.get("THETA_SWEEP_TARGET_EXPERIMENT", exp_dict[TARGET_EXP_ID])
OUTPUT_SUBDIR = os.environ.get("THETA_SWEEP_OUTPUT_SUBDIR", "theta_sweep")
SWEEP_DETERMINISTIC = os.environ.get("THETA_SWEEP_DETERMINISTIC", "1").lower() not in {"0", "false", "no"}

# If None, derive the sweep bounds from the target experiment config.
THETA_GRID: Optional[np.ndarray] = None
N_THETA_POINTS = int(os.environ.get("THETA_SWEEP_N_THETA_POINTS", "41"))

# Run both a noiseless identification sweep and the nominal-noise sweep.
PROCESS_NOISE_SCALES = [1.0]
EPISODES_PER_THETA = int(os.environ.get("THETA_SWEEP_EPISODES_PER_THETA", "20"))
EVAL_BASE_SEED = int(os.environ.get("THETA_SWEEP_EVAL_BASE_SEED", "12345"))
CALIBRATION_NUM_BINS = 10
COVERAGE_KS = tuple(np.arange(0.1, 2.1, 0.1))

# Ignore the padded warm-up prefix when summarizing encoder predictions.
TAIL_FRACTION = 1.0
IGNORE_FIRST_STEPS: Optional[int] = None # if none, ignores the first `window_length` steps
SAVE_STEP_LEVEL_CSV = os.environ.get("THETA_SWEEP_SAVE_STEP_LEVEL_CSV", "1").lower() not in {"0", "false", "no"}
COLLECT_STEP_LEVEL_PREDICTIONS = (
    SAVE_STEP_LEVEL_CSV
    or os.environ.get("THETA_SWEEP_COLLECT_STEP_LEVEL_PREDICTIONS", "0").lower() in {"1", "true", "yes"}
)
SAVE_TRAJECTORY_TRACE = os.environ.get("THETA_SWEEP_SAVE_TRAJECTORY_TRACE", "0").lower() in {"1", "true", "yes"}
RETURN_MODE = os.environ.get("THETA_SWEEP_RETURN_MODE", "reward").lower()
if RETURN_MODE not in {"reward", "quadratic"}:
    raise ValueError("THETA_SWEEP_RETURN_MODE must be either 'reward' or 'quadratic'.")

# Controller-level scorecard thresholds. The tail metric is the most useful one
# for this LQR family because nominal/no-exploration RMA mostly loses away from
# theta=0, while the center checks whether exploration damages nominal control.
TAIL_ABS_THETA_THRESHOLD = float(os.environ.get("THETA_SWEEP_TAIL_ABS_THETA_THRESHOLD", "0.15"))
CENTER_ABS_THETA_THRESHOLD = float(os.environ.get("THETA_SWEEP_CENTER_ABS_THETA_THRESHOLD", "0.05"))
BASELINE_CONTROLLER_FOR_GAIN = os.environ.get(
    "THETA_SWEEP_BASELINE_CONTROLLER",
    "nominal_rma_no_exploration",
)
BASELINE_AGGREGATE_PATH = os.environ.get("THETA_SWEEP_BASELINE_AGGREGATE_PATH")


### ---------- Controllers to compare ---------- ###
_controller_specs_json = os.environ.get("THETA_SWEEP_CONTROLLER_SPECS_JSON")
_single_ppo_exp = os.environ.get("THETA_SWEEP_SINGLE_PPO_EXPERIMENT")
if _controller_specs_json:
    CONTROLLER_SPECS = json.loads(_controller_specs_json)
elif _single_ppo_exp:
    CONTROLLER_SPECS = [
        {"label": os.environ.get("THETA_SWEEP_SINGLE_LABEL", Path(_single_ppo_exp).name), "kind": "ppo", "experiment": _single_ppo_exp},
        {"label": "lqr", "kind": "lqr", "experiment": TARGET_EXPERIMENT},
    ]
else:
    CONTROLLER_SPECS = [
        {
            "label": "nll_50_async_10_rand_exp",
            "kind": "ppo",
            "experiment": exp_dict[1],
        },
        # {
        #     "label": "no uncertainty, no penalty, warm up",
        #     "kind": "ppo",
        #     "experiment": exp_dict[2],
        # },
        # {
        #     "label": "uncertainty input, no penalty",
        #     "kind": "ppo",
        #     "experiment": exp_dict[3],
        # },
        {
            "label": "lqr",
            "kind": "lqr",
            "experiment": TARGET_EXPERIMENT,
        },
    ]


MODEL_REGISTRY = {"PPO": PPO, "RecurrentPPO": RecurrentPPO, "UnifiedContextPPO": UnifiedContextPPO}
WRAPPER_REGISTRY = {
    "PreviousActionObservationWrapper": PreviousActionObservationWrapper,
    "RewardWrapper": RewardWrapper,
}


@dataclass
class ControllerRuntime:
    label: str
    kind: str
    experiment: str
    env: gym.Env
    model: Any | None
    assumed_theta: float | None = None


def _find_wrapper_attr(env: gym.Env, attr: str):
    e = env
    while True:
        if hasattr(e, attr):
            return getattr(e, attr)
        if not hasattr(e, "env"):
            break
        e = e.env
    raise AttributeError(f"Could not find attribute {attr!r} in env wrapper stack")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _extract_theta_range(cfg: dict[str, Any]) -> tuple[float, float]:
    for wr_spec in cfg.get("wrappers", []):
        if wr_spec.get("enabled", True) is False:
            continue
        if wr_spec.get("name") not in {"DomainRandomizationWrapper", "ChangingCartPoleDynamics"}:
            continue
        params = wr_spec.get("params", {}) or {}
        bounds = params.get("theta_mult_range")
        if bounds is not None and len(bounds) == 2:
            return float(bounds[0]), float(bounds[1])
    return (-0.25, 0.25)


def _resolve_theta_grid(cfg: dict[str, Any]) -> np.ndarray:
    explicit_values = os.environ.get("THETA_SWEEP_THETA_VALUES")
    if explicit_values:
        values = [float(value.strip()) for value in explicit_values.split(",") if value.strip()]
        if not values:
            raise ValueError("THETA_SWEEP_THETA_VALUES did not contain any numeric theta values.")
        return np.asarray(values, dtype=np.float64)
    if THETA_GRID is not None:
        return np.asarray(THETA_GRID, dtype=np.float64)
    theta_lo, theta_hi = _extract_theta_range(cfg)
    return np.linspace(theta_lo, theta_hi, num=N_THETA_POINTS, dtype=np.float64)


def make_env(cfg: dict[str, Any]) -> gym.Env:
    environment = str(cfg.get("environment", "cartpole")).lower()
    if environment == "cartpole":
        max_episode_steps = cfg.get("max_episode_steps", None)
        task_kwargs = None
        if max_episode_steps is not None:
            task_kwargs = {"time_limit": float(max_episode_steps) * 0.01}
        env = gym.make(
            "dm_control/cartpole-swingup_sparse-v0",
            render_mode=cfg.get("render_mode", None),
            task_kwargs=task_kwargs,
        )
        env = ForceMujocoFixedCamera(env, camera_id=0, width=1200, height=800)
    elif environment == "lqr":
        install_lqr_adapter_for_domain_randomization()
        lqr_cfg = dict(cfg.get("lqr_env", {}) or {})
        lqr_cfg.setdefault("max_episode_steps", int(cfg.get("max_episode_steps", 200)))
        env = LinearQuadraticEnv(**lqr_cfg)
    elif environment == "stopping_car":
        install_lqr_adapter_for_domain_randomization()
        stopping_cfg = dict(cfg.get("stopping_car_env", {}) or {})
        stopping_cfg.setdefault("max_episode_steps", int(cfg.get("max_episode_steps", 80)))
        env = StoppingCarEnv(**stopping_cfg)
    elif environment == "emergency_brake":
        install_lqr_adapter_for_domain_randomization()
        emergency_cfg = dict(cfg.get("emergency_brake_env", {}) or {})
        emergency_cfg.setdefault("max_episode_steps", int(cfg.get("max_episode_steps", 120)))
        env = EmergencyBrakeEnv(**emergency_cfg)
    else:
        raise ValueError(
            f"Unknown environment={environment}. Expected 'cartpole', 'lqr', "
            "'stopping_car', or 'emergency_brake'."
        )

    # For the parameter sweep we intentionally skip DomainRandomizationWrapper and set theta manually.
    for wr_spec in cfg.get("wrappers", []):
        if wr_spec.get("enabled", True) is False:
            continue
        wr_name = wr_spec["name"]
        if wr_name in {"DomainRandomizationWrapper", "ChangingCartPoleDynamics"}:
            continue
        wr_params = wr_spec.get("params", {}) or {}
        env = WRAPPER_REGISTRY[wr_name](env, **wr_params)

    return env


def _control_observation(env: gym.Env, obs: np.ndarray) -> np.ndarray:
    try:
        getter = _find_wrapper_attr(env, "get_current_control_observation")
    except AttributeError:
        return _flatten_obs(obs).astype(np.float64).reshape(-1)
    return np.asarray(getter(obs), dtype=np.float64).reshape(-1)


def build_controller(spec: dict[str, Any], target_cfg: dict[str, Any]) -> Optional[ControllerRuntime]:
    label = str(spec["label"])
    kind = str(spec["kind"])
    experiment = str(spec["experiment"])

    if kind not in {
        "ppo",
        "lqr",
        "nominal_lqr",
        "fixed_lqr",
        "stopping_nominal_brake",
        "stopping_true_m_brake",
        "stopping_early_probe",
        "stopping_coast",
        "emergency_coast",
        "emergency_nominal_brake",
        "emergency_true_m_brake",
    }:
        raise ValueError(f"Unknown controller kind={kind!r} for controller {label!r}")
    if not Path(experiment).exists():
        print(f"[skip] Controller {label!r}: experiment path does not exist: {experiment}")
        return None

    env_cfg = target_cfg
    model = None

    if kind == "ppo":
        controller_cfg = _load_yaml(Path(experiment) / "config.yaml")
        env_cfg = copy.deepcopy(controller_cfg)
        env_cfg["lqr_env"] = copy.deepcopy(target_cfg.get("lqr_env", {}) or {})
        env_cfg["stopping_car_env"] = copy.deepcopy(target_cfg.get("stopping_car_env", {}) or {})
        env_cfg["emergency_brake_env"] = copy.deepcopy(target_cfg.get("emergency_brake_env", {}) or {})
        env_cfg["environment"] = target_cfg.get("environment", env_cfg.get("environment"))
        if "max_episode_steps" in target_cfg:
            env_cfg["max_episode_steps"] = target_cfg["max_episode_steps"]
        model_spec = controller_cfg.get("model", None)
        if model_spec is None:
            raise ValueError(f"Experiment {experiment} has no model config.")
        model_name = model_spec["name"]
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model {model_name!r} in {experiment}")
        weights_name = spec.get("weights_name") or os.environ.get("THETA_SWEEP_WEIGHTS_NAME")
        if weights_name:
            weights_path = Path(experiment) / str(weights_name)
            if weights_path.suffix != ".zip":
                weights_path = weights_path.with_suffix(".zip")
        else:
            weights_path = (Path(experiment) / "weights_best.zip") if (Path(experiment) / "weights_best.zip").exists() else (Path(experiment) / "weights.zip")
        weights_path = str(weights_path)
        env = make_env(env_cfg)
        model = MODEL_REGISTRY[model_name].load(weights_path, env=env)
        model.naive_action_noise_std = model_spec.get("params", {}).get("naive_action_noise_std", getattr(model, "naive_action_noise_std", 0.0))
        model.naive_action_noise_dist = model_spec.get("params", {}).get("naive_action_noise_dist", getattr(model, "naive_action_noise_dist", "gaussian"))
        mean_context_override = spec.get("mean_context_override")
        if mean_context_override is not None:
            if not isinstance(model, UnifiedContextPPO):
                raise ValueError(f"mean_context_override requires UnifiedContextPPO, got {type(model).__name__}")
            if str(getattr(model, "context_mode", "")) == "privileged":
                raise ValueError("mean_context_override requires an encoder-based context mode.")

            mean_context_override = str(mean_context_override)
            if mean_context_override not in {"zeros", "quantize"}:
                raise ValueError("mean_context_override must be one of {'zeros', 'quantize'}.")
            model._eval_mean_context_override = mean_context_override

            if mean_context_override == "quantize":
                raw_step = float(spec["mean_context_quantization_step"])
                if raw_step <= 0.0:
                    raise ValueError("mean_context_quantization_step must be positive.")
                get_param_denoms = _find_wrapper_attr(env, "get_param_denoms")
                denoms = np.asarray(get_param_denoms(model.regression_param_names), dtype=np.float32)
                if denoms.shape != (model._latent_dim,):
                    raise ValueError(
                        "Mean-context quantization requires one regression parameter per latent dimension."
                    )
                model._eval_mean_context_quantization_steps = (
                    raw_step / np.maximum(np.abs(denoms), 1e-8) * float(model.z_scale)
                )
        uncertainty_override = spec.get("uncertainty_override")
        if uncertainty_override is not None:
            if not isinstance(model, UnifiedContextPPO):
                raise ValueError(f"uncertainty_override requires UnifiedContextPPO, got {type(model).__name__}")
            if str(getattr(model, "context_mode", "")) not in {"encoder_nll", "closed_form"} or not bool(
                getattr(model, "condition_on_uncertainty", False)
            ):
                raise ValueError(
                    "uncertainty_override requires uncertainty-conditioned encoder_nll or closed_form mode."
                )

            uncertainty_override = str(uncertainty_override)
            if uncertainty_override not in {"predicted", "zeros", "constant", "random_uniform", "reflected"}:
                raise ValueError(
                    "uncertainty_override must be one of "
                    "{'predicted', 'zeros', 'constant', 'random_uniform', 'reflected'}."
                )
            model.privileged_uncertainty_mode = uncertainty_override
            if uncertainty_override in {"constant", "reflected"}:
                model.privileged_uncertainty_value = float(spec["uncertainty_value"])
            elif uncertainty_override == "random_uniform":
                low = float(spec["uncertainty_low"])
                high = float(spec["uncertainty_high"])
                if high <= low:
                    raise ValueError(f"uncertainty_high must exceed uncertainty_low, got {low} and {high}.")
                model._eval_uncertainty_uniform_low = low
                model._eval_uncertainty_uniform_high = high
                model._eval_seed_random_uncertainty = True
    else:
        env = make_env(env_cfg)

    assumed_theta = float(spec["assumed_theta"]) if kind == "fixed_lqr" else None
    return ControllerRuntime(
        label=label,
        kind=kind,
        experiment=experiment,
        env=env,
        model=model,
        assumed_theta=assumed_theta,
    )


def _stopping_braking_action(obs: np.ndarray, *, m_assumed: float) -> np.ndarray:
    obs_before = np.asarray(obs, dtype=np.float64).reshape(-1)
    position = max(float(obs_before[0]), 1e-6)
    velocity = float(obs_before[1])
    if velocity < 0.0:
        required_acc = (velocity * velocity) / (2.0 * position)
        u = required_acc / max(float(m_assumed), 1e-6)
    else:
        u = -0.2
    return np.asarray([u], dtype=np.float32)


def _estimate_stopping_m(samples: list[tuple[float, float, float]]) -> float:
    xs = []
    ys = []
    for dv, dt, u in samples:
        if abs(u) < 1e-8:
            continue
        xs.append(dt * u)
        ys.append(dv)
    if not xs:
        return 1.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    denom = float(x @ x)
    if denom <= 1e-12:
        return 1.0
    return float(np.clip((x @ y) / denom, 0.2, 3.0))


def _emergency_braking_action(obs: np.ndarray, env: gym.Env, *, m_assumed: float) -> np.ndarray:
    obs_before = np.asarray(obs, dtype=np.float64).reshape(-1)
    velocity = float(obs_before[0])
    age_scaled = float(obs_before[1]) if obs_before.size > 1 else 0.0
    deadline_steps = int(_find_wrapper_attr(env, "deadline_steps"))
    safe_speed = float(_find_wrapper_attr(env, "safe_speed"))
    dt = float(_find_wrapper_attr(env, "dt"))
    if age_scaled <= 0.0 or abs(velocity) <= safe_speed:
        return np.asarray([0.0], dtype=np.float32)

    age = max(1, int(round(age_scaled * float(deadline_steps))))
    remaining_actions = max(1, deadline_steps - age + 1)
    target_velocity = 0.0
    u = (target_velocity - velocity) / (dt * max(float(m_assumed), 1e-6) * remaining_actions)
    return np.asarray([u], dtype=np.float32)


def _fixed_theta_lqr_action(env: gym.Env, obs: np.ndarray, assumed_theta: float) -> np.ndarray:
    obs_before = _control_observation(env, obs)
    a_nom = np.asarray(_find_wrapper_attr(env, "A_nominal"), dtype=np.float64)
    b_nom = np.asarray(_find_wrapper_attr(env, "B_nominal"), dtype=np.float64)
    delta_a = np.asarray(_find_wrapper_attr(env, "delta_A"), dtype=np.float64)
    delta_b = np.asarray(_find_wrapper_attr(env, "delta_B"), dtype=np.float64)
    q = np.asarray(_find_wrapper_attr(env, "Q"), dtype=np.float64)
    r = np.asarray(_find_wrapper_attr(env, "R"), dtype=np.float64)

    a_assumed = a_nom + float(assumed_theta) * delta_a
    b_assumed = b_nom + float(assumed_theta) * delta_b
    p = solve_discrete_are(a_assumed, b_assumed, q, r)
    k = np.linalg.solve(r + b_assumed.T @ p @ b_assumed, b_assumed.T @ p @ a_assumed)
    action = -(k @ obs_before)
    return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)


def _set_fixed_dynamics(env: gym.Env, theta: float, process_noise_scale: float) -> None:
    setter = _find_wrapper_attr(env, "set_dynamics_scales")
    setter(theta=theta, process_noise_scale=process_noise_scale)


def _get_true_param_dict(env: gym.Env) -> dict[str, float]:
    getter = _find_wrapper_attr(env, "get_true_params_dict")
    return dict(getter())


def _get_regression_targets(env: gym.Env, model: UnifiedContextPPO) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    get_params = _find_wrapper_attr(env, "get_true_params")
    get_param_denoms = _find_wrapper_attr(env, "get_param_denoms")
    raw = np.asarray(get_params(model.regression_param_names), dtype=np.float32)
    denoms = np.asarray(get_param_denoms(model.regression_param_names), dtype=np.float32)
    scaled = raw / np.maximum(denoms, 1e-8) * float(getattr(model, "z_scale", 1.0))
    return raw, denoms, scaled


def _predict_privileged_action(model: UnifiedContextPPO, obs: np.ndarray, env: gym.Env) -> np.ndarray:
    obs_np = _flatten_obs(obs)
    obs_th = th.as_tensor(obs_np, device=model.device, dtype=th.float32)
    _, _, true_scaled = _get_regression_targets(env, model)
    true_scaled_th = th.as_tensor(true_scaled[None, :], device=model.device, dtype=th.float32)
    privileged_context_th = model._build_privileged_context(true_scaled_th)

    model.policy.eval()
    with th.no_grad():
        actions_th, _, _ = model.policy.forward_with_z(obs_th, privileged_context_th, deterministic=SWEEP_DETERMINISTIC)
    actions_np = actions_th.cpu().numpy()

    if isinstance(model.action_space, gym.spaces.Box):
        actions_np = np.clip(actions_np, model.action_space.low, model.action_space.high)
    return actions_np


def _compute_encoder_prediction(
    model: UnifiedContextPPO,
    state,
    next_obs: np.ndarray,
    env: gym.Env,
) -> dict[str, Any]:
    obs_hist, act_hist = state
    current_obs_np = _flatten_obs(next_obs) if getattr(model, "use_transition_features", False) else None
    traj_window_np = model._build_traj_window(obs_hist, act_hist, current_obs=current_obs_np)
    traj_window_th = th.as_tensor(traj_window_np, device=model.device, dtype=th.float32)

    true_raw, denoms, true_scaled = _get_regression_targets(env, model)
    cached_context = np.asarray(getattr(model, "_predict_cached_context"), dtype=np.float64)[0]
    latent_dim = int(getattr(model, "_latent_dim", len(model.regression_param_names)))
    pred_scaled = cached_context[:latent_dim].astype(np.float64)
    pred_std: np.ndarray | None = None
    if bool(getattr(model, "condition_on_uncertainty", False)) and cached_context.shape[0] >= 2 * latent_dim:
        pred_std = cached_context[latent_dim:2 * latent_dim].astype(np.float64)

    est_pred_scaled: np.ndarray | None = None
    est_pred_std: np.ndarray | None = None

    model.policy.eval()
    with th.no_grad():
        mode = str(getattr(model, "context_mode", ""))
        if mode == "encoder_nll":
            mu, logvar = model.policy.encode_context(traj_window_th, return_logvar=True)
            logvar = model.policy.clamp_logvar(logvar)
            est_pred_scaled = mu.squeeze(0).detach().cpu().numpy().astype(np.float64)
            est_pred_std = th.exp(0.5 * logvar).squeeze(0).detach().cpu().numpy().astype(np.float64)

    z_scale = float(getattr(model, "z_scale", 1.0))
    pred_raw = pred_scaled / max(z_scale, 1e-8) * denoms.astype(np.float64)
    pred_std_raw = None if pred_std is None else pred_std / max(z_scale, 1e-8) * denoms.astype(np.float64)

    out: dict[str, Any] = {
        "true_raw": true_raw.astype(np.float64),
        "true_scaled": true_scaled.astype(np.float64),
        "pred_raw": pred_raw.astype(np.float64),
        "pred_scaled": pred_scaled.astype(np.float64),
    }
    if pred_std is not None and pred_std_raw is not None:
        out["pred_std"] = pred_std_raw.astype(np.float64)
        out["pred_std_scaled"] = pred_std.astype(np.float64)
    if est_pred_scaled is not None:
        out["est_pred_raw"] = (est_pred_scaled / max(z_scale, 1e-8) * denoms.astype(np.float64)).astype(np.float64)
        out["est_pred_scaled"] = est_pred_scaled.astype(np.float64)
    if est_pred_std is not None:
        out["est_pred_std"] = (est_pred_std / max(z_scale, 1e-8) * denoms.astype(np.float64)).astype(np.float64)
        out["est_pred_std_scaled"] = est_pred_std.astype(np.float64)
    return out


def _squeeze_action_for_env(action: np.ndarray) -> np.ndarray:
    action_env = action
    if isinstance(action_env, np.ndarray) and action_env.ndim >= 1 and action_env.shape[0] == 1:
        action_env = action_env[0]
    return action_env


def _summarize_episode_predictions(
    model: UnifiedContextPPO,
    step_prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not step_prediction_rows:
        return {}

    ignore_first = IGNORE_FIRST_STEPS
    if ignore_first is None:
        ignore_first = int(getattr(model, "window_length", 0))

    total_steps = len(step_prediction_rows)
    tail_len = max(1, int(math.ceil(TAIL_FRACTION * total_steps)))
    tail_start = max(ignore_first, total_steps - tail_len)
    tail_rows = step_prediction_rows[tail_start:] if tail_start < total_steps else step_prediction_rows[-1:]

    out: dict[str, Any] = {
        "prediction_steps_total": total_steps,
        "prediction_tail_start_step": int(tail_rows[0]["step"]),
    }

    for name in model.regression_param_names:
        pred_key = f"pred_{name}"
        true_key = f"true_{name}"
        last_pred = float(step_prediction_rows[-1][pred_key])
        true_val = float(step_prediction_rows[-1][true_key])
        tail_preds = np.asarray([float(row[pred_key]) for row in tail_rows], dtype=np.float64)
        tail_errors = tail_preds - true_val

        out[f"{pred_key}_last"] = last_pred
        out[f"{pred_key}_tail_mean"] = float(np.mean(tail_preds))
        out[f"{pred_key}_tail_std"] = float(np.std(tail_preds))
        out[f"{name}_error_last"] = float(last_pred - true_val)
        out[f"{name}_abs_error_last"] = float(abs(last_pred - true_val))
        out[f"{name}_error_tail_mean"] = float(np.mean(tail_errors))
        out[f"{name}_abs_error_tail_mean"] = float(np.mean(np.abs(tail_errors)))
        out[f"{name}_rmse_tail"] = float(np.sqrt(np.mean(np.square(tail_errors))))

        std_key = f"pred_{name}_std"
        if std_key in step_prediction_rows[-1]:
            tail_stds = np.asarray([float(row[std_key]) for row in tail_rows], dtype=np.float64)
            out[f"{std_key}_tail_mean"] = float(np.mean(tail_stds))

    return out


def run_episode(
    runtime: ControllerRuntime,
    theta: float,
    process_noise_scale: float,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    env = runtime.env
    _set_fixed_dynamics(env, theta=theta, process_noise_scale=process_noise_scale)
    obs, info = env.reset(seed=seed)

    state = None
    episode_start = np.ones((1,), dtype=bool)
    if runtime.kind == "ppo" and bool(getattr(runtime.model, "_eval_seed_random_uncertainty", False)):
        th.manual_seed(seed)

    terminated = False
    truncated = False
    ep_return = 0.0
    ep_len = 0
    step_prediction_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    cumulative_info_proxy = 0.0
    cumulative_cost = 0.0
    cumulative_state_cost = 0.0
    cumulative_action_cost = 0.0
    cumulative_theta_sensitivity_sq = 0.0
    cumulative_nominal_deviation_sq = 0.0
    cumulative_nominal_deviation_r = 0.0
    cumulative_crashes = 0.0
    first_crash_step: int | None = None
    stopping_estimate_samples: list[tuple[float, float, float]] = []

    while not (terminated or truncated):
        obs_before = _control_observation(env, obs)
        lqr_action: np.ndarray | None = None
        nominal_action: np.ndarray | None = None
        k_nom: np.ndarray | None = None
        try:
            lqr_action = np.asarray(_find_wrapper_attr(env, "lqr_action")(obs), dtype=np.float64).reshape(-1)
            k_nom = np.asarray(_find_wrapper_attr(env, "K_nominal"), dtype=np.float64)
            nominal_action = -(k_nom @ obs_before)
            nominal_action = np.clip(nominal_action, env.action_space.low, env.action_space.high)
        except Exception:
            pass
        if runtime.kind == "lqr":
            if lqr_action is None:
                raise ValueError(f"Controller {runtime.label!r} requested kind='lqr', but env has no lqr_action().")
            action = lqr_action
        elif runtime.kind == "nominal_lqr":
            if nominal_action is None:
                raise ValueError(
                    f"Controller {runtime.label!r} requested kind='nominal_lqr', but env has no K_nominal."
                )
            action = nominal_action
        elif runtime.kind == "fixed_lqr":
            if lqr_action is None:
                raise ValueError(f"Controller {runtime.label!r} requested kind='fixed_lqr', but env has no LQR support.")
            if runtime.assumed_theta is None:
                raise ValueError(f"fixed_lqr controller {runtime.label!r} has no assumed_theta")
            action = _fixed_theta_lqr_action(env, obs, runtime.assumed_theta)
        elif runtime.kind == "stopping_coast":
            action = np.asarray([0.0], dtype=np.float32)
        elif runtime.kind == "stopping_nominal_brake":
            action = _stopping_braking_action(obs_before, m_assumed=1.0)
        elif runtime.kind == "stopping_true_m_brake":
            m = float(_find_wrapper_attr(env, "m"))
            action = _stopping_braking_action(obs_before, m_assumed=m)
        elif runtime.kind == "stopping_early_probe":
            if ep_len < 8:
                action = np.asarray([0.35], dtype=np.float32)
            else:
                action = _stopping_braking_action(
                    obs_before,
                    m_assumed=_estimate_stopping_m(stopping_estimate_samples),
                )
        elif runtime.kind == "emergency_coast":
            action = np.asarray([0.0], dtype=np.float32)
        elif runtime.kind == "emergency_nominal_brake":
            action = _emergency_braking_action(obs_before, env, m_assumed=1.0)
        elif runtime.kind == "emergency_true_m_brake":
            m = float(_find_wrapper_attr(env, "m"))
            action = _emergency_braking_action(obs_before, env, m_assumed=m)
        else:
            model = runtime.model
            if isinstance(model, UnifiedContextPPO) and str(getattr(model, "context_mode", "")) == "privileged":
                action = _predict_privileged_action(model, obs, env)
            else:
                action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=SWEEP_DETERMINISTIC)

        action_env = _squeeze_action_for_env(action)
        if isinstance(env.action_space, gym.spaces.Box):
            action_env = np.clip(action_env, env.action_space.low, env.action_space.high)
        next_obs, reward, terminated, truncated, info = env.step(action_env)
        ep_return += float(reward)
        ep_len += 1
        action_flat = np.asarray(action_env, dtype=np.float64).reshape(-1)
        action_delta = None if lqr_action is None else action_flat - lqr_action
        if runtime.kind.startswith("stopping_"):
            next_obs_flat = _flatten_obs(next_obs).astype(np.float64).reshape(-1)
            dt = float(_find_wrapper_attr(env, "dt"))
            stopping_estimate_samples.append(
                (float(next_obs_flat[1] - obs_before[1]), dt, float(action_flat[0]))
            )

        info_proxy = float("nan")
        theta_sensitivity_sq = float("nan")
        nominal_deviation_sq = float("nan")
        nominal_deviation_r = float("nan")
        if k_nom is not None and nominal_action is not None:
            nominal_mismatch = k_nom @ obs_before + action_flat
            info_proxy = float(np.sum(np.square(nominal_mismatch)))
            try:
                delta_b = np.asarray(_find_wrapper_attr(env, "delta_B"), dtype=np.float64)
                theta_sensitivity_sq = float(np.sum(np.square(delta_b @ nominal_mismatch.reshape(-1))))
            except Exception:
                theta_sensitivity_sq = float(np.sum(np.square(nominal_mismatch)))
            nominal_delta = action_flat - nominal_action
            nominal_deviation_sq = float(np.sum(np.square(nominal_delta)))
            try:
                r = np.asarray(_find_wrapper_attr(env, "R"), dtype=np.float64)
                nominal_deviation_r = float(nominal_delta @ r @ nominal_delta)
            except Exception:
                nominal_deviation_r = nominal_deviation_sq
        else:
            theta_sensitivity_sq = float(np.sum(np.square(action_flat)))

        cumulative_info_proxy += 0.0 if not np.isfinite(info_proxy) else info_proxy
        cumulative_theta_sensitivity_sq += (
            0.0 if not np.isfinite(theta_sensitivity_sq) else theta_sensitivity_sq
        )
        cumulative_nominal_deviation_sq += (
            0.0 if not np.isfinite(nominal_deviation_sq) else nominal_deviation_sq
        )
        cumulative_nominal_deviation_r += (
            0.0 if not np.isfinite(nominal_deviation_r) else nominal_deviation_r
        )
        cumulative_cost += float(info.get("cost", -reward))
        cumulative_state_cost += float(info.get("state_cost", 0.0))
        cumulative_action_cost += float(info.get("action_cost", 0.0))
        crashed = float(info.get("crashed", 0.0))
        cumulative_crashes += crashed
        if crashed and first_crash_step is None:
            first_crash_step = int(ep_len)

        trace_row: dict[str, Any] | None = None
        if SAVE_TRAJECTORY_TRACE:
            trace_row = {
                "controller": runtime.label,
                "controller_kind": runtime.kind,
                "experiment": runtime.experiment,
                "theta": float(theta),
                "process_noise_scale": float(process_noise_scale),
                "seed": int(seed),
                "step": int(ep_len),
                "reward": float(reward),
                "cost": float(info.get("cost", -reward)),
                "state_cost": float(info.get("state_cost", 0.0)),
                "action_cost": float(info.get("action_cost", 0.0)),
                "episode_return_so_far": float(ep_return),
                "info_proxy": info_proxy,
                "theta_sensitivity_sq": theta_sensitivity_sq,
                "nominal_deviation_sq": nominal_deviation_sq,
                "nominal_deviation_r": nominal_deviation_r,
                "cumulative_info_proxy": float(cumulative_info_proxy),
                "cumulative_theta_sensitivity_sq": float(cumulative_theta_sensitivity_sq),
                "cumulative_nominal_deviation_sq": float(cumulative_nominal_deviation_sq),
                "cumulative_nominal_deviation_r": float(cumulative_nominal_deviation_r),
                "cumulative_cost": float(cumulative_cost),
                "cumulative_state_cost": float(cumulative_state_cost),
                "cumulative_action_cost": float(cumulative_action_cost),
            }
            for idx, value in enumerate(obs_before):
                trace_row[f"x{idx}"] = float(value)
            for idx, value in enumerate(_flatten_obs(obs).reshape(-1)):
                trace_row[f"policy_obs{idx}"] = float(value)
            for idx, value in enumerate(action_flat):
                trace_row[f"u{idx}"] = float(value)
            if lqr_action is not None:
                for idx, value in enumerate(lqr_action):
                    trace_row[f"u_lqr{idx}"] = float(value)
            if nominal_action is not None:
                for idx, value in enumerate(nominal_action):
                    trace_row[f"u_nominal{idx}"] = float(value)
            if action_delta is not None:
                for idx, value in enumerate(action_delta):
                    trace_row[f"u_delta_lqr{idx}"] = float(value)
            if nominal_action is not None:
                for idx, value in enumerate(action_flat - nominal_action):
                    trace_row[f"u_delta_nominal{idx}"] = float(value)
            for key in [
                "true_cost",
                "position_cost",
                "cruise_cost",
                "emergency_velocity_cost",
                "crash_cost",
                "failure_cost",
                "crashed",
                "failed",
                "success",
                "emergency_age",
                "deadline_steps",
                "emergency_active",
                "event_started",
                "event_time",
                "post_action_velocity",
                "m",
            ]:
                if key in info:
                    trace_row[key] = float(info[key])

        if runtime.kind == "ppo":
            model = runtime.model
            if isinstance(model, UnifiedContextPPO) and str(getattr(model, "context_mode", "")) in {"encoder_mle", "encoder_nll", "closed_form"} and state is not None:
                pred = _compute_encoder_prediction(model, state=state, next_obs=next_obs, env=env)
                row: dict[str, Any] = {
                    "controller": runtime.label,
                    "experiment": runtime.experiment,
                    "theta": float(theta),
                    "process_noise_scale": float(process_noise_scale),
                    "seed": int(seed),
                    "step": int(ep_len),
                    "reward": float(reward),
                    "episode_return_so_far": float(ep_return),
                }
                for name, value in zip(model.regression_param_names, pred["true_raw"]):
                    row[f"true_{name}"] = float(value)
                for name, value in zip(model.regression_param_names, pred["pred_raw"]):
                    row[f"pred_{name}"] = float(value)
                for name, value in zip(model.regression_param_names, pred["pred_scaled"]):
                    row[f"pred_{name}_scaled"] = float(value)
                if "pred_std" in pred:
                    for name, value in zip(model.regression_param_names, pred["pred_std"]):
                        row[f"pred_{name}_std"] = float(value)
                if "pred_std_scaled" in pred:
                    for name, value in zip(model.regression_param_names, pred["pred_std_scaled"]):
                        row[f"pred_{name}_std_scaled"] = float(value)
                if "est_pred_raw" in pred:
                    for name, value in zip(model.regression_param_names, pred["est_pred_raw"]):
                        row[f"est_pred_{name}"] = float(value)
                if "est_pred_scaled" in pred:
                    for name, value in zip(model.regression_param_names, pred["est_pred_scaled"]):
                        row[f"est_pred_{name}_scaled"] = float(value)
                if "est_pred_std" in pred:
                    for name, value in zip(model.regression_param_names, pred["est_pred_std"]):
                        row[f"est_pred_{name}_std"] = float(value)
                if "est_pred_std_scaled" in pred:
                    for name, value in zip(model.regression_param_names, pred["est_pred_std_scaled"]):
                        row[f"est_pred_{name}_std_scaled"] = float(value)
                step_prediction_rows.append(row)
                if trace_row is not None:
                    for idx, value in enumerate(pred.get("pred_scaled", [])):
                        trace_row[f"latent_cached_scaled_{idx}"] = float(value)
                    if "pred_raw" in pred:
                        for idx, value in enumerate(pred["pred_raw"]):
                            trace_row[f"latent_cached_raw_{idx}"] = float(value)
                    if "pred_std_scaled" in pred:
                        for idx, value in enumerate(pred["pred_std_scaled"]):
                            trace_row[f"latent_cached_std_scaled_{idx}"] = float(value)
                    if "pred_std" in pred:
                        for idx, value in enumerate(pred["pred_std"]):
                            trace_row[f"latent_cached_std_raw_{idx}"] = float(value)
                    if "est_pred_scaled" in pred:
                        for idx, value in enumerate(pred["est_pred_scaled"]):
                            trace_row[f"latent_est_scaled_{idx}"] = float(value)
                    if "est_pred_raw" in pred:
                        for idx, value in enumerate(pred["est_pred_raw"]):
                            trace_row[f"latent_est_raw_{idx}"] = float(value)
                        for name, value in zip(model.regression_param_names, pred["est_pred_raw"]):
                            trace_row[f"est_pred_{name}"] = float(value)
                    if "est_pred_std_scaled" in pred:
                        for idx, value in enumerate(pred["est_pred_std_scaled"]):
                            trace_row[f"latent_est_std_scaled_{idx}"] = float(value)
                    if "est_pred_std" in pred:
                        for idx, value in enumerate(pred["est_pred_std"]):
                            trace_row[f"latent_est_std_raw_{idx}"] = float(value)
                        for name, value in zip(model.regression_param_names, pred["est_pred_std"]):
                            trace_row[f"est_pred_{name}_std"] = float(value)

        if trace_row is not None:
            trajectory_rows.append(trace_row)

        obs = next_obs
        episode_start[...] = bool(terminated or truncated)

    quadratic_return = -(cumulative_state_cost + cumulative_action_cost)
    selected_return = quadratic_return if RETURN_MODE == "quadratic" else ep_return

    episode_row: dict[str, Any] = {
        "controller": runtime.label,
        "controller_kind": runtime.kind,
        "experiment": runtime.experiment,
        "theta": float(theta),
        "process_noise_scale": float(process_noise_scale),
        "seed": int(seed),
        "episode_return": float(selected_return),
        "episode_reward_return": float(ep_return),
        "episode_quadratic_return": float(quadratic_return),
        "episode_log_cost": float(cumulative_cost),
        "episode_quadratic_cost": float(cumulative_state_cost + cumulative_action_cost),
        "episode_state_cost": float(cumulative_state_cost),
        "episode_action_cost": float(cumulative_action_cost),
        "episode_info_proxy": float(cumulative_info_proxy),
        "episode_theta_sensitivity_sq": float(cumulative_theta_sensitivity_sq),
        "episode_nominal_deviation_sq": float(cumulative_nominal_deviation_sq),
        "episode_nominal_deviation_r": float(cumulative_nominal_deviation_r),
        "episode_information_per_action_cost": float(
            cumulative_theta_sensitivity_sq / max(cumulative_action_cost, 1e-12)
        ),
        "episode_information_per_nominal_deviation_r": float(
            cumulative_theta_sensitivity_sq / max(cumulative_nominal_deviation_r, 1e-12)
        ),
        "episode_crashes": float(cumulative_crashes),
        "episode_crashed": float(cumulative_crashes > 0.0),
        "episode_first_crash_step": float(first_crash_step) if first_crash_step is not None else float("nan"),
        "episode_length": int(ep_len),
    }
    episode_row.update(_get_true_param_dict(env))

    if runtime.kind == "ppo":
        model = runtime.model
        if isinstance(model, UnifiedContextPPO):
            episode_row["model_context_mode"] = str(getattr(model, "context_mode", ""))
        if isinstance(model, UnifiedContextPPO) and str(getattr(model, "context_mode", "")) in {"encoder_mle", "encoder_nll", "closed_form"}:
            episode_row.update(_summarize_episode_predictions(model, step_prediction_rows))

    return episode_row, step_prediction_rows, trajectory_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _add_metric_box(ax, lines: list[str]) -> None:
    if not lines:
        return
    ax.text(
        0.01,
        0.99,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.75"},
    )


def _group_episode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["controller"]), float(row["process_noise_scale"]), float(row["theta"]))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (controller, process_noise_scale, theta), bucket in sorted(grouped.items()):
        agg: dict[str, Any] = {
            "controller": controller,
            "process_noise_scale": process_noise_scale,
            "theta": theta,
            "num_episodes": len(bucket),
            "episode_return_mean": float(np.mean([float(row["episode_return"]) for row in bucket])),
            "episode_return_std": float(np.std([float(row["episode_return"]) for row in bucket])),
            "episode_return_q10": float(np.quantile([float(row["episode_return"]) for row in bucket], 0.10)),
            "episode_return_median": float(np.quantile([float(row["episode_return"]) for row in bucket], 0.50)),
            "episode_return_q90": float(np.quantile([float(row["episode_return"]) for row in bucket], 0.90)),
            "episode_length_mean": float(np.mean([float(row["episode_length"]) for row in bucket])),
        }

        numeric_keys = set()
        for row in bucket:
            for key, value in row.items():
                if key in agg or key in {"seed", "experiment", "controller_kind", "model_context_mode"}:
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)):
                    numeric_keys.add(key)

        for key in sorted(numeric_keys):
            values = np.asarray([float(row[key]) for row in bucket], dtype=np.float64)
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
            agg[f"{key}_q10"] = float(np.quantile(values, 0.10))
            agg[f"{key}_median"] = float(np.quantile(values, 0.50))
            agg[f"{key}_q90"] = float(np.quantile(values, 0.90))
            if key.endswith("_rmse_tail"):
                agg[f"{key}_across_episodes"] = float(np.sqrt(np.mean(np.square(values))))
        out.append(agg)

    return out


def _plot_return_vs_theta(agg_rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not agg_rows:
        return

    has_quantiles = all(
        key in agg_rows[0]
        for key in ["episode_return_median", "episode_return_q10", "episode_return_q90"]
    )
    if has_quantiles:
        fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharex=True)
        ax_mean, ax_mean_line = axes[0]
        ax_quantile, ax_quantile_line = axes[1]
    else:
        fig, ax_mean = plt.subplots(figsize=(10, 6))
        ax_mean_line = None
        ax_quantile = None
        ax_quantile_line = None

    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in agg_rows:
        key = (str(row["controller"]), float(row["process_noise_scale"]))
        groups.setdefault(key, []).append(row)

    metric_lines: list[str] = []
    for (controller, process_noise_scale), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda x: float(x["theta"]))
        x = np.asarray([float(row["theta"]) for row in rows], dtype=np.float64)
        y = np.asarray([float(row["episode_return_mean"]) for row in rows], dtype=np.float64)
        y_std = np.asarray([float(row["episode_return_std"]) for row in rows], dtype=np.float64)
        label = f"{controller} | noise_scale={process_noise_scale:g}"
        ax_mean.plot(x, y, linewidth=2, label=label)
        ax_mean.fill_between(x, y - y_std, y + y_std, alpha=0.15)
        if ax_mean_line is not None:
            ax_mean_line.plot(x, y, linewidth=2, label=label)

        if ax_quantile is not None:
            y_med = np.asarray([float(row["episode_return_median"]) for row in rows], dtype=np.float64)
            y_q10 = np.asarray([float(row["episode_return_q10"]) for row in rows], dtype=np.float64)
            y_q90 = np.asarray([float(row["episode_return_q90"]) for row in rows], dtype=np.float64)
            ax_quantile.plot(x, y_med, linewidth=2, label=label)
            ax_quantile.fill_between(x, y_q10, y_q90, alpha=0.15)
            ax_quantile_line.plot(x, y_med, linewidth=2, label=label)

        tail = y[np.abs(x) >= TAIL_ABS_THETA_THRESHOLD]
        center = y[np.abs(x) <= CENTER_ABS_THETA_THRESHOLD]
        metric_lines.append(
            f"{controller}: mean={np.mean(y):.1f}, tail={np.mean(tail):.1f}, center={np.mean(center):.1f}"
        )

    ax_mean.set_ylabel("Episode Return")
    ax_mean.set_title("Return vs Fixed Theta (Mean +/- Std)")
    ax_mean.grid(True, alpha=0.3)
    ax_mean.legend()

    if ax_quantile is not None:
        ax_mean_line.set_ylabel("Episode Return")
        ax_mean_line.set_title("Return vs Fixed Theta (Mean Only)")
        ax_mean_line.grid(True, alpha=0.3)
        ax_mean_line.legend()

        ax_quantile.set_xlabel("theta")
        ax_quantile.set_ylabel("Episode Return")
        ax_quantile.set_title("Return vs Fixed Theta (Median with 10/90 Quantiles)")
        ax_quantile.grid(True, alpha=0.3)
        ax_quantile.legend()

        ax_quantile_line.set_xlabel("theta")
        ax_quantile_line.set_ylabel("Episode Return")
        ax_quantile_line.set_title("Return vs Fixed Theta (Median Only)")
        ax_quantile_line.grid(True, alpha=0.3)
        ax_quantile_line.legend()
        _add_metric_box(ax_mean_line, metric_lines)
    else:
        ax_mean.set_xlabel("theta")
        _add_metric_box(ax_mean, metric_lines)

    fig.tight_layout()
    fig.savefig(output_dir / "return_vs_theta.png", dpi=200)
    plt.close(fig)


def _plot_param_prediction_vs_theta(agg_rows: list[dict[str, Any]], output_dir: Path, param_name: str) -> None:
    rows = [row for row in agg_rows if f"pred_{param_name}_tail_mean_mean" in row]
    if not rows:
        return

    key_median = f"pred_{param_name}_last_median"
    key_q10 = f"pred_{param_name}_last_q10"
    key_q90 = f"pred_{param_name}_last_q90"
    has_quantiles = all(key in rows[0] for key in [key_median, key_q10, key_q90])
    if has_quantiles:
        fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharex=True)
        ax_mean, ax_mean_line = axes[0]
        ax_quantile, ax_quantile_line = axes[1]
    else:
        fig, ax_mean = plt.subplots(figsize=(10, 6))
        ax_mean_line = None
        ax_quantile = None
        ax_quantile_line = None

    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["controller"]), float(row["process_noise_scale"]))
        groups.setdefault(key, []).append(row)

    x_all = np.asarray([float(row["theta"]) for row in rows], dtype=np.float64)
    ax_mean.plot(
        [float(np.min(x_all)), float(np.max(x_all))],
        [float(np.min(x_all)), float(np.max(x_all))],
        linestyle="--",
        linewidth=1.5,
        color="black",
        label="ideal",
    )
    if ax_mean_line is not None:
        ax_mean_line.plot(
            [float(np.min(x_all)), float(np.max(x_all))],
            [float(np.min(x_all)), float(np.max(x_all))],
            linestyle="--",
            linewidth=1.5,
            color="black",
            label="ideal",
        )
    if ax_quantile is not None:
        ax_quantile.plot(
            [float(np.min(x_all)), float(np.max(x_all))],
            [float(np.min(x_all)), float(np.max(x_all))],
            linestyle="--",
            linewidth=1.5,
            color="black",
            label="ideal",
        )
        ax_quantile_line.plot(
            [float(np.min(x_all)), float(np.max(x_all))],
            [float(np.min(x_all)), float(np.max(x_all))],
            linestyle="--",
            linewidth=1.5,
            color="black",
            label="ideal",
        )

    metric_lines: list[str] = []
    for (controller, process_noise_scale), bucket in sorted(groups.items()):
        bucket = sorted(bucket, key=lambda x: float(x["theta"]))
        x = np.asarray([float(row["theta"]) for row in bucket], dtype=np.float64)
        y = np.asarray([float(row[f"pred_{param_name}_tail_mean_mean"]) for row in bucket], dtype=np.float64)
        y_std = np.asarray([float(row[f"pred_{param_name}_tail_mean_std"]) for row in bucket], dtype=np.float64)
        label = f"{controller} | noise_scale={process_noise_scale:g}"
        ax_mean.plot(x, y, linewidth=2, label=label)
        ax_mean.fill_between(x, y - y_std, y + y_std, alpha=0.15)
        if ax_mean_line is not None:
            ax_mean_line.plot(x, y, linewidth=2, label=label)

        if ax_quantile is not None:
            y_med = np.asarray([float(row[key_median]) for row in bucket], dtype=np.float64)
            y_q10 = np.asarray([float(row[key_q10]) for row in bucket], dtype=np.float64)
            y_q90 = np.asarray([float(row[key_q90]) for row in bucket], dtype=np.float64)
            ax_quantile.plot(x, y_med, linewidth=2, label=label)
            ax_quantile.fill_between(x, y_q10, y_q90, alpha=0.15)
            ax_quantile_line.plot(x, y_med, linewidth=2, label=label)

        ok = np.isfinite(x) & np.isfinite(y)
        slope = float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() > 1 else float("nan")
        rmse_key = f"{param_name}_rmse_tail_across_episodes"
        rmse_vals = [float(row[rmse_key]) for row in bucket if row.get(rmse_key) not in {None, ""}]
        tail_vals = [
            float(row[rmse_key])
            for row in bucket
            if row.get(rmse_key) not in {None, ""} and abs(float(row["theta"])) >= TAIL_ABS_THETA_THRESHOLD
        ]
        if rmse_vals:
            tail_rmse = float(np.mean(tail_vals)) if tail_vals else float("nan")
            metric_lines.append(
                f"{controller}: RMSE={np.mean(rmse_vals):.3f}, tail={tail_rmse:.3f}, slope={slope:.2f}"
            )

    ax_mean.set_ylabel(f"Predicted {param_name} (tail mean)")
    ax_mean.set_title(f"Predicted {param_name} vs True {param_name} (Mean +/- Std)")
    ax_mean.grid(True, alpha=0.3)
    ax_mean.legend()

    if ax_quantile is not None:
        ax_mean_line.set_ylabel(f"Predicted {param_name} (tail mean)")
        ax_mean_line.set_title(f"Predicted {param_name} vs True {param_name} (Mean Only)")
        ax_mean_line.grid(True, alpha=0.3)
        ax_mean_line.legend()

        ax_quantile.set_xlabel(f"True {param_name}")
        ax_quantile.set_ylabel(f"Predicted {param_name} (last prediction)")
        ax_quantile.set_title(f"Predicted {param_name} vs True {param_name} (Median with 10/90 Quantiles)")
        ax_quantile.grid(True, alpha=0.3)
        ax_quantile.legend()

        ax_quantile_line.set_xlabel(f"True {param_name}")
        ax_quantile_line.set_ylabel(f"Predicted {param_name} (last prediction)")
        ax_quantile_line.set_title(f"Predicted {param_name} vs True {param_name} (Median Only)")
        ax_quantile_line.grid(True, alpha=0.3)
        ax_quantile_line.legend()
        _add_metric_box(ax_mean_line, metric_lines)
    else:
        ax_mean.set_xlabel(f"True {param_name}")
        _add_metric_box(ax_mean, metric_lines)

    fig.tight_layout()
    fig.savefig(output_dir / f"{param_name}_prediction_vs_theta.png", dpi=200)
    plt.close(fig)


def _plot_param_rmse_vs_theta(agg_rows: list[dict[str, Any]], output_dir: Path, param_name: str) -> None:
    rows = [row for row in agg_rows if f"{param_name}_rmse_tail_across_episodes" in row]
    if not rows:
        return

    plt.figure(figsize=(10, 6))
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["controller"]), float(row["process_noise_scale"]))
        groups.setdefault(key, []).append(row)

    for (controller, process_noise_scale), bucket in sorted(groups.items()):
        bucket = sorted(bucket, key=lambda x: float(x["theta"]))
        x = np.asarray([float(row["theta"]) for row in bucket], dtype=np.float64)
        y = np.asarray([float(row[f"{param_name}_rmse_tail_across_episodes"]) for row in bucket], dtype=np.float64)
        label = f"{controller} | noise_scale={process_noise_scale:g}"
        plt.plot(x, y, linewidth=2, label=label)

    plt.xlabel("theta")
    plt.ylabel(f"{param_name} tail RMSE")
    plt.title(f"{param_name} RMSE vs Fixed Theta")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{param_name}_rmse_vs_theta.png", dpi=200)
    plt.close()


def _plot_param_uncertainty_vs_theta(agg_rows: list[dict[str, Any]], output_dir: Path, param_name: str) -> None:
    rows = [row for row in agg_rows if f"pred_{param_name}_std_tail_mean_mean" in row]
    if not rows:
        return

    plt.figure(figsize=(10, 6))
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["controller"]), float(row["process_noise_scale"]))
        groups.setdefault(key, []).append(row)

    for (controller, process_noise_scale), bucket in sorted(groups.items()):
        bucket = sorted(bucket, key=lambda x: float(x["theta"]))
        x = np.asarray([float(row["theta"]) for row in bucket], dtype=np.float64)
        y = np.asarray([float(row[f"pred_{param_name}_std_tail_mean_mean"]) for row in bucket], dtype=np.float64)
        y_std = np.asarray([float(row[f"pred_{param_name}_std_tail_mean_std"]) for row in bucket], dtype=np.float64)
        label = f"{controller} | noise_scale={process_noise_scale:g}"
        plt.plot(x, y, linewidth=2, label=label)
        plt.fill_between(x, np.maximum(0.0, y - y_std), y + y_std, alpha=0.15)

    plt.xlabel(f"True {param_name}")
    plt.ylabel(f"Predicted {param_name} std (tail mean)")
    plt.title(f"Predicted {param_name} Uncertainty vs True {param_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{param_name}_uncertainty_vs_theta.png", dpi=200)
    plt.close()


def _tail_step_prediction_rows(
    step_rows: list[dict[str, Any]],
    runtimes: list[ControllerRuntime],
) -> list[dict[str, Any]]:
    if not step_rows:
        return []

    controller_ignore_first: dict[str, int] = {}
    for runtime in runtimes:
        if IGNORE_FIRST_STEPS is not None:
            controller_ignore_first[runtime.label] = int(IGNORE_FIRST_STEPS)
        elif runtime.kind == "ppo" and isinstance(runtime.model, UnifiedContextPPO):
            controller_ignore_first[runtime.label] = int(getattr(runtime.model, "window_length", 0))
        else:
            controller_ignore_first[runtime.label] = 0

    grouped: dict[tuple[str, float, float, int], list[dict[str, Any]]] = {}
    for row in step_rows:
        key = (
            str(row["controller"]),
            float(row["process_noise_scale"]),
            float(row["theta"]),
            int(row["seed"]),
        )
        grouped.setdefault(key, []).append(row)

    tail_rows: list[dict[str, Any]] = []
    for (controller, _process_noise_scale, _theta, _seed), bucket in grouped.items():
        bucket = sorted(bucket, key=lambda x: int(x["step"]))
        total_steps = len(bucket)
        tail_len = max(1, int(math.ceil(TAIL_FRACTION * total_steps)))
        ignore_first = int(controller_ignore_first.get(controller, 0))
        tail_start = max(ignore_first, total_steps - tail_len)
        selected = bucket[tail_start:] if tail_start < total_steps else bucket[-1:]
        tail_rows.extend(selected)

    return tail_rows


def _plot_uncertainty_calibration(
    step_rows: list[dict[str, Any]],
    runtimes: list[ControllerRuntime],
    output_dir: Path,
    param_name: str,
) -> None:
    tail_rows = _tail_step_prediction_rows(step_rows, runtimes)
    std_key = f"pred_{param_name}_std"
    pred_key = f"pred_{param_name}"
    true_key = f"true_{param_name}"
    rows = [row for row in tail_rows if std_key in row and pred_key in row and true_key in row]
    if not rows:
        return

    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["controller"]), float(row["process_noise_scale"]))
        groups.setdefault(key, []).append(row)

    all_std = np.asarray([float(row[std_key]) for row in rows], dtype=np.float64)
    all_abs_err = np.asarray(
        [abs(float(row[pred_key]) - float(row[true_key])) for row in rows],
        dtype=np.float64,
    )
    if all_std.size == 0:
        return

    gaussian_abs_err_factor = math.sqrt(2.0 / math.pi)
    x_max = max(float(np.max(all_std)), 1e-8)
    y_max = max(float(np.max(all_abs_err)), gaussian_abs_err_factor * x_max, 1e-8)
    ref_x = np.linspace(0.0, x_max, num=200, dtype=np.float64)

    quantiles = np.linspace(0.0, 1.0, num=CALIBRATION_NUM_BINS + 1, dtype=np.float64)
    bin_edges = np.unique(np.quantile(all_std, quantiles))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), squeeze=False)
    ax_scatter, ax_binned, ax_coverage = axes[0]

    for (controller, process_noise_scale), bucket in sorted(groups.items()):
        stds = np.asarray([float(row[std_key]) for row in bucket], dtype=np.float64)
        abs_errs = np.asarray(
            [abs(float(row[pred_key]) - float(row[true_key])) for row in bucket],
            dtype=np.float64,
        )
        label = f"{controller} | noise_scale={process_noise_scale:g}"

        ax_scatter.scatter(
            stds,
            abs_errs,
            s=6,
            alpha=0.08,
            label=label,
            rasterized=len(bucket) > 5000,
        )

        if bin_edges.size >= 2:
            bin_x: list[float] = []
            bin_y: list[float] = []
            for idx in range(bin_edges.size - 1):
                left = float(bin_edges[idx])
                right = float(bin_edges[idx + 1])
                if idx == bin_edges.size - 2:
                    mask = (stds >= left) & (stds <= right)
                else:
                    mask = (stds >= left) & (stds < right)
                if not bool(mask.any()):
                    continue
                bin_x.append(float(np.mean(stds[mask])))
                bin_y.append(float(np.mean(abs_errs[mask])))
            if bin_x:
                ax_binned.plot(bin_x, bin_y, marker="o", linewidth=2, label=label)
        else:
            ax_binned.scatter(
                [float(np.mean(stds))],
                [float(np.mean(abs_errs))],
                s=40,
                label=label,
            )

        coverage = [
            float(np.mean(abs_errs <= k * np.maximum(stds, 1e-12)))
            for k in COVERAGE_KS
        ]
        ax_coverage.plot(COVERAGE_KS, coverage, marker="o", linewidth=2, label=label)

    ref_label = "Gaussian-calibrated E|error|"
    ax_scatter.plot(ref_x, gaussian_abs_err_factor * ref_x, linestyle="--", color="black", linewidth=1.5, label=ref_label)
    ax_binned.plot(ref_x, gaussian_abs_err_factor * ref_x, linestyle="--", color="black", linewidth=1.5, label=ref_label)

    ideal_coverage = [math.erf(k / math.sqrt(2.0)) for k in COVERAGE_KS]
    ax_coverage.plot(COVERAGE_KS, ideal_coverage, linestyle="--", color="black", linewidth=1.5, label="Gaussian ideal coverage")

    ax_scatter.set_xlabel(f"Predicted {param_name} std")
    ax_scatter.set_ylabel(f"|Predicted {param_name} - True {param_name}|")
    ax_scatter.set_title("Std vs Absolute Error")
    ax_scatter.set_xlim(left=0.0)
    ax_scatter.set_ylim(bottom=0.0, top=y_max)
    ax_scatter.grid(True, alpha=0.3)

    ax_binned.set_xlabel(f"Binned predicted {param_name} std")
    ax_binned.set_ylabel(f"Mean |Predicted {param_name} - True {param_name}|")
    ax_binned.set_title("Binned Calibration")
    ax_binned.set_xlim(left=0.0)
    ax_binned.set_ylim(bottom=0.0, top=y_max)
    ax_binned.grid(True, alpha=0.3)

    ax_coverage.set_xlabel("k in mu ± k·std")
    ax_coverage.set_ylabel("Empirical coverage")
    ax_coverage.set_title("Coverage")
    ax_coverage.set_xticks(list(COVERAGE_KS))
    ax_coverage.set_ylim(0.0, 1.0)
    ax_coverage.grid(True, alpha=0.3)

    handles: list[Any] = []
    labels: list[str] = []
    for ax in (ax_scatter, ax_binned, ax_coverage):
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    fig.suptitle(f"{param_name} Uncertainty Calibration")
    fig.legend(handles, labels, loc="lower center", ncol=max(1, min(4, len(labels))))
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.96))
    fig.savefig(output_dir / f"{param_name}_uncertainty_calibration.png", dpi=200)
    plt.close(fig)


def _plot_cumulative_information_vs_cost(trace_rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not trace_rows:
        return

    groups: dict[tuple[str, float], dict[int, list[tuple[float, float]]]] = {}
    for row in trace_rows:
        if "cumulative_info_proxy" not in row or "cumulative_cost" not in row:
            continue
        key = (str(row["controller"]), float(row["process_noise_scale"]))
        step = int(row["step"])
        groups.setdefault(key, {}).setdefault(step, []).append(
            (float(row["cumulative_info_proxy"]), float(row["cumulative_cost"]))
        )

    if not groups:
        return

    plt.figure(figsize=(8, 6))
    for (controller, process_noise_scale), by_step in sorted(groups.items()):
        steps = sorted(by_step)
        info = np.asarray([np.mean([v[0] for v in by_step[step]]) for step in steps], dtype=np.float64)
        cost = np.asarray([np.mean([v[1] for v in by_step[step]]) for step in steps], dtype=np.float64)
        plt.plot(cost, info, linewidth=2, label=f"{controller} | noise_scale={process_noise_scale:g}")

    plt.xlabel("Cumulative cost")
    plt.ylabel("Cumulative information proxy")
    plt.title("Information Proxy vs Cost")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_information_vs_cost.png", dpi=200)
    plt.close()


def _plot_trajectory_diagnostics(trace_rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not trace_rows:
        return

    groups: dict[tuple[str, float], dict[int, list[dict[str, Any]]]] = {}
    for row in trace_rows:
        key = (str(row["controller"]), float(row["process_noise_scale"]))
        groups.setdefault(key, {}).setdefault(int(row["step"]), []).append(row)

    if not groups:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_info, ax_cum_info, ax_delta, ax_info_action_cost = axes.ravel()

    for (controller, process_noise_scale), by_step in sorted(groups.items()):
        steps = sorted(by_step)
        label = f"{controller} | noise_scale={process_noise_scale:g}"

        info = np.asarray([
            np.mean([float(r["info_proxy"]) for r in by_step[step] if np.isfinite(float(r["info_proxy"]))])
            for step in steps
        ], dtype=np.float64)
        cum_info = np.asarray([
            np.mean([float(r["cumulative_info_proxy"]) for r in by_step[step]])
            for step in steps
        ], dtype=np.float64)
        action_cost = np.asarray([
            np.mean([float(r["cumulative_action_cost"]) for r in by_step[step]])
            for step in steps
        ], dtype=np.float64)

        delta_cols = sorted(
            {
                key
                for rows in by_step.values()
                for r in rows
                for key in r
                if key.startswith("u_delta_lqr")
            }
        )
        if delta_cols:
            delta = np.asarray([
                np.mean([
                    math.sqrt(sum(float(r[col]) ** 2 for col in delta_cols))
                    for r in by_step[step]
                ])
                for step in steps
            ], dtype=np.float64)
        else:
            delta = np.full(len(steps), np.nan, dtype=np.float64)

        ax_info.plot(steps, info, linewidth=2, label=label)
        ax_cum_info.plot(steps, cum_info, linewidth=2, label=label)
        ax_delta.plot(steps, delta, linewidth=2, label=label)
        ax_info_action_cost.plot(action_cost, cum_info, linewidth=2, label=label)

    ax_info.set_title("Information Proxy Over Time")
    ax_info.set_xlabel("step")
    ax_info.set_ylabel(r"$||K_{nom}x + u||^2$")
    ax_cum_info.set_title("Cumulative Information Proxy")
    ax_cum_info.set_xlabel("step")
    ax_cum_info.set_ylabel("cumulative proxy")
    ax_delta.set_title("Deviation From LQR Action")
    ax_delta.set_xlabel("step")
    ax_delta.set_ylabel(r"$||u - u_{LQR}||$")
    ax_info_action_cost.set_title("Information vs Action Cost")
    ax_info_action_cost.set_xlabel("cumulative action cost")
    ax_info_action_cost.set_ylabel("cumulative proxy")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    handles, labels = ax_info.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=max(1, min(3, len(labels))))
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(output_dir / "trajectory_diagnostics.png", dpi=200)
    plt.close(fig)


def _aggregate_lookup(
    rows: list[dict[str, Any]],
    controller: str,
) -> dict[tuple[float, float], dict[str, Any]]:
    return {
        (float(row["process_noise_scale"]), float(row["theta"])): row
        for row in rows
        if str(row.get("controller")) == controller
    }


def _mean_or_none(values: list[float]) -> float | None:
    values = [float(v) for v in values if np.isfinite(float(v))]
    if not values:
        return None
    return float(np.mean(values))


def _format_optional(value: float | None, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}"


def _load_external_baseline_rows() -> list[dict[str, Any]]:
    if not BASELINE_AGGREGATE_PATH:
        return []
    return _read_csv_rows(Path(BASELINE_AGGREGATE_PATH))


def _build_controller_scorecard(
    agg_rows: list[dict[str, Any]],
    external_baseline_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    external_baseline_rows = external_baseline_rows or []
    controllers = sorted({str(row["controller"]) for row in agg_rows})
    lqr_lookup = _aggregate_lookup(agg_rows, "lqr")

    baseline_lookup = _aggregate_lookup(agg_rows, BASELINE_CONTROLLER_FOR_GAIN)
    if not baseline_lookup and external_baseline_rows:
        baseline_lookup = _aggregate_lookup(
            external_baseline_rows,
            BASELINE_CONTROLLER_FOR_GAIN,
        )

    scorecard: list[dict[str, Any]] = []
    for controller in controllers:
        rows = [row for row in agg_rows if str(row["controller"]) == controller]
        all_returns = [float(row["episode_return_mean"]) for row in rows]
        tail_rows = [
            row for row in rows
            if abs(float(row["theta"])) >= TAIL_ABS_THETA_THRESHOLD
        ]
        center_rows = [
            row for row in rows
            if abs(float(row["theta"])) <= CENTER_ABS_THETA_THRESHOLD
        ]

        def keyed_values(reference: dict[tuple[float, float], dict[str, Any]]) -> tuple[list[float], list[float], list[float]]:
            all_vals: list[float] = []
            tail_vals: list[float] = []
            center_vals: list[float] = []
            for row in rows:
                key = (float(row["process_noise_scale"]), float(row["theta"]))
                ref = reference.get(key)
                if ref is None:
                    continue
                delta = float(row["episode_return_mean"]) - float(ref["episode_return_mean"])
                all_vals.append(delta)
                abs_theta = abs(float(row["theta"]))
                if abs_theta >= TAIL_ABS_THETA_THRESHOLD:
                    tail_vals.append(delta)
                if abs_theta <= CENTER_ABS_THETA_THRESHOLD:
                    center_vals.append(delta)
            return all_vals, tail_vals, center_vals

        lqr_gaps, lqr_tail_gaps, lqr_center_gaps = keyed_values(lqr_lookup)
        baseline_gains, baseline_tail_gains, baseline_center_gains = keyed_values(baseline_lookup)

        rmse_key = "theta_rmse_tail_across_episodes"
        rmse_values = [float(row[rmse_key]) for row in rows if row.get(rmse_key) not in {None, ""}]
        tail_rmse_values = [
            float(row[rmse_key])
            for row in tail_rows
            if row.get(rmse_key) not in {None, ""}
        ]
        center_rmse_values = [
            float(row[rmse_key])
            for row in center_rows
            if row.get(rmse_key) not in {None, ""}
        ]
        info_key = "episode_info_proxy_mean"
        info_values = [float(row[info_key]) for row in rows if row.get(info_key) not in {None, ""}]
        tail_info_values = [
            float(row[info_key])
            for row in tail_rows
            if row.get(info_key) not in {None, ""}
        ]
        center_info_values = [
            float(row[info_key])
            for row in center_rows
            if row.get(info_key) not in {None, ""}
        ]

        scorecard.append(
            {
                "controller": controller,
                "num_theta_points": len(rows),
                "tail_abs_theta_threshold": TAIL_ABS_THETA_THRESHOLD,
                "center_abs_theta_threshold": CENTER_ABS_THETA_THRESHOLD,
                "mean_return": _mean_or_none(all_returns),
                "tail_mean_return": _mean_or_none([float(row["episode_return_mean"]) for row in tail_rows]),
                "center_mean_return": _mean_or_none([float(row["episode_return_mean"]) for row in center_rows]),
                "mean_gap_to_lqr": _mean_or_none(lqr_gaps),
                "tail_gap_to_lqr": _mean_or_none(lqr_tail_gaps),
                "center_gap_to_lqr": _mean_or_none(lqr_center_gaps),
                "baseline_controller": BASELINE_CONTROLLER_FOR_GAIN,
                "mean_gain_over_baseline": _mean_or_none(baseline_gains),
                "tail_gain_over_baseline": _mean_or_none(baseline_tail_gains),
                "center_gain_over_baseline": _mean_or_none(baseline_center_gains),
                "theta_rmse": _mean_or_none(rmse_values),
                "tail_theta_rmse": _mean_or_none(tail_rmse_values),
                "center_theta_rmse": _mean_or_none(center_rmse_values),
                "mean_info_proxy": _mean_or_none(info_values),
                "tail_mean_info_proxy": _mean_or_none(tail_info_values),
                "center_mean_info_proxy": _mean_or_none(center_info_values),
            }
        )

    return scorecard


def _write_summary(path: Path, episode_rows: list[dict[str, Any]], agg_rows: list[dict[str, Any]]) -> None:
    controller_names = sorted({str(row["controller"]) for row in episode_rows})
    external_baseline_rows = _load_external_baseline_rows()
    scorecard_rows = _build_controller_scorecard(
        agg_rows,
        external_baseline_rows=external_baseline_rows,
    )
    lines = [
        f"target_experiment: {TARGET_EXPERIMENT}",
        f"num_episode_rows: {len(episode_rows)}",
        f"num_aggregate_rows: {len(agg_rows)}",
        f"controllers: {controller_names}",
        f"process_noise_scales: {PROCESS_NOISE_SCALES}",
        f"tail_abs_theta_threshold: {TAIL_ABS_THETA_THRESHOLD}",
        f"center_abs_theta_threshold: {CENTER_ABS_THETA_THRESHOLD}",
        f"baseline_controller_for_gain: {BASELINE_CONTROLLER_FOR_GAIN}",
    ]
    if BASELINE_AGGREGATE_PATH:
        lines.append(f"baseline_aggregate_path: {BASELINE_AGGREGATE_PATH}")

    for controller in controller_names:
        controller_rows = [row for row in episode_rows if str(row["controller"]) == controller]
        returns = np.asarray([float(row["episode_return"]) for row in controller_rows], dtype=np.float64)
        lines.append(
            f"{controller}: mean_return={np.mean(returns):.4f}, std_return={np.std(returns):.4f}, "
            f"median_return={np.median(returns):.4f}, num_episodes={len(controller_rows)}"
        )

        theta_rmse_keys = sorted(
            {
                key
                for row in controller_rows
                for key in row.keys()
                if key.endswith("_rmse_tail")
            }
        )
        for key in theta_rmse_keys:
            values = np.asarray([float(row[key]) for row in controller_rows], dtype=np.float64)
            lines.append(f"{controller}: mean_{key}={np.mean(values):.6f}")

    lines.append("")
    lines.append("scorecard:")
    for row in scorecard_rows:
        lines.append(
            f"{row['controller']}: "
            f"mean_return={_format_optional(row['mean_return'])}, "
            f"tail_return={_format_optional(row['tail_mean_return'])}, "
            f"center_return={_format_optional(row['center_mean_return'])}, "
            f"tail_gap_to_lqr={_format_optional(row['tail_gap_to_lqr'])}, "
            f"tail_gain_over_{row['baseline_controller']}="
            f"{_format_optional(row['tail_gain_over_baseline'])}, "
            f"tail_theta_rmse={_format_optional(row['tail_theta_rmse'], precision=6)}"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    target_cfg = _load_yaml(Path(TARGET_EXPERIMENT) / "config.yaml")
    theta_grid = _resolve_theta_grid(target_cfg)

    output_dir = Path(TARGET_EXPERIMENT) / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    runtimes: list[ControllerRuntime] = []
    for spec in CONTROLLER_SPECS:
        runtime = build_controller(spec, target_cfg)
        if runtime is not None:
            runtimes.append(runtime)

    if not runtimes:
        raise RuntimeError("No valid controllers configured for the sweep.")

    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []

    total_runs = len(runtimes) * len(PROCESS_NOISE_SCALES) * len(theta_grid) * EPISODES_PER_THETA
    progress = tqdm(total=total_runs, desc="theta sweep")

    try:
        for runtime in runtimes:
            for process_noise_scale in PROCESS_NOISE_SCALES:
                for theta in theta_grid:
                    for ep_idx in range(EPISODES_PER_THETA):
                        seed = EVAL_BASE_SEED + ep_idx
                        episode_row, step_prediction_rows, trace_rows = run_episode(
                            runtime=runtime,
                            theta=float(theta),
                            process_noise_scale=float(process_noise_scale),
                            seed=seed,
                        )
                        episode_rows.append(episode_row)
                        collect_for_calibration = (
                            COLLECT_STEP_LEVEL_PREDICTIONS
                            and runtime.kind == "ppo"
                            and isinstance(runtime.model, UnifiedContextPPO)
                            and bool(getattr(runtime.model, "condition_on_uncertainty", False))
                        )
                        if SAVE_STEP_LEVEL_CSV or collect_for_calibration:
                            step_rows.extend(step_prediction_rows)
                        if SAVE_TRAJECTORY_TRACE:
                            trajectory_rows.extend(trace_rows)
                        progress.update(1)
    finally:
        progress.close()
        for runtime in runtimes:
            runtime.env.close()

    agg_rows = _group_episode_rows(episode_rows)
    external_baseline_rows = _load_external_baseline_rows()
    scorecard_rows = _build_controller_scorecard(
        agg_rows,
        external_baseline_rows=external_baseline_rows,
    )

    _write_csv(output_dir / "episode_summary.csv", episode_rows)
    if SAVE_STEP_LEVEL_CSV:
        _write_csv(output_dir / "step_predictions.csv", step_rows)
    if SAVE_TRAJECTORY_TRACE:
        _write_csv(output_dir / "trajectory_trace.csv", trajectory_rows)
    _write_csv(output_dir / "theta_sweep_aggregate.csv", agg_rows)
    _write_csv(output_dir / "controller_scorecard.csv", scorecard_rows)
    _write_summary(output_dir / "summary.txt", episode_rows, agg_rows)

    _plot_return_vs_theta(agg_rows, output_dir)
    _plot_param_prediction_vs_theta(agg_rows, output_dir, param_name="theta")
    _plot_param_rmse_vs_theta(agg_rows, output_dir, param_name="theta")
    _plot_param_uncertainty_vs_theta(agg_rows, output_dir, param_name="theta")
    _plot_uncertainty_calibration(step_rows, runtimes, output_dir, param_name="theta")
    if SAVE_TRAJECTORY_TRACE:
        _plot_cumulative_information_vs_cost(trajectory_rows, output_dir)
        _plot_trajectory_diagnostics(trajectory_rows, output_dir)

    print(f"Wrote sweep outputs to: {output_dir}")


if __name__ == "__main__":
    main()
