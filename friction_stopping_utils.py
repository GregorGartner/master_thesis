from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv

from lqr_model import StoppingCarEnv, install_lqr_adapter_for_domain_randomization
from run_two_action_pipeline import CONFIG_PATH, ROOT
from unified_context_ppo import UnifiedContextPPO
from wrappers import DomainRandomizationWrapper


TRAIN_CMD = ["python3", str(ROOT / "cartpole_ppo_sb3_training.py")]
OUT_ROOT = ROOT / "experiments" / "friction_stopping"
MU_VALUES = [0.25, 0.5, 0.7]


def timestamp() -> str:
    return datetime.now().strftime("%m-%d__%H-%M")


def smoke_enabled() -> bool:
    return (
        os.environ.get("FRICTION_STOPPING_SMOKE", "0").lower() in {"1", "true", "yes"}
        or os.environ.get("FRICTION_STOPPING_V2_SMOKE", "0").lower() in {"1", "true", "yes"}
        or os.environ.get("FRICTION_STOPPING_V3_SMOKE", "0").lower() in {"1", "true", "yes"}
        or os.environ.get("FRICTION_STOPPING_V4_SMOKE", "0").lower() in {"1", "true", "yes"}
        or os.environ.get("FRICTION_STOPPING_V5_SMOKE", "0").lower() in {"1", "true", "yes"}
        or os.environ.get("FRICTION_STOPPING_V6_SMOKE", "0").lower() in {"1", "true", "yes"}
        or os.environ.get("FRICTION_STOPPING_SWEEP_SMOKE", "0").lower() in {"1", "true", "yes"}
    )


def steps(value: int) -> int:
    return 16_384 if smoke_enabled() else int(value)


def env_config(
    *,
    phase: str = "final",
    start_visible: bool = False,
    visible_distance_low: float | None = None,
    visible_distance_high: float | None = None,
    max_episode_steps: int = 200,
    event_time_low: int = 50,
    event_time_high: int = 180,
    initial_velocity_low: float = 2.4,
    initial_velocity_high: float = 2.6,
    brake_accel_max: float = 8.0,
    throttle_accel_max: float = 2.0,
    saturate_throttle_by_friction: bool = False,
    sigma_v: float = 0.5,
    action_cost_weight: float = 0.02,
    safety_cost_weight: float = 5.0,
    crash_penalty: float = 50.0,
    crash_remaining_penalty: float = 50.0,
    success_bonus: float = 5.0,
) -> dict[str, Any]:
    if visible_distance_low is None or visible_distance_high is None:
        if phase == "easy":
            visible_distance_low, visible_distance_high = 1.3, 1.8
        elif phase == "medium":
            visible_distance_low, visible_distance_high = 1.1, 1.4
        elif phase == "start_visible":
            visible_distance_low, visible_distance_high = 1.5, 2.0
        else:
            visible_distance_low, visible_distance_high = 0.9, 1.2
    return {
        "dt": 0.05,
        "mu": 0.5,
        "process_noise_std": 0.0,
        "initial_velocity_low": float(initial_velocity_low),
        "initial_velocity_high": float(initial_velocity_high),
        "action_low": -1.0,
        "action_high": 1.0,
        "max_episode_steps": int(max_episode_steps),
        "v_ref": 2.5,
        "sigma_v": float(sigma_v),
        "d_max": 5.0,
        "event_time_mode": "start_visible" if start_visible else "uniform_once",
        "event_time_low": int(event_time_low),
        "event_time_high": int(event_time_high),
        "visible_distance_low": float(visible_distance_low),
        "visible_distance_high": float(visible_distance_high),
        "start_visible": bool(start_visible),
        "g": 9.81,
        "brake_accel_max": float(brake_accel_max),
        "throttle_accel_max": float(throttle_accel_max),
        "saturate_throttle_by_friction": bool(saturate_throttle_by_friction),
        "speed_reward_weight": 1.0,
        "action_cost_weight": float(action_cost_weight),
        "safety_cost_weight": float(safety_cost_weight),
        "safety_alpha": 5.0,
        "d_buffer": 0.15,
        "v_safe": 0.1,
        "crash_penalty": float(crash_penalty),
        "crash_remaining_penalty": float(crash_remaining_penalty),
        "success_bonus": float(success_bonus),
        "terminate_on_success": True,
        "terminate_on_crash": True,
    }


def domain_randomization_wrapper() -> list[dict[str, Any]]:
    return [
        {
            "name": "DomainRandomizationWrapper",
            "enabled": True,
            "params": {
                "change_prob": 1.0,
                "only_at_episode_end": True,
                "randomize_on_reset": True,
                "randomize_theta": False,
                "randomize_mu": True,
                "mu_range": MU_VALUES,
                "randomize_process_noise_scale": False,
                "process_noise_scale_mult_range": [1.0],
                "categorical": True,
            },
        }
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def run_training_stage(label: str, cfg: dict[str, Any]) -> None:
    print(f"START friction stopping {label}: {cfg['training']['experiment_name']} steps={cfg['total_timesteps']}", flush=True)
    with CONFIG_PATH.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")
    subprocess.run(TRAIN_CMD, cwd=ROOT, env=env, check=True)
    print(f"END   friction stopping {label}", flush=True)


def make_base_cfg(
    base_cfg: dict[str, Any],
    *,
    exp_root: Path,
    exp_name: str,
    total_timesteps: int,
    phase: str = "final",
    start_visible: bool = False,
    env_overrides: dict[str, Any] | None = None,
    model_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
    cfg["environment"] = "stopping_car"
    cfg["stopping_car_env"] = env_config(phase=phase, start_visible=start_visible, **(env_overrides or {}))
    cfg["max_episode_steps"] = cfg["stopping_car_env"]["max_episode_steps"]
    cfg["wrappers"] = domain_randomization_wrapper()
    cfg["total_timesteps"] = int(total_timesteps)
    cfg["training"]["experiment_root"] = str(exp_root)
    cfg["training"]["experiment_name"] = exp_name
    cfg["training"]["experiment_name_suffix"] = None
    cfg["training"]["load_weights"] = False
    cfg["training"]["load_weights_from"] = None
    cfg["training"]["load_weights_name"] = "weights"
    cfg["training"]["load_encoder_only"] = False
    cfg["model"]["name"] = "UnifiedContextPPO"

    params = cfg["model"]["params"]
    params["policy"] = "MlpPolicy"
    params["regression_param_names"] = ["mu"]
    params["latent_dim"] = 1
    params["window_length"] = 30
    params["nominal_warmup_steps"] = 29
    params["id_update_interval"] = 5
    params["initial_context"] = None
    params["z_scale"] = 1.0
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
    params["learning_rate"] = 3e-4
    params["n_steps"] = 2048
    params["batch_size"] = 512
    params["n_epochs"] = 8
    params["gamma"] = 0.995
    params["gae_lambda"] = 0.95
    params["clip_range"] = 0.2
    params["normalize_advantage"] = True
    params["ent_coef"] = 0.02
    params["vf_coef"] = 0.5
    params["max_grad_norm"] = 1.0
    params["target_kl"] = None
    params["verbose"] = 1
    params["device"] = "cpu"
    params["uncertainty_regularization_coef"] = 0.0
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["uncertainty_penalty_metric"] = "std"
    if model_overrides:
        params.update(model_overrides)
    for cb in cfg.get("callbacks", []):
        if cb.get("name") == "LivePlotCallback":
            cb["enabled"] = True
            cb.setdefault("params", {})["update_every_episodes"] = 200
        if cb.get("name") == "SaveModelCallback":
            cb["enabled"] = True
            cb.setdefault("params", {})["save_on_training_end"] = True
            cb["params"]["best_metric"] = "episode_reward"
    return cfg


def set_load(cfg: dict[str, Any], source: Path | None, weights_name: str = "weights") -> None:
    cfg["training"]["load_weights"] = source is not None
    cfg["training"]["load_weights_from"] = str(source.resolve()) if source is not None else None
    cfg["training"]["load_weights_name"] = weights_name
    cfg["training"]["load_encoder_only"] = False


def set_privileged(cfg: dict[str, Any], *, ent: float = 0.02, lr: float = 3e-4) -> None:
    params = cfg["model"]["params"]
    params["context_mode"] = "privileged"
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["condition_on_uncertainty"] = False
    params["privileged_uncertainty_mode"] = "zeros"
    params["privileged_context_probability"] = 0.0
    params["learning_rate"] = float(lr)
    params["ent_coef"] = float(ent)


def set_encoder_pretrain(cfg: dict[str, Any], *, method: str) -> None:
    params = cfg["model"]["params"]
    params["context_mode"] = "encoder_nll" if method == "nll" else "encoder_mle"
    params["freeze_ppo"] = True
    params["regression_coef"] = 1.0
    params["condition_on_uncertainty"] = method == "nll"
    params["privileged_uncertainty_mode"] = "predicted" if method == "nll" else "zeros"
    params["privileged_context_probability"] = 0.0
    params["naive_action_noise_std"] = [0.0, 0.45]
    params["naive_action_noise_dist"] = ["gaussian", "uniform"]
    params["learning_rate"] = 1e-5
    params["ent_coef"] = 0.0
    for cb in cfg.get("callbacks", []):
        if cb.get("name") == "SaveModelCallback":
            cb.setdefault("params", {})["best_metric"] = "regression_mse"


def set_gradual_policy(
    cfg: dict[str, Any],
    *,
    method: str,
    condition_on_uncertainty: bool,
    encoder_probability: float,
    ent: float,
    lr: float,
    freeze_encoder: bool = False,
) -> None:
    params = cfg["model"]["params"]
    params["context_mode"] = "encoder_nll" if method == "nll" else "encoder_mle"
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0 if freeze_encoder else 1.0
    params["condition_on_uncertainty"] = bool(condition_on_uncertainty)
    params["privileged_uncertainty_mode"] = "predicted" if condition_on_uncertainty else "zeros"
    params["privileged_context_probability"] = 1.0 - float(encoder_probability)
    params["naive_action_noise_std"] = 0.0
    params["learning_rate"] = float(lr)
    params["ent_coef"] = float(ent)


def set_staged_policy(cfg: dict[str, Any], *, ent: float, lr: float) -> None:
    params = cfg["model"]["params"]
    params["context_mode"] = "encoder_mle"
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["condition_on_uncertainty"] = False
    params["privileged_uncertainty_mode"] = "zeros"
    params["privileged_context_probability"] = 0.0
    params["naive_action_noise_std"] = 0.0
    params["learning_rate"] = float(lr)
    params["ent_coef"] = float(ent)


def make_env(
    mu: float,
    seed: int,
    *,
    phase: str = "final",
    start_visible: bool = False,
    env_overrides: dict[str, Any] | None = None,
) -> StoppingCarEnv:
    env = StoppingCarEnv(**env_config(phase=phase, start_visible=start_visible, **(env_overrides or {})))
    env.set_friction(mu)
    env.reset(seed=seed)
    return env


def commanded_accel_from_config(action: float, env_cfg: dict[str, Any]) -> float:
    brake_accel_max = float(env_cfg.get("brake_accel_max", 8.0))
    throttle_accel_max = float(env_cfg.get("throttle_accel_max", 2.0))
    u = float(np.clip(action, -1.0, 1.0))
    return float(brake_accel_max * u if u < 0.0 else throttle_accel_max * u)


def low_friction_saturation_thresholds(env_cfg: dict[str, Any]) -> tuple[float, float]:
    g = float(env_cfg.get("g", 9.81))
    mu_min = min(MU_VALUES)
    brake_accel_max = float(env_cfg.get("brake_accel_max", 8.0))
    throttle_accel_max = float(env_cfg.get("throttle_accel_max", 2.0))
    brake_threshold = -float(mu_min * g / max(brake_accel_max, 1e-8))
    throttle_threshold = float(mu_min * g / max(throttle_accel_max, 1e-8))
    return brake_threshold, throttle_threshold


def saturation_candidate_mask(actions: np.ndarray, env_cfg: dict[str, Any]) -> np.ndarray:
    brake_threshold, throttle_threshold = low_friction_saturation_thresholds(env_cfg)
    actions = np.asarray(actions, dtype=np.float64)
    mask = actions <= brake_threshold
    if bool(env_cfg.get("saturate_throttle_by_friction", False)):
        mask = np.logical_or(mask, actions >= throttle_threshold)
    return mask


def speed_control_action(env: StoppingCarEnv, target_v: float, kp: float = 2.0) -> float:
    desired_accel = kp * (float(target_v) - float(env.velocity))
    if desired_accel >= 0.0:
        u = desired_accel / env.throttle_accel_max
    else:
        u = desired_accel / env.brake_accel_max
    return float(np.clip(u, env.action_space.low[0], env.action_space.high[0]))


def safe_target_speed(env: StoppingCarEnv, assumed_mu: float) -> float:
    available_d = max(env.visible_distance_low - env.d_buffer, 1e-6)
    safe_v = 0.92 * math.sqrt(2.0 * assumed_mu * env.g * available_d)
    return float(min(env.v_ref, safe_v))


def controller_action(env: StoppingCarEnv, name: str, step: int) -> float:
    if name == "coast_cruise":
        return speed_control_action(env, env.v_ref)
    if name == "true_mu_safe":
        target = safe_target_speed(env, env.mu)
    elif name == "nominal_mu":
        target = safe_target_speed(env, 0.5)
    elif name == "always_conservative":
        target = safe_target_speed(env, min(MU_VALUES))
    elif name == "periodic_probe":
        if (not env.obstacle_visible) and step % 30 in {0, 1}:
            return -1.0
        target = env.v_ref
    else:
        raise ValueError(f"Unknown controller: {name}")

    if env.obstacle_visible:
        margin = env.stopping_distance() + env.d_buffer - env.distance
        if margin >= -0.05 or env.velocity > env.v_safe:
            return -1.0
        return 0.0
    return speed_control_action(env, target)


def rollout_handcoded(
    controller: str,
    *,
    mu: float,
    seed: int,
    phase: str = "final",
    env_overrides: dict[str, Any] | None = None,
    collect_trace: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = make_env(mu, seed, phase=phase, env_overrides=env_overrides)
    obs, _ = env.reset(seed=seed)
    rows: list[dict[str, Any]] = []
    totals: dict[str, float] = {
        "return": 0.0,
        "speed_reward": 0.0,
        "action_cost": 0.0,
        "safety_cost": 0.0,
        "crash_cost": 0.0,
        "success_reward": 0.0,
        "strong_probe": 0.0,
    }
    done = False
    step = 0
    while not done:
        u = controller_action(env, controller, step)
        obs, reward, terminated, truncated, info = env.step(np.asarray([u], dtype=np.float32))
        totals["return"] += float(reward)
        for key in ["speed_reward", "action_cost", "safety_cost", "crash_cost", "success_reward", "strong_probe"]:
            totals[key] += float(info.get(key, 0.0))
        if collect_trace:
            rows.append(
                {
                    "controller": controller,
                    "mu": float(mu),
                    "seed": int(seed),
                    "step": int(step),
                    "u": float(u),
                    "v": float(info["velocity"]),
                    "d": float(info["distance"]),
                    "b": float(info["obstacle_visible"]),
                    "reward": float(reward),
                    "realized_accel": float(info["realized_accel"]),
                    "stopping_margin": float(info["stopping_margin"]),
                    "crashed": float(info["crashed"]),
                    "success": float(info["success"]),
                }
            )
        done = bool(terminated or truncated)
        step += 1
    summary = {
        "controller": controller,
        "mu": float(mu),
        "seed": int(seed),
        "steps": int(step),
        "crashed": float(info.get("crashed", 0.0)),
        "success": float(info.get("success", 0.0)),
        **totals,
    }
    return summary, rows


def run_sanity_checks(output_dir: Path, *, env_overrides: dict[str, Any] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for mu in MU_VALUES:
        env = make_env(mu, seed=100, env_overrides=env_overrides)
        for u in [-0.2, 0.2, -1.0, 1.0]:
            commanded_accel = float(env._commanded_accel(u))
            saturated = bool(commanded_accel < -mu * env.g)
            if env.saturate_throttle_by_friction:
                saturated = saturated or bool(commanded_accel > mu * env.g)
            rows.append(
                {
                    "check": "accel_response",
                    "mu": float(mu),
                    "u": float(u),
                    "commanded_accel": commanded_accel,
                    "realized_accel": float(env.realized_accel(u)),
                    "saturated": float(saturated),
                }
            )
    for mu in MU_VALUES:
        env = make_env(mu, seed=200, start_visible=True, env_overrides=env_overrides)
        env.velocity = env.v_ref
        env.distance = 1.0
        env.obstacle_visible = True
        obs = env._obs()
        done = False
        step = 0
        last_info: dict[str, Any] = {}
        while not done and step < 40:
            obs, reward, terminated, truncated, last_info = env.step(np.asarray([-1.0], dtype=np.float32))
            done = bool(terminated or truncated)
            step += 1
        rows.append(
            {
                "check": "late_max_brake",
                "mu": float(mu),
                "u": -1.0,
                "steps": int(step),
                "final_v": float(env.velocity),
                "final_d": float(env.distance),
                "crashed": float(last_info.get("crashed", 0.0)),
                "success": float(last_info.get("success", 0.0)),
            }
        )
    write_csv(output_dir / "sanity_checks.csv", rows)
    _plot_sanity(rows, output_dir / "sanity_checks.png")


def _plot_sanity(rows: list[dict[str, Any]], path: Path) -> None:
    accel_rows = [r for r in rows if r["check"] == "accel_response"]
    late_rows = [r for r in rows if r["check"] == "late_max_brake"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for u in sorted({float(r["u"]) for r in accel_rows}):
        subset = [r for r in accel_rows if float(r["u"]) == u]
        axes[0].plot([r["mu"] for r in subset], [r["realized_accel"] for r in subset], marker="o", label=f"u={u:g}")
    axes[0].set_title("Friction revealed by saturation")
    axes[0].set_xlabel("mu")
    axes[0].set_ylabel("realized acceleration")
    axes[0].legend()
    axes[1].bar([str(r["mu"]) for r in late_rows], [r["success"] - r["crashed"] for r in late_rows])
    axes[1].set_title("Late max braking from d=1.0")
    axes[1].set_xlabel("mu")
    axes[1].set_ylabel("success - crash")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def evaluate_handcoded(
    output_dir: Path,
    *,
    seeds: int = 20,
    phase: str = "final",
    env_overrides: dict[str, Any] | None = None,
) -> None:
    controllers = ["coast_cruise", "nominal_mu", "always_conservative", "periodic_probe", "true_mu_safe"]
    if smoke_enabled():
        seeds = 2
    summaries: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for controller in controllers:
        for mu in MU_VALUES:
            for seed_idx in range(seeds):
                summary, trace = rollout_handcoded(
                    controller,
                    mu=mu,
                    seed=1000 + seed_idx,
                    phase=phase,
                    env_overrides=env_overrides,
                    collect_trace=seed_idx < 2,
                )
                summaries.append(summary)
                traces.extend(trace)
    write_csv(output_dir / "handcoded_rollouts.csv", summaries)
    write_csv(output_dir / "handcoded_traces.csv", traces)
    write_csv(output_dir / "handcoded_scorecard.csv", aggregate_rollouts(summaries, group_key="controller"))
    _plot_scorecard(output_dir / "handcoded_scorecard.csv", output_dir / "handcoded_scorecard.png", title="Hand-coded controllers")
    _plot_representative_traces(traces, output_dir / "handcoded_representative_traces.png")


def aggregate_rollouts(rows: list[dict[str, Any]], *, group_key: str = "controller") -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)
    keys = [
        "return",
        "crashed",
        "success",
        "action_cost",
        "safety_cost",
        "crash_cost",
        "speed_reward",
        "strong_probe",
        "pre_bound_frac",
        "pre_brake_bound_frac",
        "pre_throttle_bound_frac",
    ]
    out: list[dict[str, Any]] = []
    for key, bucket in groups.items():
        row: dict[str, Any] = {group_key: key, "n": len(bucket)}
        for metric in keys:
            vals = np.asarray([float(r.get(metric, 0.0)) for r in bucket], dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals))
        out.append(row)
    return out


def aggregate_behavior_by_mu(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        model = str(row.get("model", row.get("controller", "")))
        mu = float(row["mu"])
        groups.setdefault((model, mu), []).append(row)
    out: list[dict[str, Any]] = []
    for (model, mu), bucket in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        pre = [r for r in bucket if float(r.get("b", 0.0)) == 0.0]
        post = [r for r in bucket if float(r.get("b", 0.0)) == 1.0]
        row: dict[str, Any] = {"model": model, "mu": float(mu), "n_steps": len(bucket), "n_pre": len(pre), "n_post": len(post)}
        for prefix, subset in [("pre", pre), ("post", post)]:
            if subset:
                u_vals = np.asarray([float(r.get("u", r.get("clipped_action", 0.0))) for r in subset], dtype=np.float64)
                v_vals = np.asarray([float(r.get("v", 0.0)) for r in subset], dtype=np.float64)
                row[f"{prefix}_mean_abs_u"] = float(np.mean(np.abs(u_vals)))
                row[f"{prefix}_mean_brake"] = float(np.mean(np.maximum(-u_vals, 0.0)))
                row[f"{prefix}_strong_brake_frac"] = float(np.mean(u_vals <= -0.8))
                row[f"{prefix}_bound_frac"] = float(np.mean(np.abs(u_vals) >= 0.99))
                row[f"{prefix}_brake_bound_frac"] = float(np.mean(u_vals <= -0.99))
                row[f"{prefix}_throttle_bound_frac"] = float(np.mean(u_vals >= 0.99))
                row[f"{prefix}_mean_v"] = float(np.mean(v_vals))
            else:
                row[f"{prefix}_mean_abs_u"] = float("nan")
                row[f"{prefix}_mean_brake"] = float("nan")
                row[f"{prefix}_strong_brake_frac"] = float("nan")
                row[f"{prefix}_bound_frac"] = float("nan")
                row[f"{prefix}_brake_bound_frac"] = float("nan")
                row[f"{prefix}_throttle_bound_frac"] = float("nan")
                row[f"{prefix}_mean_v"] = float("nan")
        if bucket and "sigma" in bucket[0]:
            sigmas = np.asarray([float(r.get("sigma", np.nan)) for r in bucket if not math.isnan(float(r.get("sigma", np.nan)))])
            row["sigma_mean"] = float(np.mean(sigmas)) if sigmas.size else float("nan")
            row["sigma_max"] = float(np.max(sigmas)) if sigmas.size else float("nan")
        out.append(row)
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _plot_scorecard(csv_path: Path, path: Path, *, title: str) -> None:
    if not csv_path.exists():
        return
    rows = _read_csv(csv_path)
    def _label(row: dict[str, str]) -> str:
        for key in ["label", "model", "controller"]:
            value = str(row.get(key, "")).strip()
            if value and value.lower() != "nan":
                return value
        return ""

    labels = [_label(r) for r in rows]
    fig, axes = plt.subplots(3, 4, figsize=(17, 11))
    axes = axes.reshape(-1)
    for ax, metric, ylabel in [
        (axes[0], "return_mean", "return"),
        (axes[1], "crashed_mean", "crash rate"),
        (axes[2], "success_mean", "success rate"),
        (axes[3], "strong_probe_mean", "pre-obstacle probes"),
        (axes[4], "speed_reward_mean", "speed reward"),
        (axes[5], "action_cost_mean", "action cost"),
        (axes[6], "safety_cost_mean", "safety cost"),
        (axes[7], "crash_cost_mean", "crash cost"),
        (axes[8], "pre_bound_frac_mean", "pre |u|>=0.99 frac"),
        (axes[9], "pre_brake_bound_frac_mean", "pre u<=-0.99 frac"),
        (axes[10], "pre_throttle_bound_frac_mean", "pre u>=0.99 frac"),
    ]:
        ax.bar(labels, [float(r.get(metric, 0.0)) for r in rows])
        ax.set_title(ylabel)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    for ax in axes[11:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_behavior_by_mu(rows: list[dict[str, Any]], path: Path, *, title: str) -> None:
    if not rows:
        return
    mus = sorted({float(r["mu"]) for r in rows})
    labels = [str(mu) for mu in mus]
    has_sigma = any("sigma_mean" in r for r in rows)
    metrics = [
        ("pre_mean_v", "pre mean v"),
        ("pre_mean_abs_u", "pre mean |u|"),
        ("pre_mean_brake", "pre mean brake"),
        ("pre_strong_brake_frac", "pre strong-brake frac"),
        ("pre_bound_frac", "pre bound frac"),
        ("pre_brake_bound_frac", "pre brake-bound frac"),
        ("pre_throttle_bound_frac", "pre throttle-bound frac"),
        ("post_mean_v", "post mean v"),
        ("post_mean_abs_u", "post mean |u|"),
        ("post_mean_brake", "post mean brake"),
        ("post_strong_brake_frac", "post strong-brake frac"),
        ("post_bound_frac", "post bound frac"),
    ]
    if has_sigma:
        metrics.extend([("sigma_mean", "sigma mean"), ("sigma_max", "sigma max")])
    ncols = 4
    nrows = int(math.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows))
    axes_flat = np.asarray(axes).reshape(-1)
    for ax, (metric, ylabel) in zip(axes_flat, metrics):
        vals = []
        for mu in mus:
            subset = [r for r in rows if abs(float(r["mu"]) - mu) < 1e-8]
            vals.append(float(subset[0].get(metric, float("nan"))) if subset else float("nan"))
        ax.bar(labels, vals)
        ax.set_title(ylabel)
        ax.set_xlabel("mu")
    for ax in axes_flat[len(metrics) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_representative_traces(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    labels = sorted({str(r["controller"]) for r in rows})
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for label in labels:
        subset = [r for r in rows if str(r["controller"]) == label and int(r["seed"]) == 1000 and abs(float(r["mu"]) - 0.25) < 1e-8]
        if not subset:
            continue
        steps = [int(r["step"]) for r in subset]
        axes[0].plot(steps, [float(r["v"]) for r in subset], label=label)
        axes[1].plot(steps, [float(r["d"]) for r in subset], label=label)
        axes[2].plot(steps, [float(r["u"]) for r in subset], label=label)
    axes[0].set_ylabel("v")
    axes[1].set_ylabel("d")
    axes[2].set_ylabel("u")
    axes[2].set_xlabel("step")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _dummy_vec_for_model(cfg: dict[str, Any]) -> DummyVecEnv:
    install_lqr_adapter_for_domain_randomization()

    def make_wrapped():
        env = StoppingCarEnv(**(cfg.get("stopping_car_env", {}) or {}))
        for wr_spec in cfg.get("wrappers", []):
            if wr_spec.get("enabled", True) is False:
                continue
            if wr_spec["name"] == "DomainRandomizationWrapper":
                env = DomainRandomizationWrapper(env, **(wr_spec.get("params", {}) or {}))
        return env

    return DummyVecEnv([make_wrapped])


def load_model(exp_dir: Path, weights_name: str = "weights") -> tuple[UnifiedContextPPO, dict[str, Any]]:
    with (exp_dir / "config.yaml").open("r") as f:
        cfg = yaml.safe_load(f)
    env = _dummy_vec_for_model(cfg)
    model = UnifiedContextPPO.load(str(exp_dir / weights_name), env=env)
    model.set_env(env)
    return model, cfg


def rollout_model(
    model: UnifiedContextPPO,
    cfg: dict[str, Any],
    *,
    mu: float,
    seed: int,
    collect_trace: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = StoppingCarEnv(**(cfg.get("stopping_car_env", {}) or env_config()))
    env.set_friction(mu)
    model.set_env(DummyVecEnv([lambda: env]))
    obs, _ = env.reset(seed=seed)
    episode_start = np.asarray([True])
    state = None
    totals: dict[str, float] = {
        "return": 0.0,
        "speed_reward": 0.0,
        "action_cost": 0.0,
        "safety_cost": 0.0,
        "crash_cost": 0.0,
        "success_reward": 0.0,
        "strong_probe": 0.0,
        "pre_steps": 0.0,
        "pre_bound_steps": 0.0,
        "pre_brake_bound_steps": 0.0,
        "pre_throttle_bound_steps": 0.0,
    }
    trace: list[dict[str, Any]] = []
    done = False
    step = 0
    last_info: dict[str, Any] = {}
    while not done:
        action, state = model.predict(
            obs.reshape(1, -1),
            state=state,
            episode_start=episode_start,
            deterministic=True,
        )
        episode_start = np.asarray([False])
        raw_u = float(action.reshape(-1)[0])
        clipped_u = float(np.clip(raw_u, env.action_space.low[0], env.action_space.high[0]))
        obs, reward, terminated, truncated, last_info = env.step(action.reshape(-1))
        totals["return"] += float(reward)
        for key in ["speed_reward", "action_cost", "safety_cost", "crash_cost", "success_reward", "strong_probe"]:
            totals[key] += float(last_info.get(key, 0.0))
        if float(last_info.get("pre_obstacle", 0.0)) > 0.5:
            totals["pre_steps"] += 1.0
            if abs(clipped_u) >= 0.99:
                totals["pre_bound_steps"] += 1.0
            if clipped_u <= -0.99:
                totals["pre_brake_bound_steps"] += 1.0
            if clipped_u >= 0.99:
                totals["pre_throttle_bound_steps"] += 1.0
        if collect_trace:
            row = {
                "mu": float(mu),
                "seed": int(seed),
                "step": int(step),
                "u": float(clipped_u),
                "raw_action": float(raw_u),
                "clipped_action": float(clipped_u),
                "v": float(last_info["velocity"]),
                "d": float(last_info["distance"]),
                "b": float(last_info["obstacle_visible"]),
                "reward": float(reward),
                "commanded_accel": float(last_info["commanded_accel"]),
                "realized_accel": float(last_info["realized_accel"]),
                "brake_saturated": float(last_info.get("brake_saturated", 0.0)),
                "throttle_saturated": float(last_info.get("throttle_saturated", 0.0)),
                "accel_saturated": float(last_info.get("accel_saturated", 0.0)),
                "pre_obstacle": float(last_info.get("pre_obstacle", 0.0)),
                "stopping_margin": float(last_info["stopping_margin"]),
                "crashed": float(last_info["crashed"]),
                "success": float(last_info["success"]),
            }
            cached = getattr(model, "_predict_cached_context", None)
            if cached is not None:
                flat = np.asarray(cached).reshape(-1)
                if flat.size >= 1:
                    row["mu_hat"] = float(flat[0])
                if flat.size >= 2:
                    row["sigma"] = float(flat[1])
            trace.append(row)
        done = bool(terminated or truncated)
        step += 1
    summary = {
        "mu": float(mu),
        "seed": int(seed),
        "steps": int(step),
        "crashed": float(last_info.get("crashed", 0.0)),
        "success": float(last_info.get("success", 0.0)),
        **totals,
    }
    pre_steps = max(float(summary.pop("pre_steps", 0.0)), 1.0)
    summary["pre_bound_frac"] = float(summary.pop("pre_bound_steps", 0.0) / pre_steps)
    summary["pre_brake_bound_frac"] = float(summary.pop("pre_brake_bound_steps", 0.0) / pre_steps)
    summary["pre_throttle_bound_frac"] = float(summary.pop("pre_throttle_bound_steps", 0.0) / pre_steps)
    return summary, trace


def evaluate_model(
    exp_dir: Path,
    output_dir: Path,
    *,
    label: str,
    weights_name: str = "weights",
    seeds: int = 10,
) -> None:
    if smoke_enabled():
        seeds = 2
    model, cfg = load_model(exp_dir, weights_name=weights_name)
    summaries: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for mu in MU_VALUES:
        for seed_idx in range(seeds):
            summary, trace = rollout_model(
                model,
                cfg,
                mu=mu,
                seed=5000 + seed_idx,
                collect_trace=seed_idx < 3,
            )
            summary["model"] = label
            summary["model_mu"] = f"{label}_mu{mu:g}"
            summaries.append(summary)
            for row in trace:
                row["model"] = label
            traces.extend(trace)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "episode_summary.csv", summaries)
    write_csv(output_dir / "trajectory_trace.csv", traces)
    score = aggregate_rollouts(summaries, group_key="model")
    write_csv(output_dir / "scorecard.csv", score)
    by_mu = aggregate_rollouts(summaries, group_key="model_mu")
    write_csv(output_dir / "scorecard_by_mu.csv", by_mu)
    behavior = aggregate_behavior_by_mu(traces)
    write_csv(output_dir / "behavior_by_mu.csv", behavior)
    _plot_scorecard(output_dir / "scorecard.csv", output_dir / "scorecard.png", title=label)
    _plot_behavior_by_mu(behavior, output_dir / "behavior_by_mu.png", title=label)
    _plot_model_traces(traces, output_dir / "representative_traces.png", title=label, cfg=cfg)
    if str(getattr(model, "context_mode", "")) == "encoder_nll" and bool(getattr(model, "condition_on_uncertainty", False)):
        run_fixed_context_intervention(
            model,
            cfg,
            output_dir / "fixed_context_intervention.csv",
            output_dir / "fixed_context_intervention.png",
        )
        run_rollout_state_sigma_intervention(
            model,
            cfg,
            output_dir / "trajectory_trace.csv",
            output_dir / "rollout_state_sigma_intervention.csv",
            output_dir / "rollout_state_sigma_intervention.png",
        )
        run_encoder_calibration_diagnostic(
            model,
            cfg,
            output_dir / "encoder_calibration.csv",
            output_dir / "encoder_calibration.png",
            trace_csv_path=output_dir / "trajectory_trace.csv",
        )
        run_trace_window_information_diagnostic(
            model,
            cfg,
            output_dir / "trajectory_trace.csv",
            output_dir / "trace_window_information.csv",
            output_dir / "trace_window_information_summary.csv",
            output_dir / "trace_window_information.png",
        )


def _plot_model_traces(rows: list[dict[str, Any]], path: Path, *, title: str, cfg: dict[str, Any] | None = None) -> None:
    if not rows:
        return
    env_cfg = (cfg or {}).get("stopping_car_env", {}) or env_config()
    has_mu_hat = any("mu_hat" in r for r in rows)
    has_sigma = any("sigma" in r for r in rows)
    panels = ["v", "d", "u", "a_cmd", "a_real"]
    if has_mu_hat:
        panels.append("mu_hat")
    if has_sigma:
        panels.append("sigma")
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.15 * len(panels) + 1.0), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for mu in MU_VALUES:
        subset = [r for r in rows if abs(float(r["mu"]) - mu) < 1e-8 and int(r["seed"]) == 5000]
        if not subset:
            continue
        steps = [int(r["step"]) for r in subset]
        values = {
            "v": [float(r["v"]) for r in subset],
            "d": [float(r["d"]) for r in subset],
            "u": [float(r["u"]) for r in subset],
            "a_cmd": [
                float(r.get("commanded_accel", commanded_accel_from_config(float(r["u"]), env_cfg)))
                for r in subset
            ],
            "a_real": [float(r.get("realized_accel", np.nan)) for r in subset],
            "mu_hat": [float(r.get("mu_hat", np.nan)) for r in subset],
            "sigma": [float(r.get("sigma", np.nan)) for r in subset],
        }
        for ax, panel in zip(axes, panels):
            ax.plot(steps, values[panel], label=f"mu={mu}")
    for ax, panel in zip(axes, panels):
        ax.set_ylabel(panel)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("step")
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_fixed_context_intervention(
    model: UnifiedContextPPO,
    cfg: dict[str, Any],
    csv_path: Path,
    plot_path: Path,
) -> None:
    env_cfg = cfg.get("stopping_car_env", {}) or env_config()
    obs = th.as_tensor([[2.45, 5.0, 0.0], [2.45, 1.0, 1.0]], dtype=th.float32, device=model.device)
    contexts = [
        ("low_sigma", 0.5, 0.02),
        ("mid_sigma", 0.5, 0.15),
        ("high_sigma", 0.5, 0.35),
        ("known_low", 0.25, 0.02),
        ("known_high", 0.7, 0.02),
    ]
    rows: list[dict[str, Any]] = []
    for obs_idx, obs_label in enumerate(["pre_obstacle", "visible_obstacle"]):
        for label, mu_hat, sigma in contexts:
            z = th.as_tensor([[mu_hat, sigma]], dtype=th.float32, device=model.device)
            action, _, _ = model.policy.forward_with_z(obs[obs_idx : obs_idx + 1], z, deterministic=True)
            raw_action = float(action.detach().cpu().numpy().reshape(-1)[0])
            rows.append(
                {
                    "state": obs_label,
                    "context": label,
                    "mu_hat": float(mu_hat),
                    "sigma": float(sigma),
                    "raw_action": float(raw_action),
                    "clipped_action": float(np.clip(raw_action, -1.0, 1.0)),
                    "commanded_accel": commanded_accel_from_config(raw_action, env_cfg),
                }
            )
    write_csv(csv_path, rows)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    x = np.arange(len(contexts))
    width = 0.35
    for idx, state in enumerate(["pre_obstacle", "visible_obstacle"]):
        action_vals = [float(r["clipped_action"]) for r in rows if r["state"] == state]
        accel_vals = [float(r["commanded_accel"]) for r in rows if r["state"] == state]
        axes[0].bar(x + (idx - 0.5) * width, action_vals, width=width, label=state)
        axes[1].bar(x + (idx - 0.5) * width, accel_vals, width=width, label=state)
    axes[0].set_ylabel("clipped action u")
    axes[0].set_title("Fixed state/mean, varied uncertainty")
    axes[0].legend()
    axes[1].set_ylabel("commanded accel")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c[0] for c in contexts], rotation=25, ha="right")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def run_rollout_state_sigma_intervention(
    model: UnifiedContextPPO,
    cfg: dict[str, Any],
    trace_csv_path: Path,
    csv_path: Path,
    plot_path: Path,
) -> None:
    if str(getattr(model, "context_mode", "")) != "encoder_nll" or not bool(
        getattr(model, "condition_on_uncertainty", False)
    ):
        return
    if not trace_csv_path.exists():
        return
    import pandas as pd

    trace = pd.read_csv(trace_csv_path)
    required = {"mu", "seed", "step", "v", "d", "b", "mu_hat"}
    if trace.empty or not required.issubset(trace.columns):
        return

    sigma_contexts = [("low_sigma", 0.02), ("mid_sigma", 0.15), ("high_sigma", 0.35)]
    mu_hat_bins = [0.0, 0.375, 0.625, 1.0]
    mu_hat_labels = ["low hat", "mid hat", "high hat"]
    env_cfg = cfg.get("stopping_car_env", {}) or env_config()
    low_sat_threshold, high_sat_threshold = low_friction_saturation_thresholds(env_cfg)
    symmetric_saturation = bool(env_cfg.get("saturate_throttle_by_friction", False))

    rows: list[dict[str, Any]] = []
    model.policy.eval()
    with th.no_grad():
        for _, src in trace.iterrows():
            obs = th.as_tensor(
                [[float(src["v"]), float(src["d"]), float(src["b"])]],
                dtype=th.float32,
                device=model.device,
            )
            mu_hat = float(src["mu_hat"])
            for sigma_label, sigma_value in sigma_contexts:
                z = th.as_tensor([[mu_hat, sigma_value]], dtype=th.float32, device=model.device)
                action, _, _ = model.policy.forward_with_z(obs, z, deterministic=True)
                raw_action = float(action.detach().cpu().numpy().reshape(-1)[0])
                clipped_action = float(np.clip(raw_action, -1.0, 1.0))
                commanded_accel = commanded_accel_from_config(clipped_action, env_cfg)
                saturation_candidate = clipped_action < low_sat_threshold
                if symmetric_saturation:
                    saturation_candidate = saturation_candidate or clipped_action > high_sat_threshold
                rows.append(
                    {
                        "mu": float(src["mu"]),
                        "seed": int(src["seed"]),
                        "step": int(src["step"]),
                        "state_region": "visible_obstacle" if float(src["b"]) >= 0.5 else "pre_obstacle",
                        "v": float(src["v"]),
                        "d": float(src["d"]),
                        "b": float(src["b"]),
                        "mu_hat": mu_hat,
                        "original_sigma": float(src.get("sigma", float("nan"))),
                        "sigma_context": sigma_label,
                        "sigma_value": float(sigma_value),
                        "raw_action": raw_action,
                        "clipped_action": clipped_action,
                        "commanded_accel": commanded_accel,
                        "brake_probe_low_sat": float(clipped_action < low_sat_threshold),
                        "saturation_candidate": float(saturation_candidate),
                        "strong_brake": float(clipped_action < -0.5),
                        "full_throttle": float(clipped_action > 0.99),
                        "full_brake": float(clipped_action < -0.99),
                    }
                )

    write_csv(csv_path, rows)
    if not rows:
        return

    df = pd.DataFrame(rows)
    df["mu_hat_bin"] = pd.cut(df["mu_hat"], bins=mu_hat_bins, labels=mu_hat_labels, include_lowest=True)
    sigma_labels = [label for label, _ in sigma_contexts]
    metrics = [
        ("clipped_action", "mean action u"),
        ("commanded_accel", "mean commanded accel"),
        ("saturation_candidate", "sat-candidate frac"),
        ("brake_probe_low_sat", f"brake-probe frac\nu<{low_sat_threshold:.3f}"),
        ("full_throttle", "full-throttle frac\nu>0.99"),
    ]
    fig, axes = plt.subplots(2, len(metrics), figsize=(4.4 * len(metrics), 7.5), sharex=True)
    for row_idx, region in enumerate(["pre_obstacle", "visible_obstacle"]):
        region_df = df[df["state_region"] == region]
        for col_idx, (metric, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for bin_label in mu_hat_labels:
                subset = region_df[region_df["mu_hat_bin"].astype(str) == bin_label]
                vals = []
                ns = []
                for sigma_label in sigma_labels:
                    bucket = subset[subset["sigma_context"] == sigma_label]
                    vals.append(float(bucket[metric].mean()) if len(bucket) else float("nan"))
                    ns.append(int(len(bucket)))
                ax.plot(sigma_labels, vals, marker="o", linewidth=2, label=f"{bin_label}")
                if col_idx == 0:
                    for x_idx, (value, n) in enumerate(zip(vals, ns)):
                        if np.isfinite(value):
                            ax.text(x_idx, value, f"n={n}", fontsize=7, ha="center", va="bottom")
            ax.set_title(f"{region}\n{title}")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(True, alpha=0.25)
            if metric in {"clipped_action", "commanded_accel"}:
                ax.axhline(0.0, color="black", linewidth=0.8)
            if col_idx == 0:
                ax.set_ylabel(region)
            if row_idx == 0 and col_idx == len(metrics) - 1:
                ax.legend(fontsize=8, loc="best")
    fig.suptitle("Counterfactual sigma intervention on real rollout states")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def _trajectory_window_from_actions(
    cfg: dict[str, Any],
    *,
    mu: float,
    actions: list[float],
    seed: int,
) -> np.ndarray:
    window_length = int(cfg["model"]["params"].get("window_length", 30))
    env = StoppingCarEnv(**(cfg.get("stopping_car_env", {}) or env_config()))
    env.set_friction(mu)
    obs, _ = env.reset(seed=seed)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    feature_dim = obs_dim + act_dim + obs_dim
    traj = np.zeros((window_length, feature_dim), dtype=np.float32)
    records: list[np.ndarray] = []
    for raw_u in actions:
        s = obs.astype(np.float32, copy=True)
        a = np.asarray([raw_u], dtype=np.float32)
        next_obs, _, terminated, truncated, _ = env.step(a)
        trans = (next_obs - s).astype(np.float32, copy=False)
        records.append(np.concatenate([s, np.asarray([np.clip(raw_u, -1.0, 1.0)], dtype=np.float32), trans], axis=0))
        obs = next_obs
        if terminated or truncated:
            break
    records = records[-window_length:]
    if records:
        traj[-len(records) :, :] = np.asarray(records, dtype=np.float32)
    return traj


def _empty_trajectory_window(cfg: dict[str, Any]) -> np.ndarray:
    window_length = int(cfg["model"]["params"].get("window_length", 30))
    env = StoppingCarEnv(**(cfg.get("stopping_car_env", {}) or env_config()))
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    return np.zeros((window_length, obs_dim + act_dim + obs_dim), dtype=np.float32)


def _cruise_action(env: StoppingCarEnv, obs: np.ndarray) -> float:
    # Stay well below the low-friction saturation threshold while regulating speed.
    velocity = float(obs[0])
    target_accel = float(np.clip(2.0 * (env.v_ref - velocity), -1.2, 1.2))
    if target_accel >= 0.0:
        return float(np.clip(target_accel / env.throttle_accel_max, -1.0, 1.0))
    return float(np.clip(target_accel / env.brake_accel_max, -1.0, 1.0))


def _trajectory_window_from_action_policy(
    cfg: dict[str, Any],
    *,
    mu: float,
    seed: int,
    total_steps: int,
    action_fn: Callable[[int, StoppingCarEnv, np.ndarray], float],
) -> np.ndarray:
    window_length = int(cfg["model"]["params"].get("window_length", 30))
    env = StoppingCarEnv(**(cfg.get("stopping_car_env", {}) or env_config()))
    env.set_friction(mu)
    obs, _ = env.reset(seed=seed)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    feature_dim = obs_dim + act_dim + obs_dim
    records: list[np.ndarray] = []
    for step_idx in range(int(total_steps)):
        s = obs.astype(np.float32, copy=True)
        raw_u = float(action_fn(step_idx, env, obs))
        clipped_u = float(np.clip(raw_u, -1.0, 1.0))
        next_obs, _, terminated, truncated, _ = env.step(np.asarray([clipped_u], dtype=np.float32))
        trans = (next_obs - s).astype(np.float32, copy=False)
        records.append(np.concatenate([s, np.asarray([clipped_u], dtype=np.float32), trans], axis=0))
        obs = next_obs
        if terminated or truncated:
            break
    traj = np.zeros((window_length, feature_dim), dtype=np.float32)
    records = records[-window_length:]
    if records:
        traj[-len(records) :, :] = np.asarray(records, dtype=np.float32)
    return traj


def _trajectory_windows_from_trace_csv(
    trace_csv_path: Path,
    cfg: dict[str, Any],
    *,
    max_windows_per_mu: int = 3,
) -> list[tuple[float, int, np.ndarray]]:
    if not trace_csv_path.exists():
        return []
    import pandas as pd

    df = pd.read_csv(trace_csv_path)
    required = {"mu", "seed", "step", "u", "v", "d", "b"}
    if df.empty or not required.issubset(df.columns):
        return []
    window_length = int(cfg["model"]["params"].get("window_length", 30))
    env = StoppingCarEnv(**(cfg.get("stopping_car_env", {}) or env_config()))
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    feature_dim = obs_dim + act_dim + obs_dim
    windows: list[tuple[float, int, np.ndarray]] = []
    for mu in MU_VALUES:
        mu_df = df[np.isclose(df["mu"].astype(float), float(mu))]
        count = 0
        for seed, group in mu_df.groupby("seed"):
            group = group.sort_values("step").reset_index(drop=True)
            records: list[np.ndarray] = []
            for idx in range(1, len(group)):
                prev = group.iloc[idx - 1]
                cur = group.iloc[idx]
                s = np.asarray([prev["v"], prev["d"], prev["b"]], dtype=np.float32)
                next_obs = np.asarray([cur["v"], cur["d"], cur["b"]], dtype=np.float32)
                action = np.asarray([np.clip(float(cur["u"]), -1.0, 1.0)], dtype=np.float32)
                records.append(np.concatenate([s, action, next_obs - s], axis=0))
            if not records:
                continue
            traj = np.zeros((window_length, feature_dim), dtype=np.float32)
            records = records[-window_length:]
            traj[-len(records) :, :] = np.asarray(records, dtype=np.float32)
            windows.append((float(mu), int(seed), traj))
            count += 1
            if count >= max_windows_per_mu:
                break
    return windows


def _context_from_traj(model: UnifiedContextPPO, traj_np: np.ndarray) -> np.ndarray:
    traj = th.as_tensor(traj_np[None, :, :], device=model.device, dtype=th.float32)
    context = model.policy.build_context_features(traj)
    return context.detach().cpu().numpy().reshape(-1)


def run_encoder_calibration_diagnostic(
    model: UnifiedContextPPO,
    cfg: dict[str, Any],
    csv_path: Path,
    plot_path: Path,
    trace_csv_path: Path | None = None,
) -> None:
    if str(getattr(model, "context_mode", "")) != "encoder_nll":
        return
    window_length = int(cfg["model"]["params"].get("window_length", 30))
    cases = ["empty_padded", "gentle_cruise", "recent_strong_probe", "probe_left_window"]
    rows: list[dict[str, Any]] = []
    model.policy.eval()
    with th.no_grad():
        for mu in MU_VALUES:
            for label in cases:
                if label == "empty_padded":
                    traj_np = _empty_trajectory_window(cfg)
                elif label == "gentle_cruise":
                    traj_np = _trajectory_window_from_action_policy(
                        cfg,
                        mu=mu,
                        seed=7000,
                        total_steps=window_length,
                        action_fn=lambda _step, env, obs: _cruise_action(env, obs),
                    )
                elif label == "recent_strong_probe":
                    probe_start = max(0, window_length - 8)
                    probe_end = min(window_length, probe_start + 3)
                    traj_np = _trajectory_window_from_action_policy(
                        cfg,
                        mu=mu,
                        seed=7000,
                        total_steps=window_length,
                        action_fn=lambda step, env, obs, lo=probe_start, hi=probe_end: -1.0
                        if lo <= step < hi
                        else _cruise_action(env, obs),
                    )
                else:
                    traj_np = _trajectory_window_from_action_policy(
                        cfg,
                        mu=mu,
                        seed=7000,
                        total_steps=window_length + 12,
                        action_fn=lambda step, env, obs: -1.0 if step < 3 else _cruise_action(env, obs),
                    )
                flat = _context_from_traj(model, traj_np)
                rows.append(
                    {
                        "case": label,
                        "source": "synthetic",
                        "mu": float(mu),
                        "sample": 0,
                        "mu_hat": float(flat[0]) if flat.size >= 1 else float("nan"),
                        "sigma": float(flat[1]) if flat.size >= 2 else float("nan"),
                    }
                )
        if trace_csv_path is not None:
            for mu, seed, traj_np in _trajectory_windows_from_trace_csv(trace_csv_path, cfg):
                flat = _context_from_traj(model, traj_np)
                rows.append(
                    {
                        "case": "sampled_rollout_window",
                        "source": "rollout_trace",
                        "mu": float(mu),
                        "sample": int(seed),
                        "mu_hat": float(flat[0]) if flat.size >= 1 else float("nan"),
                        "sigma": float(flat[1]) if flat.size >= 2 else float("nan"),
                    }
                )
    write_csv(csv_path, rows)

    plot_rows: list[dict[str, Any]] = []
    for case in cases + ["sampled_rollout_window"]:
        for mu in MU_VALUES:
            subset = [r for r in rows if r["case"] == case and abs(float(r["mu"]) - mu) < 1e-8]
            if not subset:
                continue
            plot_rows.append(
                {
                    "case": case,
                    "mu": float(mu),
                    "mu_hat": float(np.mean([float(r["mu_hat"]) for r in subset])),
                    "sigma": float(np.mean([float(r["sigma"]) for r in subset])),
                }
            )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [r for r in cases + ["sampled_rollout_window"] if any(row["case"] == r for row in plot_rows)]
    x = np.arange(len(labels))
    width = 0.25
    for idx, mu in enumerate(MU_VALUES):
        subset = [
            next((r for r in plot_rows if r["case"] == label and abs(float(r["mu"]) - mu) < 1e-8), None)
            for label in labels
        ]
        mu_hat_vals = [float(r["mu_hat"]) if r is not None else np.nan for r in subset]
        sigma_vals = [float(r["sigma"]) if r is not None else np.nan for r in subset]
        axes[0].bar(x + (idx - 1) * width, mu_hat_vals, width=width, label=f"mu={mu}")
        axes[1].bar(x + (idx - 1) * width, sigma_vals, width=width, label=f"mu={mu}")
    axes[0].axhline(0.5, color="k", linewidth=1, linestyle="--", alpha=0.6)
    axes[0].set_title("encoder mean")
    axes[1].set_title("encoder sigma")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_xlabel("window type")
    axes[0].set_ylabel("mu_hat")
    axes[1].set_ylabel("sigma")
    axes[1].legend(fontsize=8)
    fig.suptitle("NLL encoder calibration probes")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def _safe_corr(frame: Any, a: str, b: str, *, method: str = "pearson") -> float:
    if a not in frame.columns or b not in frame.columns or len(frame) < 3:
        return float("nan")
    x = frame[a].astype(float)
    y = frame[b].astype(float)
    if float(x.std(ddof=0)) <= 1e-12 or float(y.std(ddof=0)) <= 1e-12:
        return float("nan")
    return float(x.corr(y, method=method))


def run_trace_window_information_diagnostic(
    model: UnifiedContextPPO,
    cfg: dict[str, Any],
    trace_csv_path: Path,
    csv_path: Path,
    summary_path: Path,
    plot_path: Path,
) -> None:
    if str(getattr(model, "context_mode", "")) != "encoder_nll" or not bool(
        getattr(model, "condition_on_uncertainty", False)
    ):
        return
    if not trace_csv_path.exists():
        return
    import pandas as pd

    df = pd.read_csv(trace_csv_path)
    required = {"mu", "seed", "step", "u", "v", "d", "b", "realized_accel"}
    if df.empty or not required.issubset(df.columns):
        return

    window_length = int(cfg["model"]["params"].get("window_length", 30))
    env_cfg = cfg.get("stopping_car_env", {}) or env_config()
    brake_accel_max = float(env_cfg.get("brake_accel_max", 8.0))
    g = float(env_cfg.get("g", 9.81))
    mu_min = min(MU_VALUES)
    low_saturation_u = -float(mu_min * g / brake_accel_max)
    env = StoppingCarEnv(**env_cfg)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    feature_dim = obs_dim + act_dim + obs_dim

    rows: list[dict[str, Any]] = []
    model.policy.eval()
    with th.no_grad():
        for (mu, seed), group in df.groupby(["mu", "seed"]):
            group = group.sort_values("step").reset_index(drop=True)
            records: list[np.ndarray] = []
            action_hist: list[float] = []
            realized_hist: list[float] = []
            commanded_hist: list[float] = []
            visible_hist: list[float] = []
            for idx in range(1, len(group)):
                prev = group.iloc[idx - 1]
                cur = group.iloc[idx]
                action = float(np.clip(cur["u"], -1.0, 1.0))
                s = np.asarray([prev["v"], prev["d"], prev["b"]], dtype=np.float32)
                next_obs = np.asarray([cur["v"], cur["d"], cur["b"]], dtype=np.float32)
                records.append(
                    np.concatenate([s, np.asarray([action], dtype=np.float32), next_obs - s], axis=0)
                )
                action_hist.append(action)
                realized = float(cur["realized_accel"])
                commanded = commanded_accel_from_config(action, env_cfg)
                realized_hist.append(realized)
                commanded_hist.append(float(commanded))
                visible_hist.append(float(cur["b"]))

                recent_records = records[-window_length:]
                traj = np.zeros((window_length, feature_dim), dtype=np.float32)
                traj[-len(recent_records) :, :] = np.asarray(recent_records, dtype=np.float32)
                flat = _context_from_traj(model, traj)

                win_actions = np.asarray(action_hist[-window_length:], dtype=np.float64)
                win_realized = np.asarray(realized_hist[-window_length:], dtype=np.float64)
                win_commanded = np.asarray(commanded_hist[-window_length:], dtype=np.float64)
                win_visible = np.asarray(visible_hist[-window_length:], dtype=np.float64)
                brake_mask = win_actions < 0.0
                strong_mask = win_actions <= -0.8
                low_sat_candidate_mask = win_actions <= low_saturation_u
                sat_candidate_mask = saturation_candidate_mask(win_actions, env_cfg)
                saturation_gap = np.abs(win_realized - win_commanded)
                informative_gap = saturation_gap[sat_candidate_mask]

                mu_hat = float(flat[0]) if flat.size >= 1 else float("nan")
                sigma = float(flat[1]) if flat.size >= 2 else float("nan")
                abs_error = abs(mu_hat - float(mu))
                rows.append(
                    {
                        "mu": float(mu),
                        "seed": int(seed),
                        "end_step": int(cur["step"]),
                        "window_filled_frac": float(min(len(records), window_length) / window_length),
                        "mu_hat": mu_hat,
                        "sigma": sigma,
                        "abs_error": float(abs_error),
                        "squared_error": float(abs_error * abs_error),
                        "mean_abs_u": float(np.mean(np.abs(win_actions))) if win_actions.size else float("nan"),
                        "brake_frac": float(np.mean(brake_mask)) if win_actions.size else float("nan"),
                        "strong_brake_count": int(np.sum(strong_mask)),
                        "strong_brake_frac": float(np.mean(strong_mask)) if win_actions.size else float("nan"),
                        "low_sat_candidate_count": int(np.sum(low_sat_candidate_mask)),
                        "low_sat_candidate_frac": float(np.mean(low_sat_candidate_mask)) if win_actions.size else float("nan"),
                        "saturation_candidate_count": int(np.sum(sat_candidate_mask)),
                        "saturation_candidate_frac": float(np.mean(sat_candidate_mask)) if win_actions.size else float("nan"),
                        "saturation_gap_mean": float(np.mean(informative_gap)) if informative_gap.size else 0.0,
                        "saturation_gap_max": float(np.max(informative_gap)) if informative_gap.size else 0.0,
                        "obstacle_visible_frac": float(np.mean(win_visible)) if win_visible.size else float("nan"),
                    }
                )

    write_csv(csv_path, rows)
    if not rows:
        return
    info_df = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for label, subset in [("all", info_df)] + [(f"mu={mu:g}", info_df[np.isclose(info_df["mu"], mu)]) for mu in MU_VALUES]:
        summary_rows.append(
            {
                "group": label,
                "n": int(len(subset)),
                "mean_sigma": float(subset["sigma"].mean()),
                "mean_abs_error": float(subset["abs_error"].mean()),
                "mean_strong_brake_frac": float(subset["strong_brake_frac"].mean()),
                "mean_low_sat_candidate_frac": float(subset["low_sat_candidate_frac"].mean()),
                "mean_saturation_candidate_frac": float(subset["saturation_candidate_frac"].mean()),
                "mean_saturation_gap": float(subset["saturation_gap_mean"].mean()),
                "corr_sigma_abs_error": _safe_corr(subset, "sigma", "abs_error"),
                "spearman_sigma_abs_error": _safe_corr(subset, "sigma", "abs_error", method="spearman"),
                "corr_info_sigma": _safe_corr(subset, "saturation_candidate_frac", "sigma"),
                "spearman_info_sigma": _safe_corr(subset, "saturation_candidate_frac", "sigma", method="spearman"),
                "corr_gap_sigma": _safe_corr(subset, "saturation_gap_mean", "sigma"),
                "spearman_gap_sigma": _safe_corr(subset, "saturation_gap_mean", "sigma", method="spearman"),
                "corr_info_abs_error": _safe_corr(subset, "saturation_candidate_frac", "abs_error"),
                "spearman_info_abs_error": _safe_corr(subset, "saturation_candidate_frac", "abs_error", method="spearman"),
            }
        )
    write_csv(summary_path, summary_rows)

    eps = 1e-5

    def _binned_stats(subset: Any, x_col: str, y_col: str, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xs: list[float] = []
        med: list[float] = []
        q25: list[float] = []
        q75: list[float] = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi == edges[-1]:
                mask = (subset[x_col] >= lo) & (subset[x_col] <= hi)
            else:
                mask = (subset[x_col] >= lo) & (subset[x_col] < hi)
            vals = subset.loc[mask, y_col].astype(float).to_numpy()
            if vals.size < 3:
                continue
            xs.append(float(0.5 * (lo + hi)))
            med.append(float(np.median(vals)))
            q25.append(float(np.quantile(vals, 0.25)))
            q75.append(float(np.quantile(vals, 0.75)))
        return np.asarray(xs), np.asarray(med), np.asarray(q25), np.asarray(q75)

    plot_df = info_df.copy()
    plot_df["sigma_eps"] = plot_df["sigma"].astype(float) + eps
    plot_df["abs_error_eps"] = plot_df["abs_error"].astype(float) + eps
    sigma_vals = plot_df["sigma_eps"].astype(float).to_numpy()
    sigma_edges = np.logspace(
        math.log10(max(eps, float(np.nanmin(sigma_vals)))),
        math.log10(max(eps * 10.0, float(np.nanmax(sigma_vals)))),
        9,
    )
    info_edges = np.linspace(0.0, 1.0, 9)

    plot_groups: list[tuple[str, Any]] = [("all", plot_df)] + [
        (f"mu={mu:g}", plot_df[np.isclose(plot_df["mu"], mu)].copy()) for mu in MU_VALUES
    ]
    fig, axes = plt.subplots(2, len(plot_groups), figsize=(5.0 * len(plot_groups), 8), sharey="row")
    colors = {0.25: "tab:blue", 0.5: "tab:orange", 0.7: "tab:green"}
    for col, (group_label, subset) in enumerate(plot_groups):
        color = "tab:purple" if group_label == "all" else colors.get(float(subset["mu"].iloc[0]), "tab:blue")

        ax = axes[0, col]
        if group_label == "all":
            for mu in MU_VALUES:
                mu_subset = subset[np.isclose(subset["mu"], mu)]
                ax.scatter(
                    mu_subset["saturation_candidate_frac"],
                    mu_subset["sigma_eps"],
                    s=9,
                    alpha=0.13,
                    color=colors.get(mu, None),
                    edgecolors="none",
                    label=f"mu={mu:g}",
                )
        else:
            ax.scatter(
                subset["saturation_candidate_frac"],
                subset["sigma_eps"],
                s=10,
                alpha=0.18,
                color=color,
                edgecolors="none",
            )
        xs, med, q25, q75 = _binned_stats(subset, "saturation_candidate_frac", "sigma_eps", info_edges)
        if xs.size:
            ax.plot(xs, med, color="black", linewidth=2, marker="o", markersize=4, label="bin median")
            ax.fill_between(xs, q25, q75, color="black", alpha=0.16, label="bin IQR")
        ax.set_title(f"{group_label}: information -> sigma")
        ax.set_xlabel("fraction saturation-candidate actions")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("predicted sigma + eps")
            ax.legend(fontsize=8)

        ax = axes[1, col]
        if group_label == "all":
            for mu in MU_VALUES:
                mu_subset = subset[np.isclose(subset["mu"], mu)]
                ax.scatter(
                    mu_subset["sigma_eps"],
                    mu_subset["abs_error_eps"],
                    s=9,
                    alpha=0.13,
                    color=colors.get(mu, None),
                    edgecolors="none",
                )
        else:
            ax.scatter(
                subset["sigma_eps"],
                subset["abs_error_eps"],
                s=10,
                alpha=0.18,
                color=color,
                edgecolors="none",
            )
        xs, med, q25, q75 = _binned_stats(subset, "sigma_eps", "abs_error_eps", sigma_edges)
        if xs.size:
            ax.plot(xs, med, color="black", linewidth=2, marker="o", markersize=4)
            ax.fill_between(xs, q25, q75, color="black", alpha=0.16)
        ax.set_title(f"{group_label}: sigma -> mean error")
        ax.set_xlabel("predicted sigma + eps")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("|mu_hat - mu| + eps")

    fig.suptitle("Trace windows: binned information, uncertainty, and error")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def _plot_final_behavior_by_mu(rows: list[dict[str, Any]], path: Path, *, title: str) -> None:
    if not rows:
        return
    methods = list(dict.fromkeys(str(r["model"]) for r in rows))
    mus = sorted({float(r["mu"]) for r in rows})
    metrics = [
        ("pre_mean_v", "pre mean v"),
        ("pre_mean_abs_u", "pre mean |u|"),
        ("pre_mean_brake", "pre mean brake"),
        ("pre_strong_brake_frac", "pre strong-brake frac"),
        ("post_mean_v", "post mean v"),
        ("post_mean_abs_u", "post mean |u|"),
        ("post_mean_brake", "post mean brake"),
        ("post_strong_brake_frac", "post strong-brake frac"),
    ]
    x = np.arange(len(methods), dtype=np.float64)
    width = 0.8 / max(len(mus), 1)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes_flat = axes.reshape(-1)
    for ax, (metric, ylabel) in zip(axes_flat, metrics):
        for idx, mu in enumerate(mus):
            vals = []
            for method in methods:
                row = next(
                    (
                        r
                        for r in rows
                        if str(r["model"]) == method and abs(float(r["mu"]) - mu) < 1e-8
                    ),
                    None,
                )
                vals.append(float(row.get(metric, float("nan"))) if row is not None else float("nan"))
            ax.bar(x + (idx - 0.5 * (len(mus) - 1)) * width, vals, width=width, label=f"mu={mu:g}")
        ax.set_title(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes_flat[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_probe_fraction_by_context(
    mean_only_eval_dir: Path,
    mean_std_eval_dir: Path,
    output_dir: Path,
) -> None:
    import pandas as pd

    mean_only_trace = mean_only_eval_dir / "trajectory_trace.csv"
    mean_std_trace = mean_std_eval_dir / "trajectory_trace.csv"
    if not mean_only_trace.exists() or not mean_std_trace.exists():
        return

    env_cfg = env_config()
    run_dir = mean_std_eval_dir.parents[1]
    method_prefix = mean_std_eval_dir.name.rsplit("_", 1)[0]
    cfg_candidates = [run_dir / method_prefix / "config.yaml", mean_std_eval_dir / "config.yaml"]
    cfg_candidates.extend(
        run_dir / f"gradual_nll_mean_std_v{version}" / "config.yaml"
        for version in [9, 8, 7, 6, 5, 4, 3, 2]
    )
    seen_cfgs: set[Path] = set()
    for cfg_path in cfg_candidates:
        if cfg_path in seen_cfgs:
            continue
        seen_cfgs.add(cfg_path)
        if cfg_path.exists():
            try:
                with cfg_path.open("r") as f:
                    loaded_cfg = yaml.safe_load(f)
                env_cfg = loaded_cfg.get("stopping_car_env", {}) or env_cfg
                break
            except Exception:
                pass

    mu_bins = [0.0, 0.375, 0.625, 1.0]
    mu_labels = ["low hat", "mid hat", "high hat"]
    sigma_bins = [-1e-12, 0.005, 0.05, np.inf]
    sigma_labels = ["low sigma\n<0.005", "mid sigma\n0.005-0.05", "high sigma\n>=0.05"]
    low_sat_threshold, _high_sat_threshold = low_friction_saturation_thresholds(env_cfg)
    metrics = [
        ("sat_candidate_frac", "sat-candidate", None),
        ("low_sat_frac", f"u < {low_sat_threshold:.3f}", low_sat_threshold),
        ("strong_brake_frac", "u < -0.5", -0.5),
    ]

    rows: list[dict[str, Any]] = []
    for method, trace_path in [("mean-only", mean_only_trace), ("mean+std", mean_std_trace)]:
        df = pd.read_csv(trace_path)
        if df.empty or not {"b", "mu_hat", "clipped_action"}.issubset(df.columns):
            continue
        df = df[df["b"] < 0.5].copy()
        df["mu_hat_bin"] = pd.cut(df["mu_hat"], bins=mu_bins, labels=mu_labels, include_lowest=True)
        if "sigma" in df.columns:
            df["sigma_bin"] = pd.cut(df["sigma"], bins=sigma_bins, labels=sigma_labels, include_lowest=True)
        else:
            df["sigma_bin"] = "no sigma"
        for (mu_hat_bin, sigma_bin), group in df.groupby(["mu_hat_bin", "sigma_bin"], observed=False):
            if len(group) == 0:
                continue
            row: dict[str, Any] = {
                "method": method,
                "mu_hat_bin": str(mu_hat_bin),
                "sigma_bin": str(sigma_bin),
                "n": int(len(group)),
                "mean_mu_hat": float(group["mu_hat"].mean()),
                "mean_abs_u": float(group["clipped_action"].abs().mean()),
                "mean_brake": float((-group["clipped_action"]).clip(lower=0).mean()),
            }
            for metric, _label, threshold in metrics:
                if metric == "sat_candidate_frac":
                    row[metric] = float(np.mean(saturation_candidate_mask(group["clipped_action"].to_numpy(dtype=float), env_cfg)))
                else:
                    row[metric] = float((group["clipped_action"] < float(threshold)).mean())
            if "sigma" in group.columns:
                row["mean_sigma"] = float(group["sigma"].mean())
            rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "probe_fraction_by_mu_hat_sigma.csv"
    plot_path = output_dir / "probe_fraction_by_mu_hat_sigma.png"
    write_csv(csv_path, rows)
    if not rows:
        return
    result = pd.DataFrame(rows)

    fig, axes = plt.subplots(len(metrics), 2, figsize=(13, 4 * len(metrics)), constrained_layout=True)
    axes = np.asarray(axes).reshape(len(metrics), 2)
    fig.suptitle("Pre-obstacle probing vs predicted friction mean and uncertainty", fontsize=16)
    for row_idx, (metric, metric_label, _threshold) in enumerate(metrics):
        ax = axes[row_idx, 0]
        mean_only = (
            result[result["method"] == "mean-only"]
            .set_index("mu_hat_bin")
            .reindex(mu_labels)
        )
        vals = mean_only[metric].fillna(0.0).to_numpy(dtype=float)
        ns = mean_only["n"].fillna(0).astype(int).to_numpy()
        bars = ax.bar(mu_labels, vals, color="#4c78a8")
        ax.set_ylim(0.0, max(0.4, float(np.nanmax(vals)) * 1.35 if vals.size else 0.4))
        ax.set_ylabel(f"{metric_label} fraction")
        ax.set_title(f"Mean-only: probe fraction by mu_hat ({metric_label})")
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"n={n}", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.25)

        ax = axes[row_idx, 1]
        mean_std = result[result["method"] == "mean+std"]
        mat = (
            mean_std.pivot(index="mu_hat_bin", columns="sigma_bin", values=metric)
            .reindex(index=mu_labels, columns=sigma_labels)
        )
        nmat = (
            mean_std.pivot(index="mu_hat_bin", columns="sigma_bin", values="n")
            .reindex(index=mu_labels, columns=sigma_labels)
        )
        mat_values = mat.to_numpy(dtype=float)
        vmax = max(0.4, float(np.nanmax(mat_values)) if np.isfinite(mat_values).any() else 0.4)
        im = ax.imshow(mat_values, vmin=0.0, vmax=vmax, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(sigma_labels)), sigma_labels)
        ax.set_yticks(range(len(mu_labels)), mu_labels)
        ax.set_title(f"Mean+std: probe fraction by mu_hat and sigma ({metric_label})")
        for i in range(len(mu_labels)):
            for j in range(len(sigma_labels)):
                val = mat.iloc[i, j]
                n = nmat.iloc[i, j]
                if pd.notna(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}\nn={int(n)}",
                        ha="center",
                        va="center",
                        color="white" if float(val) > 0.18 else "black",
                        fontsize=9,
                    )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[-1, :]:
        ax.set_xlabel("predicted friction / uncertainty bin")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def write_final_comparison_diagnostics(
    run_dir: Path,
    *,
    model_evaluations: dict[str, Path],
    mean_only_eval_dir: Path,
    mean_std_eval_dir: Path,
    handcoded_controllers: tuple[str, ...] = ("true_mu_safe", "nominal_mu", "always_conservative"),
) -> None:
    write_named_comparison_diagnostics(
        run_dir,
        model_evaluations=model_evaluations,
        output_prefix="final",
        title_prefix="Friction stopping final",
        mean_only_eval_dir=mean_only_eval_dir,
        mean_std_eval_dir=mean_std_eval_dir,
        handcoded_controllers=handcoded_controllers,
    )


def write_named_comparison_diagnostics(
    run_dir: Path,
    *,
    model_evaluations: dict[str, Path],
    output_prefix: str,
    title_prefix: str,
    mean_only_eval_dir: Path | None = None,
    mean_std_eval_dir: Path | None = None,
    handcoded_controllers: tuple[str, ...] = ("true_mu_safe", "nominal_mu", "always_conservative"),
) -> None:
    evaluations_dir = run_dir / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)

    score_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    for label, eval_dir in model_evaluations.items():
        score_path = eval_dir / "scorecard.csv"
        if score_path.exists():
            rows = _read_csv(score_path)
            if rows:
                row = dict(rows[0])
                row["label"] = label
                row["model"] = label
                score_rows.append(row)
        behavior_path = eval_dir / "behavior_by_mu.csv"
        if behavior_path.exists():
            for row in _read_csv(behavior_path):
                copied = dict(row)
                copied["model"] = label
                behavior_rows.append(copied)

    handcoded_score = run_dir / "handcoded" / "handcoded_scorecard.csv"
    if handcoded_score.exists():
        for row in _read_csv(handcoded_score):
            controller = str(row.get("controller", ""))
            if controller in handcoded_controllers:
                copied = dict(row)
                copied["label"] = controller
                copied["model"] = controller
                score_rows.append(copied)

    handcoded_traces = run_dir / "handcoded" / "handcoded_traces.csv"
    if handcoded_traces.exists():
        selected_trace_rows = [
            row for row in _read_csv(handcoded_traces) if str(row.get("controller", "")) in handcoded_controllers
        ]
        behavior_rows.extend(aggregate_behavior_by_mu(selected_trace_rows))

    score_csv = evaluations_dir / f"{output_prefix}_method_scorecard.csv"
    behavior_csv = evaluations_dir / f"{output_prefix}_behavior_by_mu.csv"
    write_csv(score_csv, score_rows)
    write_csv(behavior_csv, behavior_rows)
    _plot_scorecard(score_csv, evaluations_dir / f"{output_prefix}_method_scorecard.png", title=f"{title_prefix} method comparison")
    _plot_final_behavior_by_mu(
        behavior_rows,
        evaluations_dir / f"{output_prefix}_behavior_by_mu.png",
        title=f"{title_prefix} behavior by friction",
    )
    if mean_only_eval_dir is not None and mean_std_eval_dir is not None:
        write_probe_fraction_by_context(mean_only_eval_dir, mean_std_eval_dir, evaluations_dir)
