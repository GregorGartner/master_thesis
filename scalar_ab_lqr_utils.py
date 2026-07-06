from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import yaml
from scipy.linalg import solve_discrete_are
from stable_baselines3.common.vec_env import DummyVecEnv

from lqr_model import LinearQuadraticEnv, install_lqr_adapter_for_domain_randomization
from run_two_action_pipeline import CONFIG_PATH, ROOT, _resolve_exp_root, _run_with_config
from unified_context_ppo import UnifiedContextPPO
from wrappers import DomainRandomizationWrapper


SELECTION_ROOT = ROOT / "experiments" / "scalar_ab_lqr"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def _env_range(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return list(default)
    values = [float(part.strip()) for part in raw.split(",")]
    if len(values) != 2 or values[0] >= values[1]:
        raise ValueError(f"{name} must be two increasing comma-separated floats, got {raw!r}.")
    return values


def _default_range_from_env() -> dict[str, list[float]]:
    presets = {
        "stable_wide_b": {"a": [0.75, 1.05], "b": [0.40, 1.80]},
        "medium": {"a": [0.85, 1.10], "b": [0.60, 1.40]},
        "aggressive": {"a": [0.80, 1.20], "b": [0.40, 1.80]},
    }
    preset = os.environ.get("SCALAR_AB_RANGE_PRESET", "stable_wide_b")
    if preset not in presets:
        raise ValueError(f"Unknown SCALAR_AB_RANGE_PRESET={preset!r}; choose one of {sorted(presets)}.")
    selected = presets[preset]
    return {
        "a": _env_range("SCALAR_AB_A_RANGE", selected["a"]),
        "b": _env_range("SCALAR_AB_B_RANGE", selected["b"]),
    }


DEFAULT_RANGE = _default_range_from_env()
FALLBACK_RANGE = {"a": [0.85, 1.10], "b": [0.60, 1.40]}
Q = 1.0
R = 0.3
PROCESS_NOISE_STD = _env_float("SCALAR_AB_PROCESS_NOISE_STD", 0.03)
INITIAL_STATE_LOW = _env_float("SCALAR_AB_INITIAL_STATE_LOW", -0.5)
INITIAL_STATE_HIGH = _env_float("SCALAR_AB_INITIAL_STATE_HIGH", 0.5)
ACTION_BOUND = _env_float("SCALAR_AB_ACTION_BOUND", 20.0)
EPISODE_STEPS = _env_int("SCALAR_AB_EPISODE_STEPS", 128)
WINDOW_LENGTH = _env_int("SCALAR_AB_WINDOW_LENGTH", 32)
WARMUP_STEPS = _env_int("SCALAR_AB_WARMUP_STEPS", WINDOW_LENGTH - 1)
UPDATE_INTERVAL = _env_int("SCALAR_AB_UPDATE_INTERVAL", max(1, WINDOW_LENGTH // 4))
Z_SCALE = 1.0
PRIOR_MEAN = [0.9, 1.1]
PRIOR_VAR = [0.15**2, 0.7**2]


def smoke_enabled(env_name: str) -> bool:
    return os.environ.get(env_name, "0").lower() in {"1", "true", "yes"}


def timestamp() -> str:
    return datetime.now().strftime("%m-%d__%H-%M")


def scalar_gain(a: float, b: float, q: float = Q, r: float = R) -> float:
    a_arr = np.asarray([[a]], dtype=np.float64)
    b_arr = np.asarray([[b]], dtype=np.float64)
    p = solve_discrete_are(a_arr, b_arr, np.asarray([[q]], dtype=np.float64), np.asarray([[r]], dtype=np.float64))
    k = np.linalg.solve(np.asarray([[r]], dtype=np.float64) + b_arr.T @ p @ b_arr, b_arr.T @ p @ a_arr)
    return float(k[0, 0])


def closed_loop_signature(a: float, b: float, k: float) -> float:
    return float(a - b * k)


def discounted_cost_for_gain(
    a: float,
    b: float,
    k: float,
    *,
    q: float = Q,
    r: float = R,
    gamma: float = 1.0,
) -> float:
    c = a - b * k
    stage = q + r * k * k
    denom = max(1.0 - gamma * c * c, 1e-12)
    return float(stage / denom)


def scalar_env_config(param_range: dict[str, list[float]] | None = None) -> dict[str, Any]:
    param_range = param_range or DEFAULT_RANGE
    return {
        "environment": "lqr",
        "lqr_env": {
            "A": [[float(np.mean(param_range["a"]))]],
            "B": [[float(np.mean(param_range["b"]))]],
            "delta_B": [[0.0]],
            "Q": [[Q]],
            "R": [[R]],
            "process_noise_std": PROCESS_NOISE_STD,
            "initial_state_low": INITIAL_STATE_LOW,
            "initial_state_high": INITIAL_STATE_HIGH,
            "action_low": -ACTION_BOUND,
            "action_high": ACTION_BOUND,
            "max_episode_steps": EPISODE_STEPS,
            "state_termination_bound": None,
            "reward_cost_mode": "log",
            "action_cost_type": "quadratic",
        },
        "wrappers": [
            {
                "name": "DomainRandomizationWrapper",
                "enabled": True,
                "params": {
                    "change_prob": 1.0,
                    "only_at_episode_end": True,
                    "randomize_on_reset": True,
                    "randomize_theta": False,
                    "randomize_a": True,
                    "randomize_b": True,
                    "a_range": list(param_range["a"]),
                    "b_range": list(param_range["b"]),
                    "randomize_process_noise_scale": False,
                },
            }
        ],
    }


def set_scalar_common_model_params(cfg: dict[str, Any], seed: int = 1) -> None:
    params = cfg["model"]["params"]
    params["policy"] = "MlpPolicy"
    params["seed"] = int(seed)
    params["n_steps"] = 4096
    params["batch_size"] = 512
    params["n_epochs"] = 8
    params["gamma"] = 0.995
    params["gae_lambda"] = 0.95
    params["clip_range"] = 0.2
    params["normalize_advantage"] = True
    params["vf_coef"] = 0.5
    params["max_grad_norm"] = 1.0
    params["target_kl"] = None
    params["verbose"] = 1
    params["device"] = "cpu"
    params["actor_net_arch"] = [64, 64]
    params["critic_net_arch"] = [64, 64]
    params["encoder_net_arch"] = [64, 64]
    params["freeze_ppo"] = False
    params["deterministic_actions"] = False
    params["naive_action_noise_std"] = 0.0
    params["naive_action_noise_dist"] = "gaussian"
    params["regression_param_names"] = ["a", "b"]
    params["latent_dim"] = 2
    params["window_length"] = WINDOW_LENGTH
    params["id_update_interval"] = UPDATE_INTERVAL
    params["nominal_warmup_steps"] = WARMUP_STEPS
    params["z_scale"] = Z_SCALE
    params["use_transition_features"] = True
    params["transition_type"] = "delta"
    params["encoder_type"] = "mlp"
    params["detach_context_for_rl"] = True
    params["uncertainty_regularization_coef"] = 0.0
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["uncertainty_penalty_metric"] = "std"
    for cb in cfg.get("callbacks", []):
        if cb.get("name") == "LivePlotCallback":
            cb["enabled"] = True
            cb.setdefault("params", {})["update_every_episodes"] = 500
        if cb.get("name") == "SaveModelCallback":
            cb["enabled"] = True
            cb.setdefault("params", {})["save_on_training_end"] = True
            cb["params"]["best_metric"] = "episode_reward"


def base_training_cfg(
    base_cfg: dict[str, Any],
    exp_root: Path,
    exp_name: str,
    steps: int,
    param_range: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
    cfg.update(scalar_env_config(param_range))
    cfg["total_timesteps"] = int(steps)
    cfg["training"]["experiment_root"] = str(exp_root)
    cfg["training"]["experiment_name"] = exp_name
    cfg["training"]["experiment_name_suffix"] = None
    cfg["training"]["load_weights"] = False
    cfg["training"]["load_weights_from"] = None
    cfg["training"]["load_weights_name"] = "weights"
    cfg["training"]["load_encoder_only"] = False
    cfg["model"]["name"] = "UnifiedContextPPO"
    set_scalar_common_model_params(cfg)
    return cfg


def set_load(cfg: dict[str, Any], source: Path | None, weights_name: str = "weights") -> None:
    cfg["training"]["load_weights"] = source is not None
    cfg["training"]["load_weights_from"] = str(source.resolve()) if source is not None else None
    cfg["training"]["load_weights_name"] = weights_name
    cfg["training"]["load_encoder_only"] = False


def snapshot_phase_weights(exp_dir: Path, phase_name: str) -> None:
    import shutil

    for name in ["weights", "weights_best"]:
        src = exp_dir / f"{name}.zip"
        if src.exists():
            shutil.copy2(src, exp_dir / f"{name}_{phase_name}.zip")
    metric = exp_dir / "weights_best.metric"
    if metric.exists():
        shutil.copy2(metric, exp_dir / f"weights_best_{phase_name}.metric")


def run_training_stage(label: str, cfg: dict[str, Any]) -> None:
    print(f"START {label}: {cfg['training']['experiment_name']} steps={cfg['total_timesteps']}", flush=True)
    _run_with_config(cfg)
    print(f"END   {label}: {cfg['training']['experiment_name']}", flush=True)


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
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def make_eval_env(exp_cfg: dict[str, Any], a: float, b: float, seed: int) -> LinearQuadraticEnv:
    env_cfg = dict(exp_cfg.get("lqr_env", {}) or {})
    env_cfg["A"] = [[a]]
    env_cfg["B"] = [[b]]
    env_cfg["delta_B"] = [[0.0]]
    env_cfg.setdefault("max_episode_steps", EPISODE_STEPS)
    env = LinearQuadraticEnv(**env_cfg)
    env.set_scalar_ab_params(a, b, process_noise_scale=1.0)
    env.reset(seed=seed)
    return env


def _dummy_vec_for_model(exp_cfg: dict[str, Any]) -> DummyVecEnv:
    install_lqr_adapter_for_domain_randomization()

    def make_env():
        env_cfg = dict(exp_cfg.get("lqr_env", {}) or {})
        env = LinearQuadraticEnv(**env_cfg)
        for wr_spec in exp_cfg.get("wrappers", []):
            if wr_spec.get("enabled", True) is False:
                continue
            if wr_spec["name"] == "DomainRandomizationWrapper":
                env = DomainRandomizationWrapper(env, **(wr_spec.get("params", {}) or {}))
        return env

    return DummyVecEnv([make_env])


def load_ppo(exp_dir: Path, weights_name: str = "weights") -> tuple[UnifiedContextPPO, dict[str, Any]]:
    with open(exp_dir / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    env = _dummy_vec_for_model(cfg)
    model = UnifiedContextPPO.load(str(exp_dir / weights_name), env=env)
    model.set_env(env)
    return model, cfg


def evaluate_controller_grid(
    output_dir: Path,
    specs: list[dict[str, Any]],
    *,
    param_range: dict[str, list[float]] | None = None,
    grid_n: int = 31,
    seeds_per_pair: int = 3,
    base_seed: int = 12345,
) -> Path:
    param_range = param_range or DEFAULT_RANGE
    output_dir.mkdir(parents=True, exist_ok=True)
    a_values = np.linspace(param_range["a"][0], param_range["a"][1], grid_n)
    b_values = np.linspace(param_range["b"][0], param_range["b"][1], grid_n)
    runtimes: dict[str, tuple[UnifiedContextPPO, dict[str, Any]] | None] = {}
    for spec in specs:
        if spec["kind"] == "ppo":
            runtimes[spec["label"]] = load_ppo(Path(spec["experiment"]), spec.get("weights_name", "weights"))
        else:
            runtimes[spec["label"]] = None

    rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for label_runtime, runtime in runtimes.items():
        spec = next(s for s in specs if s["label"] == label_runtime)
        for a in a_values:
            for b in b_values:
                for seed_idx in range(seeds_per_pair):
                    seed = base_seed + seed_idx
                    if spec["kind"] == "oracle_lqr":
                        cfg = scalar_env_config(param_range)
                        env = make_eval_env(cfg, float(a), float(b), seed)
                        ep = _rollout_lqr(env, assumed_a=float(a), assumed_b=float(b))
                    elif spec["kind"] == "ppo":
                        assert runtime is not None
                        model, cfg = runtime
                        env = make_eval_env(cfg, float(a), float(b), seed)
                        ep, trace = _rollout_ppo(env, model, collect_trace=True)
                        for tr in trace:
                            tr.update({"controller": spec["label"], "a": float(a), "b": float(b), "seed": seed})
                        step_rows.extend(trace)
                    else:
                        raise ValueError(f"Unsupported controller kind: {spec['kind']}")
                    ep.update({"controller": spec["label"], "a": float(a), "b": float(b), "seed": seed})
                    rows.append(ep)
    write_csv(output_dir / "episode_summary.csv", rows)
    write_csv(output_dir / "trajectory_trace.csv", step_rows)
    aggregate = _aggregate_grid(rows)
    write_csv(output_dir / "scalar_ab_grid_aggregate.csv", aggregate)
    _plot_return_heatmaps(aggregate, output_dir)
    _plot_info_heatmaps(aggregate, output_dir)
    return output_dir


def _rollout_lqr(env: LinearQuadraticEnv, assumed_a: float, assumed_b: float) -> dict[str, Any]:
    obs, _ = env.reset()
    k = scalar_gain(assumed_a, assumed_b)
    total_reward = 0.0
    total_cost = 0.0
    total_state_cost = 0.0
    total_action_cost = 0.0
    z_rows: list[list[float]] = []
    done = False
    while not done:
        x = float(obs[0])
        action = np.asarray([-k * x], dtype=np.float32)
        z_rows.append([x, float(action[0])])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_cost += float(info.get("true_cost", -reward))
        total_state_cost += float(info.get("state_cost", 0.0))
        total_action_cost += float(info.get("quadratic_action_cost", info.get("action_cost", 0.0)))
        done = bool(terminated or truncated)
    return _episode_metrics(total_reward, total_cost, total_state_cost, total_action_cost, z_rows)


def _rollout_ppo(
    env: LinearQuadraticEnv,
    model: UnifiedContextPPO,
    *,
    collect_trace: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.set_env(DummyVecEnv([lambda: env]))
    obs, _ = env.reset()
    state = None
    episode_start = np.asarray([True])
    total_reward = 0.0
    total_cost = 0.0
    total_state_cost = 0.0
    total_action_cost = 0.0
    z_rows: list[list[float]] = []
    trace: list[dict[str, Any]] = []
    done = False
    step = 0
    while not done:
        action, state = model.predict(obs.reshape(1, -1), state=state, episode_start=episode_start, deterministic=True)
        episode_start = np.asarray([False])
        x = float(obs[0])
        u = float(action.reshape(-1)[0])
        z_rows.append([x, u])
        obs, reward, terminated, truncated, info = env.step(action.reshape(-1))
        total_reward += float(reward)
        total_cost += float(info.get("true_cost", -reward))
        total_state_cost += float(info.get("state_cost", 0.0))
        total_action_cost += float(info.get("quadratic_action_cost", info.get("action_cost", 0.0)))
        if collect_trace:
            trace.append({"step": step, "x": x, "u": u, "reward": float(reward)})
        done = bool(terminated or truncated)
        step += 1
    return _episode_metrics(total_reward, total_cost, total_state_cost, total_action_cost, z_rows), trace


def _episode_metrics(
    total_reward: float,
    total_cost: float,
    total_state_cost: float,
    total_action_cost: float,
    z_rows: list[list[float]],
) -> dict[str, Any]:
    z = np.asarray(z_rows, dtype=np.float64)
    gram = z.T @ z if z.size else np.zeros((2, 2), dtype=np.float64)
    eigs = np.linalg.eigvalsh(gram + 1e-12 * np.eye(2))
    ratio = z[:, 1] / np.maximum(np.abs(z[:, 0]), 1e-8) if z.size else np.asarray([])
    return {
        "return": float(total_reward),
        "true_cost": float(total_cost),
        "state_cost": float(total_state_cost),
        "action_cost": float(total_action_cost),
        "gram_lambda_min": float(eigs[0]),
        "gram_lambda_max": float(eigs[-1]),
        "gram_logdet": float(np.linalg.slogdet(gram + 1e-8 * np.eye(2))[1]),
        "gram_condition": float(eigs[-1] / max(eigs[0], 1e-12)),
        "u_over_x_var": float(np.var(ratio)) if ratio.size else 0.0,
    }


def _aggregate_grid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault((str(row["controller"]), float(row["a"]), float(row["b"])), []).append(row)
    agg: list[dict[str, Any]] = []
    keys = ["return", "true_cost", "state_cost", "action_cost", "gram_lambda_min", "gram_logdet", "gram_condition", "u_over_x_var"]
    for (controller, a, b), bucket in buckets.items():
        out = {"controller": controller, "a": a, "b": b, "n": len(bucket)}
        for key in keys:
            vals = np.asarray([float(row[key]) for row in bucket], dtype=np.float64)
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals))
        agg.append(out)
    return agg


def _plot_return_heatmaps(rows: list[dict[str, Any]], output_dir: Path) -> None:
    controllers = sorted({row["controller"] for row in rows})
    for controller in controllers:
        _plot_heatmap(rows, output_dir / f"return_heatmap_{controller}.png", controller, "return_mean", "Mean Return")


def _plot_info_heatmaps(rows: list[dict[str, Any]], output_dir: Path) -> None:
    controllers = sorted({row["controller"] for row in rows})
    for controller in controllers:
        _plot_heatmap(rows, output_dir / f"gram_lambda_min_heatmap_{controller}.png", controller, "gram_lambda_min_mean", "Mean Gram lambda_min")


def _plot_heatmap(rows: list[dict[str, Any]], path: Path, controller: str, metric: str, title: str) -> None:
    subset = [row for row in rows if row["controller"] == controller]
    if not subset:
        return
    a_vals = sorted({float(row["a"]) for row in subset})
    b_vals = sorted({float(row["b"]) for row in subset})
    grid = np.full((len(a_vals), len(b_vals)), np.nan, dtype=np.float64)
    a_idx = {v: i for i, v in enumerate(a_vals)}
    b_idx = {v: i for i, v in enumerate(b_vals)}
    for row in subset:
        grid[a_idx[float(row["a"])], b_idx[float(row["b"])]] = float(row[metric])
    plt.figure(figsize=(8, 6))
    plt.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[min(b_vals), max(b_vals), min(a_vals), max(a_vals)],
    )
    plt.colorbar(label=metric)
    plt.xlabel("b")
    plt.ylabel("a")
    plt.title(f"{title}: {controller}")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
