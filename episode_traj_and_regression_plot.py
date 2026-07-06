import os
from typing import Any, Mapping, Optional, Tuple
import gymnasium as gym
import numpy as np
import yaml
import shimmy
# import shimmy.dm_control_compatibility  # noqa: F401
import torch as th
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
# from context_ppo import ContextPPO
from unified_context_ppo import UnifiedContextPPO
from wrappers import RewardWrapper, DomainRandomizationWrapper, ForceMujocoFixedCamera # , ParamObsWrapper, SwitchPenalty, HideVelocityObs
# from helper_functions import choose_cartpole_params
from lqr_model import LinearQuadraticEnv, install_lqr_adapter_for_domain_randomization


# You can override these via CLI flags, or by setting CTX_PPO_EXPERIMENT.
EXPERIMENT_DIR = "./experiments/s_04-23__10-04_nll_50_async_10_exp_1_0"
EPISODE_SEEDS = range(12345, 12356)
EPISODE_SEED = next(iter(EPISODE_SEEDS))
DETERMINISTIC_ACTIONS = True
FORCE_DETERMINISTIC_DYNAMICS = True # (domain randomization)
CONTROLLER_MODE = "both"  # "ppo" or "lqr" or "both"


WRAPPER_REGISTRY = {
    "RewardWrapper": RewardWrapper,
    "DomainRandomizationWrapper": DomainRandomizationWrapper,
    "ChangingCartPoleDynamics": DomainRandomizationWrapper,
    # "ParamObsWrapper": ParamObsWrapper,
    # "HideVelocityObs": HideVelocityObs,xz
    # "SwitchPenalty": SwitchPenalty,
}

MODEL_REGISTRY = {
    "PPO": PPO,
    # "ContextPPO": ContextPPO,
    "UnifiedContextPPO": UnifiedContextPPO,
}


def _unwrap_until_attr(env: gym.Env, attr: str) -> Any:
    e = env
    while True:
        if hasattr(e, attr):
            return getattr(e, attr)
        if not hasattr(e, "env"):
            break
        e = e.env
    raise AttributeError(f"Could not find attribute {attr!r} in env wrapper stack")


def _extract_state(obs: Any, environment: str) -> Optional[np.ndarray]:
    """Return a flat state vector for plotting across cartpole and lqr."""

    if environment == "cartpole" and isinstance(obs, Mapping):
        # dm_control cartpole dict obs.
        if "position" in obs and "velocity" in obs:
            pos = np.asarray(obs["position"], dtype=np.float32).ravel()
            vel = np.asarray(obs["velocity"], dtype=np.float32).ravel()
            if pos.size >= 2 and vel.size >= 2:
                x = float(pos[0])
                theta = float(pos[1])
                x_dot = float(vel[0])
                theta_dot = float(vel[1])
                return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

    if isinstance(obs, Mapping):
        # Generic dict fallback: flatten all fields in insertion order.
        parts = []
        for key in obs.keys():
            arr = np.asarray(obs[key], dtype=np.float32).ravel()
            if arr.size > 0:
                parts.append(arr)
        if len(parts) > 0:
            return np.concatenate(parts, axis=0).astype(np.float32)
        return None

    # Non-dict obs (e.g., LQR Box obs).
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.ravel()


def _state_labels(environment: str, state_dim: int) -> list[str]:
    if environment == "cartpole" and state_dim == 4:
        return ["position (x)", "velocity (x_dot)", "angle (theta)", "ang. vel. (theta_dot)"]
    if environment == "lqr":
        return [f"x[{i}]" for i in range(state_dim)]
    return [f"state[{i}]" for i in range(state_dim)]


def _get_true_params_like_training(model: UnifiedContextPPO, env: gym.Env) -> np.ndarray:
    """Match ContextPPO._get_true_params() logic, but for a non-VecEnv gym env."""

    get_true_params = _unwrap_until_attr(env, "get_true_params")
    get_param_denoms = _unwrap_until_attr(env, "get_param_denoms")

    raw = np.asarray(get_true_params(model.regression_param_names), dtype=np.float32)
    denoms = model._true_param_denoms
    if denoms is None:
        denoms = np.asarray(get_param_denoms(model.regression_param_names), dtype=np.float32)

    return raw / np.asarray(denoms, dtype=np.float32)


@th.no_grad()
def _policy_used_context_and_metrics(
    model: UnifiedContextPPO,
    state,
    true_params: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    cached_context = getattr(model, "_predict_cached_context", None)
    if cached_context is None or cached_context.shape[0] == 0:
        raise RuntimeError("Policy-used context is not available. Call model.predict(...) first.")

    context_vec = np.asarray(cached_context[0], dtype=np.float32).reshape(-1)
    latent_dim = int(getattr(model, "_latent_dim", len(context_vec)))
    pred_params = context_vec[:latent_dim].copy()

    pred_std: Optional[np.ndarray] = None
    obs_hist, act_hist = state
    traj_window_np = model._build_traj_window(obs_hist, act_hist)  # (n_envs, window, traj_dim)
    traj_window_th = th.as_tensor(traj_window_np, device=model.device, dtype=th.float32)
    mode = str(getattr(model, "context_mode", ""))
    if mode == "encoder_nll":
        _, logvar_th = model.policy.encode_context(traj_window_th, return_logvar=True)
        logvar_th = model.policy.clamp_logvar(logvar_th)
        pred_std = th.exp(0.5 * logvar_th).detach().cpu().numpy().reshape(-1)
    elif mode == "closed_form" and bool(getattr(model, "condition_on_uncertainty", False)) and context_vec.size >= 2 * latent_dim:
        pred_std = context_vec[latent_dim : 2 * latent_dim].copy()

    true_vec = np.asarray(true_params, dtype=np.float32).reshape(-1)
    mse = (pred_params - true_vec) ** 2
    return pred_params, mse, pred_std


def _make_env_from_cfg(cfg: dict, *, monitor_dir: Optional[str], render_mode: Optional[str]) -> gym.Env:
    environment = str(cfg.get("environment", "cartpole")).lower()
    if environment == "cartpole":
        env = gym.make("dm_control/cartpole-swingup-v0", render_mode=render_mode)

        max_episode_steps = cfg.get("max_episode_steps", None)
        if max_episode_steps is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=int(max_episode_steps))

        env = ForceMujocoFixedCamera(env, camera_id=0, width=1200, height=800)
    elif environment == "lqr":
        install_lqr_adapter_for_domain_randomization()
        lqr_cfg = dict(cfg.get("lqr_env", {}) or {})
        lqr_cfg.setdefault("max_episode_steps", int(cfg.get("max_episode_steps", 200)))
        env = LinearQuadraticEnv(**lqr_cfg)
    else:
        raise ValueError(f"Unknown environment={environment!r}. Expected 'cartpole' or 'lqr'.")

    wr_specs = cfg.get("wrappers", [])
    for wr_spec in wr_specs:
        if wr_spec.get("enabled", True) is False:
            continue
        wr_name = wr_spec["name"]
        wr_params = wr_spec.get("params", {}) or {}
        env = WRAPPER_REGISTRY[wr_name](env, **wr_params)

    return env


def _make_env_from_cfg_deterministic(
    cfg: dict,
    *,
    monitor_dir: Optional[str],
    render_mode: Optional[str],
    episode_seed: int,
) -> gym.Env:
    """Same as _make_env_from_cfg, but enforces deterministic dynamics randomization.

    This mirrors the override used in cartpole_ppo_sb3_test.py so that
    env.reset(seed=...) also deterministically seeds ChangingCartPoleDynamics.
    """

    environment = str(cfg.get("environment", "cartpole")).lower()
    if environment == "cartpole":
        env = gym.make("dm_control/cartpole-swingup-v0", render_mode=render_mode)

        max_episode_steps = cfg.get("max_episode_steps", None)
        if max_episode_steps is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=int(max_episode_steps))

        env = ForceMujocoFixedCamera(env, camera_id=0, width=1200, height=800)
    elif environment == "lqr":
        install_lqr_adapter_for_domain_randomization()
        lqr_cfg = dict(cfg.get("lqr_env", {}) or {})
        lqr_cfg.setdefault("max_episode_steps", int(cfg.get("max_episode_steps", 200)))
        env = LinearQuadraticEnv(**lqr_cfg)
    else:
        raise ValueError(f"Unknown environment={environment!r}. Expected 'cartpole' or 'lqr'.")

    wr_specs = cfg.get("wrappers", [])
    for wr_spec in wr_specs:
        if wr_spec.get("enabled", True) is False:
            continue

        wr_name = wr_spec["name"]
        wr_params = wr_spec.get("params", {}) or {}

        # Deterministic override of config: reseed the wrapper RNG per episode.
        if wr_name in {"DomainRandomizationWrapper", "ChangingCartPoleDynamics"}:
            wr_params = dict(wr_params)
            wr_params.setdefault("reseed_on_reset", True)
            wr_params.setdefault("rng_seed", int(episode_seed))

        env = WRAPPER_REGISTRY[wr_name](env, **wr_params)

    return env


def _run_episode_for_controller(
    cfg: dict,
    *,
    test_dir: str,
    render_mode: Optional[str],
    weights_path: str,
    environment: str,
    controller_mode: str,
) -> dict[str, Any]:
    if bool(FORCE_DETERMINISTIC_DYNAMICS):
        env = _make_env_from_cfg_deterministic(
            cfg,
            monitor_dir=test_dir,
            render_mode=render_mode,
            episode_seed=int(EPISODE_SEED),
        )
    else:
        env = _make_env_from_cfg(cfg, monitor_dir=test_dir, render_mode=render_mode)

    model = None
    is_context_ppo = False
    if controller_mode == "ppo":
        model_spec = cfg.get("model", None) or {}
        model_name = model_spec.get("name", None)
        if model_name not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model name {model_name!r}. Expected one of: {list(MODEL_REGISTRY.keys())}")
        model = MODEL_REGISTRY[model_name].load(weights_path, env=env)
        # if hasattr(model, "nominal_warmup_steps"):
        #     model.nominal_warmup_steps = 0
        is_context_ppo = isinstance(model, UnifiedContextPPO)
    elif controller_mode != "lqr":
        raise ValueError(f"Unknown controller_mode={controller_mode!r}. Expected 'ppo' or 'lqr'.")

    obs, _ = env.reset(seed=int(EPISODE_SEED))
    true_params_dict0 = dict(_unwrap_until_attr(env, "get_true_params_dict")())
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    if controller_mode == "lqr" and not hasattr(base_env, "lqr_action"):
        raise ValueError("LQR controller requested, but environment does not expose lqr_action().")

    if is_context_ppo:
        try:
            true0 = _get_true_params_like_training(model, env)  # type: ignore[arg-type]
            print(
                f"Controller=PPO | Episode seed={int(EPISODE_SEED)} | "
                f"true_params(t=0)={np.array2string(true0, precision=4)}"
            )
        except Exception:
            pass

    state = None
    episode_start = np.ones((1,), dtype=bool)

    states: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    mses: list[np.ndarray] = []
    uncertainties: list[np.ndarray] = []
    true_params_list: list[np.ndarray] = []
    rewards: list[float] = []
    actions: list[np.ndarray] = []

    terminated = False
    truncated = False
    max_episode_steps = int(getattr(base_env, "max_episode_steps", cfg.get("max_episode_steps", 1000)))
    for _t in range(max_episode_steps):
        if controller_mode == "lqr":
            action = base_env.lqr_action(obs)
        elif is_context_ppo and str(getattr(model, "context_mode", "")) in {"encoder_mle", "encoder_nll", "closed_form"}:
            action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=bool(DETERMINISTIC_ACTIONS))
            true_params = _get_true_params_like_training(model, env) * float(getattr(model, "z_scale", 1.0))
            _pred_params, mse, pred_std = _policy_used_context_and_metrics(model, state, true_params)  # type: ignore[arg-type]
            predictions.append(np.asarray(_pred_params, dtype=np.float32))
            mses.append(np.asarray(mse, dtype=np.float32))
            if pred_std is not None:
                uncertainties.append(np.asarray(pred_std, dtype=np.float32))
            true_params_list.append(np.asarray(true_params, dtype=np.float32))
        else:
            action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=bool(DETERMINISTIC_ACTIONS))

        s = _extract_state(obs, environment)
        if s is None:
            raise ValueError(f"Could not extract a plottable state from observation for environment={environment!r}")
        states.append(s)

        action_env = action
        if isinstance(action_env, np.ndarray) and action_env.ndim >= 1 and action_env.shape[0] == 1:
            action_env = action_env[0]
        actions.append(np.asarray(action_env, dtype=np.float32).ravel())

        obs, reward, terminated, truncated, _info = env.step(action_env)
        rewards.append(float(reward))

        episode_start[...] = bool(terminated or truncated)
        if bool(terminated or truncated):
            break

    env.close()

    state_arr = np.stack(states, axis=0)
    action_arr = np.stack(actions, axis=0)
    prediction_arr = np.stack(predictions, axis=0) if len(predictions) > 0 else None
    mse_arr = np.stack(mses, axis=0) if len(mses) > 0 else None
    uncertainty_arr = np.stack(uncertainties, axis=0) if len(uncertainties) > 0 else None
    param_names = list(model.regression_param_names) if is_context_ppo else []
    z_scale = float(getattr(model, "z_scale", 1.0)) if is_context_ppo else None

    return {
        "label": "PPO" if controller_mode == "ppo" else "LQR",
        "states": state_arr,
        "actions": action_arr,
        "rewards": np.asarray(rewards, dtype=np.float32),
        "predictions": prediction_arr,
        "mse": mse_arr,
        "uncertainty": uncertainty_arr,
        "param_names": param_names,
        "true_params": (np.stack(true_params_list, axis=0) if len(true_params_list) > 0 else None),
        "true0": (true_params_list[0] if len(true_params_list) > 0 else None),
        "true_params_dict0": true_params_dict0,
        "z_scale": z_scale,
    }


def main() -> None:
    experiment_dir = EXPERIMENT_DIR
    test_dir = os.path.join(experiment_dir, "test")
    config_path = os.path.join(experiment_dir, "config.yaml")
    weights_path = os.path.join(experiment_dir, "weights_best" if os.path.exists(os.path.join(experiment_dir, "weights_best.zip")) else "weights")

    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(f"Experiment folder not found: {experiment_dir}\n")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.yaml at: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    render_mode = cfg.get("render_mode", None)
    environment = str(cfg.get("environment", "cartpole")).lower()

    if CONTROLLER_MODE == "both":
        controller_modes = ["ppo", "lqr"]
    elif CONTROLLER_MODE in {"ppo", "lqr"}:
        controller_modes = [CONTROLLER_MODE]
    else:
        raise ValueError(f"Unknown CONTROLLER_MODE={CONTROLLER_MODE!r}. Expected 'ppo', 'lqr', or 'both'.")

    results = [
        _run_episode_for_controller(
            cfg,
            test_dir=test_dir,
            render_mode=render_mode,
            weights_path=weights_path,
            environment=environment,
            controller_mode=mode,
        )
        for mode in controller_modes
    ]
    true_theta0 = next((r["true_params_dict0"].get("theta") for r in results if r["true_params_dict0"].get("theta") is not None), None)
    print(f"Episode seed={EPISODE_SEED}" + (f" | true_theta={true_theta0:.4f}" if true_theta0 is not None else ""))

    state_dim = int(results[0]["states"].shape[1])
    action_dim = int(results[0]["actions"].shape[1])
    state_labels = _state_labels(environment, state_dim)
    show_predictions = any(r["predictions"] is not None for r in results)
    show_mse = any(r["mse"] is not None for r in results)
    show_uncertainty = any(r["uncertainty"] is not None for r in results)

    n_rows = state_dim + action_dim + 1 + (1 if show_predictions else 0) + (1 if show_mse else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2 * n_rows), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    row = 0
    for i in range(state_dim):
        ax = axes[row]
        for res in results:
            S = res["states"]
            ax.plot(np.arange(S.shape[0], dtype=np.int64), S[:, i], label=res["label"])
        ax.set_ylabel(state_labels[i])
        if i == 0 and len(results) > 1:
            ax.legend(loc="best")
        row += 1

    if show_predictions:
        pred_axis = axes[row]
        for res in results:
            pred_arr = res["predictions"]
            true_param_arr = res["true_params"]
            if pred_arr is None or true_param_arr is None:
                continue
            t = np.arange(pred_arr.shape[0], dtype=np.int64)
            param_order = res["param_names"]
            for i, name in enumerate(param_order):
                if i >= pred_arr.shape[1] or i >= true_param_arr.shape[1]:
                    continue
                if len(results) > 1:
                    pred_label = f"{res['label']}:{name} pred"
                    true_label = f"{res['label']}:{name} true"
                else:
                    pred_label = f"{name} pred"
                    true_label = f"{name} true"
                pred_axis.plot(t, pred_arr[:, i], label=pred_label)
                pred_axis.plot(t, true_param_arr[:, i], linestyle=":", linewidth=1.8, label=true_label)
        pred_axis.set_ylabel("param pred (scaled)")
        pred_axis.legend(loc="best")
        row += 1

    if show_mse:
        mse_axis = axes[row]
        unc_axis = mse_axis.twinx() if show_uncertainty else None
        for res in results:
            mse_arr = res["mse"]
            if mse_arr is None:
                continue
            t = np.arange(mse_arr.shape[0], dtype=np.int64)
            param_order = res["param_names"]
            for i, name in enumerate(param_order):
                if i < mse_arr.shape[1]:
                    if len(results) > 1:
                        mse_axis.plot(t, mse_arr[:, i], label=f"{res['label']}:{name}")
                    else:
                        mse_axis.plot(t, mse_arr[:, i], label=name)
            unc_arr = res["uncertainty"]
            if unc_axis is not None and unc_arr is not None:
                for i, name in enumerate(param_order):
                    if i < unc_arr.shape[1]:
                        if len(results) > 1:
                            unc_label = f"{res['label']}:{name} std"
                        else:
                            unc_label = f"{name} std"
                        unc_axis.plot(t, unc_arr[:, i], linestyle="--", alpha=0.9, label=unc_label)
        mse_axis.set_ylabel("reg sq. error")
        if unc_axis is not None:
            unc_axis.set_ylabel("pred std (scaled)")
            left_handles, left_labels = mse_axis.get_legend_handles_labels()
            right_handles, right_labels = unc_axis.get_legend_handles_labels()
            mse_axis.legend(left_handles + right_handles, left_labels + right_labels, loc="best")
        else:
            mse_axis.legend(loc="best")
        row += 1

    reward_axis = axes[row]
    for res in results:
        cum_r = np.cumsum(res["rewards"])
        reward_axis.plot(np.arange(cum_r.shape[0], dtype=np.int64), cum_r, label=res["label"])
    reward_axis.set_ylabel("cum. reward")
    if len(results) > 1:
        reward_axis.legend(loc="best")
    row += 1

    for i in range(action_dim):
        action_axis = axes[row]
        for res in results:
            A = res["actions"]
            action_axis.plot(np.arange(A.shape[0], dtype=np.int64), A[:, i], label=res["label"])
        action_axis.set_ylabel(f"action[{i}]")
        if i == 0 and len(results) > 1:
            action_axis.legend(loc="best")
        row += 1

    axes[-1].set_xlabel("timestep")

    if len(results) == 1:
        r0 = results[0]
        if r0["mse"] is not None and r0["true0"] is not None and r0["z_scale"] is not None:
            fig.suptitle(
                f"One episode | seed={EPISODE_SEED} | controller={r0['label']} | "
                f"{float(r0['z_scale']):g}x true_params={np.array2string(r0['true0'], precision=3)}"
                + (f" | true_theta={true_theta0:.3f}" if true_theta0 is not None else "")
            )
        else:
            fig.suptitle(f"One episode | seed={EPISODE_SEED} | controller={r0['label']}" + (f" | true_theta={true_theta0:.3f}" if true_theta0 is not None else ""))
    else:
        fig.suptitle(f"One episode compare | seed={EPISODE_SEED} | controllers=PPO vs LQR" + (f" | true_theta={true_theta0:.3f}" if true_theta0 is not None else ""))

    fig.tight_layout()

    out_path = os.path.join(test_dir, f"episode_seed{int(EPISODE_SEED)}_{CONTROLLER_MODE}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    for EPISODE_SEED in EPISODE_SEEDS:
        main()
