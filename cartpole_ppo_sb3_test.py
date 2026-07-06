import os
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import gymnasium as gym
import shimmy
import shimmy.dm_control_compatibility  # registers dm_control/* env IDs
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from wrappers import RewardWrapper, DomainRandomizationWrapper, ForceMujocoFixedCamera # , ParamObsWrapper, SwitchPenalty, HideVelocityObs
from lqr_model import LinearQuadraticEnv, install_lqr_adapter_for_domain_randomization
from helper_functions import plot_reward_and_running_mean # , choose_cartpole_params
# from context_ppo import ContextPPO
from unified_context_ppo import UnifiedContextPPO
import torch as th

exp_dict = {
    1: "./experiments/stabler_lqr_2026-03-15_13-46-42_privileged",
    2: "./experiments/stabler_lqr_2026-03-15_13-46-42_privileged_orig_cost",
    3: "./experiments/stabler_lqr_2026-03-16_10-30-02_mle_better",
    4: "./experiments/stabler_lqr_2026-03-16_10-30-02_mle_process_noise_both",
    5: "./experiments/stabler_lqr_2026-03-17_08-49-04_vanilla",
    6: "./experiments/stabler_lqr_2026-03-17_08-49-04_vanilla_process_noise_both",

    7: "./experiments/stabler_lqr_2026-03-25_10-12-52_mle_win_len_20_no_warm-up_process_noise_3_in_both",
    8: "./experiments/stabler_lqr_2026-03-25_10-12-52_mle_win_len_20_no_warm-up_process_noise_3_in_both_naive_exp",

    # requires less process noise and R=0.5 and delta_B = [0 1]
    9: "./experiments/lqr_R_0_5_dB_0_1_2026-03-29_17-58-00_privileged",
}

### ---------- Simple test configuration ---------- ###
EXP_ID = 9
experiment = exp_dict[EXP_ID]

# Reproducible evaluation setup
N_EVAL_EPISODES = 1000
EVAL_BASE_SEED = 12345 # episode k uses seed = base + k
CONTROLLER_MODE = "ppo"  # "ppo" or "lqr"
NOMINAL_WARMUP_STEPS = 0  # set to 0 to disable; useful for older checkpoints whose saved config predates this option

experiment_dir = f"{experiment}/test"
os.makedirs(experiment_dir, exist_ok=True)

config_path = os.path.join(experiment, "config.yaml")
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

WRAPPER_REGISTRY = {"RewardWrapper": RewardWrapper,
                    "DomainRandomizationWrapper": DomainRandomizationWrapper,
                    "ChangingCartPoleDynamics": DomainRandomizationWrapper,
                    #"ParamObsWrapper": ParamObsWrapper,
                    #"HideVelocityObs": HideVelocityObs,
                    #"SwitchPenalty": SwitchPenalty,
                    }

def make_env():
    environment = str(cfg.get("environment", "cartpole")).lower()
    if environment == "cartpole":
        max_episode_steps = cfg.get("max_episode_steps", None)
        task_kwargs = None
        if max_episode_steps is not None:
            task_kwargs = {"time_limit": float(max_episode_steps) * 0.01}
        env = gym.make("dm_control/cartpole-swingup_sparse-v0", render_mode=cfg.get("render_mode", None), task_kwargs=task_kwargs)
        env = ForceMujocoFixedCamera(env, camera_id=0, width=1200, height=800)
    elif environment == "lqr":
        install_lqr_adapter_for_domain_randomization()
        lqr_cfg = dict(cfg.get("lqr_env", {}) or {})
        lqr_cfg.setdefault("max_episode_steps", int(cfg.get("max_episode_steps", 200)))
        env = LinearQuadraticEnv(**lqr_cfg)
    else:
        raise ValueError(f"Unknown environment={environment}. Expected 'cartpole' or 'lqr'.")

    # creating wrappers from config (with deterministic eval override)
    wr_specs = cfg.get("wrappers", [])
    for wr_spec in wr_specs:
        if wr_spec.get("enabled", True) is False:
            continue

        wr_name = wr_spec["name"]
        wr_params = wr_spec.get("params", {}) or {}

        # Deterministic evaluation override:
        # reseed the wrapper RNG so each episode is fully determined by reset(seed=...).
        if wr_name in {"DomainRandomizationWrapper", "ChangingCartPoleDynamics"}:
            wr_params = dict(wr_params)
            wr_params.setdefault("reseed_on_reset", True)
            wr_params.setdefault("rng_seed", EVAL_BASE_SEED)

        env = WRAPPER_REGISTRY[wr_name](env, **wr_params)

    # adding monitor wrapper from SB3
    env = Monitor(env, filename=os.path.join(experiment_dir, "monitor.csv"), info_keywords=())

    return env

env = make_env()


# create model from config
MODEL_REGISTRY = {"PPO": PPO, "UnifiedContextPPO": UnifiedContextPPO} # "ContextPPO": ContextPPO,
model_spec = cfg.get("model", None)
model_name = model_spec["name"]
model_params = model_spec.get("params", {}) or {}

# Sanity-check which weights file is being loaded
weights_path = f"{experiment}/weights_best" if os.path.exists(f"{experiment}/weights_best.zip") else f"{experiment}/weights"

if CONTROLLER_MODE == "ppo":
    model = MODEL_REGISTRY[model_name].load(weights_path, env=env)
    if hasattr(model, "nominal_warmup_steps"):
        model.nominal_warmup_steps = int(NOMINAL_WARMUP_STEPS)
    print(f"Loaded model type: {type(model).__name__}")
elif CONTROLLER_MODE == "lqr":
    model = None
else:
    raise ValueError(f"Unknown CONTROLLER_MODE={CONTROLLER_MODE!r}. Expected 'ppo' or 'lqr'.")


def _find_wrapper_attr(env: gym.Env, attr: str):
    e = env
    while True:
        if hasattr(e, attr):
            return getattr(e, attr)
        if not hasattr(e, "env"):
            break
        e = e.env
    raise AttributeError(f"Could not find attribute {attr!r} in env wrapper stack")


get_params = None
get_param_denoms = None
get_params_dict = None
get_param_denom_dict = None
try:
    get_params = _find_wrapper_attr(env, "get_true_params")
    get_param_denoms = _find_wrapper_attr(env, "get_param_denoms")
    get_params_dict = _find_wrapper_attr(env, "get_true_params_dict")
    get_param_denom_dict = _find_wrapper_attr(env, "get_param_denom_dict")
except Exception:
    get_params = None
    get_param_denoms = None
    get_params_dict = None
    get_param_denom_dict = None


logged_param_names: list[str] = []
if get_params_dict is not None and get_param_denom_dict is not None:
    try:
        params = get_params_dict()
        denoms = get_param_denom_dict()
        logged_param_names = list(params.keys())
    except Exception:
        logged_param_names = []


episode_rewards: list[float] = []
episode_lengths: list[int] = []
episode_params: list[np.ndarray] = []
episode_regression_losses: list[float] = []
episode_regression_uncertainties: list[float] = []

for ep in tqdm(range(N_EVAL_EPISODES)):
    seed = EVAL_BASE_SEED + ep
    obs, info = env.reset(seed=seed)

    # Keep the recurrent-like state returned by ContextPPO.predict().
    # If we drop it, the context window resets every step (and the encoder sees only zeros).
    state = None
    episode_start = np.ones((1,), dtype=bool)

    if get_params_dict is not None and get_param_denom_dict is not None and logged_param_names:
        params = get_params_dict()
        denoms = get_param_denom_dict()
        episode_params.append(
            np.asarray(
                [float(params[name]) / max(float(denoms[name]), 1e-8) for name in logged_param_names],
                dtype=np.float32,
            )
        )

    terminated = False
    truncated = False
    ep_return = 0.0
    ep_len = 0
    episode_step_regression_losses: list[float] = []
    episode_step_regression_uncertainties: list[float] = []

    while not (terminated or truncated):
        if CONTROLLER_MODE == "lqr":
            action = _find_wrapper_attr(env, "lqr_action")(obs)
        else:
            action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=True)
        if (
            CONTROLLER_MODE == "ppo"
            and
            get_params is not None
            and state is not None
            and hasattr(model, "_build_traj_window")
            and str(getattr(model, "context_mode", "")).startswith("encoder")
        ):
            traj_window_np = model._build_traj_window(*state)
            traj_window_th = th.as_tensor(traj_window_np, device=model.device, dtype=th.float32)
            true_raw = np.asarray(get_params(model.regression_param_names), dtype=np.float32)
            true_denoms = np.asarray(get_param_denoms(model.regression_param_names), dtype=np.float32)
            # true_params = np.log(np.maximum(true_raw / np.maximum(true_denoms, 1e-8), 1e-8)) * float(getattr(model, "z_scale", 1.0))
            true_params = (true_raw / np.maximum(true_denoms, 1e-8)) * float(getattr(model, "z_scale", 1.0))
            true_params_th = th.as_tensor(true_params[None, :], device=model.device, dtype=th.float32)
            with th.no_grad():
                if str(getattr(model, "context_mode", "")) == "encoder_nll":
                    mu, logvar = model.policy.encode_context(traj_window_th, return_logvar=True)
                    logvar = th.clamp(logvar, min=-10.0, max=5.0)
                    episode_step_regression_losses.append(float(th.mean((mu - true_params_th) ** 2).cpu().item()))
                    episode_step_regression_uncertainties.append(float(logvar.exp().mean().cpu().item()))
                else:
                    pred_params = model.policy.encode_context(traj_window_th)
                    episode_step_regression_losses.append(float(th.mean((pred_params - true_params_th) ** 2).cpu().item()))
        # ContextPPO.predict() returns actions shaped (n_envs, act_dim). When stepping a single
        # non-VecEnv gym env, squeeze the leading dimension to match the env action shape.
        action_env = action
        if isinstance(action_env, np.ndarray) and action_env.ndim >= 1 and action_env.shape[0] == 1:
            action_env = action_env[0]
        obs, reward, terminated, truncated, info = env.step(action_env)
        ep_return += float(reward)
        ep_len += 1

        episode_start[...] = bool(terminated or truncated)

    print(f"Episode {ep+1}/{N_EVAL_EPISODES} | Return: {ep_return:.2f} | Length: {ep_len}")
    episode_rewards.append(float(ep_return))
    episode_lengths.append(int(ep_len))
    if episode_step_regression_losses:
        episode_regression_losses.append(float(np.mean(episode_step_regression_losses)))
    if episode_step_regression_uncertainties:
        episode_regression_uncertainties.append(float(np.mean(episode_step_regression_uncertainties)))


### ---------- Plotting and saving results ---------- ###

info_dict = {}
info_dict["num_episodes"] = N_EVAL_EPISODES
info_dict["base_seed"] = EVAL_BASE_SEED
info_dict["reward_mean"] = round(np.mean(episode_rewards), 2)
info_dict["reward_std"] = round(np.std(episode_rewards), 2)
info_dict["reward_median"] = round(np.median(episode_rewards), 2)
info_dict["reward_10th_pct"] = round(np.percentile(episode_rewards, 10), 2)
info_dict["reward_25th_pct"] = round(np.percentile(episode_rewards, 25), 2)

print(f"Mean reward over {N_EVAL_EPISODES} episodes: {info_dict['reward_mean']:.2f} +/- {info_dict['reward_std']:.2f}")
print(f"Median reward: {info_dict['reward_median']:.2f} | 10th pct: {info_dict['reward_10th_pct']:.2f} | 25th pct: {info_dict['reward_25th_pct']:.2f}")

# Save a deterministic audit trail of evaluation episodes
df_out = pd.DataFrame(
    {
        "episode": np.arange(N_EVAL_EPISODES, dtype=np.int64),
        "seed": np.arange(EVAL_BASE_SEED, EVAL_BASE_SEED + N_EVAL_EPISODES, dtype=np.int64),
        "reward": np.asarray(episode_rewards, dtype=np.float64),
        "length": np.asarray(episode_lengths, dtype=np.int64),
    }
)

if len(episode_params) == N_EVAL_EPISODES:
    P = np.stack(episode_params, axis=0)
    for i, name in enumerate(logged_param_names):
        if i < P.shape[1]:
            df_out[f"{name}_norm"] = P[:, i]

if len(episode_regression_losses) == N_EVAL_EPISODES:
    df_out["regression_loss"] = np.asarray(episode_regression_losses, dtype=np.float64)

if len(episode_regression_uncertainties) == N_EVAL_EPISODES:
    df_out["regression_uncertainty"] = np.asarray(episode_regression_uncertainties, dtype=np.float64)

out_csv = os.path.join(experiment_dir, "eval_episodes.csv")
df_out.to_csv(out_csv, index=False)
print(f"Wrote eval episode log to: {out_csv}")

if len(episode_regression_losses) == N_EVAL_EPISODES:
    progress_df = pd.DataFrame({"train/regression_loss": np.asarray(episode_regression_losses, dtype=np.float64)})
    if len(episode_regression_uncertainties) == N_EVAL_EPISODES:
        progress_df["train/regression_uncertainty"] = np.asarray(episode_regression_uncertainties, dtype=np.float64)
    progress_df.to_csv(os.path.join(experiment_dir, "progress.csv"), index=False)

env.close()


### ---------- Plotting ---------- ###
plot_reward_and_running_mean([experiment_dir], save_plot=True, info_dict=info_dict)
