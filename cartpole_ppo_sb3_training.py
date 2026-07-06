import gymnasium as gym
from lqr_model import EmergencyBrakeEnv, LinearQuadraticEnv, StoppingCarEnv, install_lqr_adapter_for_domain_randomization
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure
import os
from datetime import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import torch
import random
import yaml
import numpy as np
from callbacks import LivePlotCallback, PrintProgressCallback, SaveModelCallback
from wrappers import PreviousActionObservationWrapper, RewardWrapper, DomainRandomizationWrapper, ForceMujocoFixedCamera # , ParamObsWrapper, SwitchPenalty, HideVelocityObs
from helper_functions import plot_reward_and_running_mean
# from context_ppo import ContextPPO
from unified_context_ppo import UnifiedContextPPO

# open config file and save into experiment folder
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)


def _resolve_training_paths(cfg: dict) -> tuple[str, bool, str | None]:
    if "training" not in cfg or cfg["training"] is None:
        raise KeyError("Missing required 'training' section in config.yaml.")

    training_cfg = dict(cfg["training"])
    experiment_root = str(training_cfg["experiment_root"])
    experiment_name = training_cfg.get("experiment_name")

    if experiment_name is not None and str(experiment_name).strip():
        experiment_dir = os.path.join(experiment_root, str(experiment_name))
    else:
        experiment_name_suffix = str(training_cfg["experiment_name_suffix"])
        timestamp = datetime.now().strftime("%m-%d__%H-%M")
        experiment_dir = os.path.join(experiment_root, f"s_{timestamp}_{experiment_name_suffix}")

    load_weights = bool(training_cfg["load_weights"])
    load_weights_from_raw = training_cfg.get("load_weights_from")
    load_weights_from = None if load_weights_from_raw is None else str(load_weights_from_raw)
    if load_weights and not load_weights_from:
        raise ValueError(
            "training.load_weights_from must be set when training.load_weights is true."
        )
    return experiment_dir, load_weights, load_weights_from


def _adapt_context_input_weight(param_name: str, src_value: torch.Tensor, tgt_value: torch.Tensor) -> torch.Tensor | None:
    if src_value.ndim != 2 or tgt_value.ndim != 2:
        return None
    if param_name in {"actor_mlp.0.weight", "critic_mlp.0.weight"}:
        if src_value.shape[0] != tgt_value.shape[0] or src_value.shape[1] >= tgt_value.shape[1]:
            return None
        adapted = torch.zeros_like(tgt_value)
        adapted[:, : src_value.shape[1]] = src_value
        return adapted
    if param_name.startswith("context_encoder.") and param_name.endswith(".weight"):
        if src_value.shape[0] >= tgt_value.shape[0] or src_value.shape[1] != tgt_value.shape[1]:
            return None
        adapted = torch.zeros_like(tgt_value)
        adapted[: src_value.shape[0], :] = src_value
        return adapted
    return None


def _adapt_context_output_bias(param_name: str, src_value: torch.Tensor, tgt_value: torch.Tensor) -> torch.Tensor | None:
    if not (param_name.startswith("context_encoder.") and param_name.endswith(".bias")):
        return None
    if src_value.ndim != 1 or tgt_value.ndim != 1 or src_value.shape[0] >= tgt_value.shape[0]:
        return None
    adapted = torch.zeros_like(tgt_value)
    adapted[: src_value.shape[0]] = src_value
    return adapted


def _build_compatible_policy_state(
    target_model,
    loaded_model,
    *,
    skip_loaded_encoder: bool,
    load_encoder_only: bool,
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    src_state = loaded_model.policy.state_dict()
    tgt_state = target_model.policy.state_dict()

    compatible: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    adapted: list[str] = []

    for key, src_value in src_state.items():
        if load_encoder_only and not key.startswith("context_encoder."):
            skipped.append(key)
            continue
        if skip_loaded_encoder and key.startswith("context_encoder."):
            skipped.append(key)
            continue

        tgt_value = tgt_state.get(key)
        if tgt_value is None:
            skipped.append(key)
            continue

        if src_value.shape == tgt_value.shape:
            compatible[key] = src_value
            continue

        adapted_value = _adapt_context_input_weight(key, src_value, tgt_value)
        if adapted_value is None:
            adapted_value = _adapt_context_output_bias(key, src_value, tgt_value)
        if adapted_value is not None:
            compatible[key] = adapted_value
            adapted.append(key)
            continue

        skipped.append(key)

    return compatible, skipped, adapted


# Create experiment folder
experiment_dir, load_weights, load_weights_from = _resolve_training_paths(cfg)
os.makedirs(experiment_dir, exist_ok=True)
shutil.copy2("config.yaml", os.path.join(experiment_dir, "config.yaml"))


# create callbacks
CALLBACK_REGISTRY = {
    "LivePlotCallback": LivePlotCallback,
    "PrintProgressCallback": PrintProgressCallback,
    "SaveModelCallback": SaveModelCallback,
}
cb_specs = cfg.get("callbacks", [])
callbacks = []

for cb_spec in cb_specs:
    if cb_spec.get("enabled", True) is False:
        continue
    cb_name = cb_spec["name"]
    cb_params = cb_spec.get("params", {}) or {}

    if cb_name == "SaveModelCallback":
        cb_params = dict(cb_params)
        cb_params.setdefault("save_dir", experiment_dir)
    callbacks.append(CALLBACK_REGISTRY[cb_name](**cb_params))

cb_by_type = {type(cb): cb for cb in callbacks}
live_plot_cb = cb_by_type.get(LivePlotCallback)
if live_plot_cb is not None:
    live_plot_cb.periodic_save_path = os.path.join(experiment_dir, "reward_plot_live.png")
    live_plot_cb.episode_csv_path = os.path.join(experiment_dir, "episode_returns_live.csv")
    live_plot_cb.regression_csv_path = os.path.join(experiment_dir, "regression_live.csv")
    hist_path = live_plot_cb.episode_csv_path if os.path.exists(live_plot_cb.episode_csv_path) else os.path.join(experiment_dir, "monitor.csv")
    loaded_any = False
    if os.path.exists(hist_path):
        hist_df = pd.read_csv(hist_path, comment="#")
        historical_returns = hist_df["r"].astype(float).tolist()
        if historical_returns:
            live_plot_cb.returns = historical_returns
            live_plot_cb.shaped_returns = hist_df["r_shaped"].astype(float).tolist() if "r_shaped" in hist_df.columns else historical_returns.copy()
            window = live_plot_cb.running_mean_window
            live_plot_cb.running_mean = [
                float(np.mean(historical_returns[max(0, i - window + 1) : i + 1]))
                for i in range(len(historical_returns))
            ]
            loaded_any = True
    if os.path.exists(live_plot_cb.regression_csv_path):
        reg_df = pd.read_csv(live_plot_cb.regression_csv_path)
        if not reg_df.empty:
            live_plot_cb.regression_losses = pd.to_numeric(reg_df["regression_loss"], errors="coerce").fillna(0.0).astype(float).tolist()
            live_plot_cb.regression_mses = pd.to_numeric(reg_df["regression_mse"], errors="coerce").astype(float).tolist() if "regression_mse" in reg_df.columns else []
            live_plot_cb.uncertainties = pd.to_numeric(reg_df["uncertainty"], errors="coerce").astype(float).tolist() if "uncertainty" in reg_df.columns else []
            loaded_any = True
    if loaded_any:
        live_plot_cb._update_plots()


# read training parameters from config
total_timesteps = cfg.get("total_timesteps", 100000)


# Creating environment
WRAPPER_REGISTRY = {"RewardWrapper": RewardWrapper,
                    "DomainRandomizationWrapper": DomainRandomizationWrapper,
                    "ChangingCartPoleDynamics": DomainRandomizationWrapper,
                    "PreviousActionObservationWrapper": PreviousActionObservationWrapper,
                    }

training_cfg = cfg.get("training", {}) or {}
num_envs = int(training_cfg.get("num_envs", 1))
vec_env_type = str(training_cfg.get("vec_env_type", "dummy")).lower()
if num_envs < 1:
    raise ValueError(f"training.num_envs must be >= 1, got {num_envs}.")
if vec_env_type not in {"dummy", "subproc"}:
    raise ValueError(f"training.vec_env_type must be 'dummy' or 'subproc', got {vec_env_type!r}.")


def make_env(rank: int = 0):
    def _init():
        install_lqr_adapter_for_domain_randomization()
        environment = str(cfg.get("environment", "lqr")).lower()
        if environment == "stopping_car":
            env_cfg = dict(cfg.get("stopping_car_env", {}) or {})
            env_cfg.setdefault("max_episode_steps", int(cfg.get("max_episode_steps", 80)))
            env = StoppingCarEnv(**env_cfg)
        elif environment == "emergency_brake":
            env_cfg = dict(cfg.get("emergency_brake_env", {}) or {})
            env_cfg.setdefault("max_episode_steps", int(cfg.get("max_episode_steps", 120)))
            env = EmergencyBrakeEnv(**env_cfg)
        else:
            lqr_cfg = dict(cfg.get("lqr_env", {}) or {})
            lqr_cfg.setdefault("max_episode_steps", int(cfg.get("max_episode_steps", 200)))
            env = LinearQuadraticEnv(**lqr_cfg)

        # creating wrappers from config
        wr_specs = cfg.get("wrappers", [])
        for wr_spec in wr_specs:
            if wr_spec.get("enabled", True) is False:
                continue

            wr_name = wr_spec["name"]
            wr_params = wr_spec.get("params", {}) or {}
            env = WRAPPER_REGISTRY[wr_name](env, **wr_params)

        # adding monitor wrapper from SB3
        monitor_name = "monitor.csv" if num_envs == 1 else f"monitor_{rank}.csv"
        monitor_path = os.path.join(experiment_dir, monitor_name)
        env = Monitor(env, filename=monitor_path, info_keywords=(), override_existing=not os.path.exists(monitor_path))

        return env

    return _init


if num_envs == 1:
    env = DummyVecEnv([make_env(0)])
elif vec_env_type == "subproc":
    env = SubprocVecEnv([make_env(rank) for rank in range(num_envs)], start_method="fork")
else:
    env = DummyVecEnv([make_env(rank) for rank in range(num_envs)])


# create model from config
MODEL_REGISTRY = {"PPO": PPO, "RecurrentPPO": RecurrentPPO, "UnifiedContextPPO": UnifiedContextPPO} # "ContextPPO": ContextPPO,
model_spec = cfg.get("model", None)
model_name = model_spec["name"]
model_params = model_spec.get("params", {}) or {}
model = MODEL_REGISTRY[model_name](env=env, **model_params)

progress_path = os.path.join(experiment_dir, "progress.csv"); progress_prev_path = os.path.join(experiment_dir, "progress_prev.csv")
if os.path.exists(progress_path): shutil.copy2(progress_path, progress_prev_path)
model.set_logger(configure(experiment_dir, ["stdout", "csv"]))


def validate_loaded_model_compatibility(target_model, loaded_model) -> None:
    if hasattr(target_model, "regression_param_names") and hasattr(loaded_model, "regression_param_names"):
        if list(target_model.regression_param_names) != list(loaded_model.regression_param_names):
            raise ValueError(
                "Checkpoint regression_param_names do not match the current config. "
                "Keep the same parameter names in the same order when reusing latent-conditioned weights."
            )
        if float(getattr(target_model, "z_scale", 1.0)) != float(getattr(loaded_model, "z_scale", 1.0)):
            raise ValueError(
                "Checkpoint z_scale does not match the current config. "
                "Keep the latent scaling unchanged when reusing latent-conditioned weights."
            )


def should_skip_loaded_encoder(target_model, loaded_model) -> bool:
    return (
        isinstance(target_model, UnifiedContextPPO)
        and isinstance(loaded_model, UnifiedContextPPO)
        and str(getattr(loaded_model, "context_mode", "")) == "privileged"
        and str(getattr(target_model, "context_mode", "")).startswith("encoder")
    )

if load_weights:
    if load_weights_from is not None:
        load_weights_name = str(cfg.get("training", {}).get("load_weights_name", "weights_best"))
        load_encoder_only = bool(cfg.get("training", {}).get("load_encoder_only", False))
        loaded = MODEL_REGISTRY[model_name].load(f"{load_weights_from}/{load_weights_name}", env=env)
        validate_loaded_model_compatibility(model, loaded)
        skip_loaded_encoder = should_skip_loaded_encoder(model, loaded)
        compatible, skipped, adapted = _build_compatible_policy_state(
            model,
            loaded,
            skip_loaded_encoder=skip_loaded_encoder,
            load_encoder_only=load_encoder_only,
        )
        if skip_loaded_encoder:
            print("[load_weights_from] Skipping context_encoder.* because source checkpoint is privileged and target is encoder-based.")
        if adapted:
            print(
                "[load_weights_from] Adapted context input weights for expanded policy inputs: "
                f"{adapted}"
            )
        if skipped:
            print(f"[load_weights_from] Skipped {len(skipped)} params with shape mismatch: {skipped}")
        model.policy.load_state_dict(compatible, strict=False)
        del loaded, compatible  # free memory
    else:
        print("training.load_weights is true but training.load_weights_from is not set. Skipping weight loading.")

model.learn(total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            log_interval=1)

env.close()
if os.path.exists(progress_prev_path): pd.concat([pd.read_csv(progress_prev_path), pd.read_csv(progress_path)], ignore_index=True).to_csv(progress_path, index=False); os.remove(progress_prev_path)



### ---------- Plotting ---------- ###

live_plot_cb = cb_by_type.get(LivePlotCallback)
if live_plot_cb is not None:
    out_path = os.path.join(experiment_dir, "reward_plot_live.png")
    live_plot_cb.fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved reward plot to: {out_path}")
    # Keep the live plot open after training
    plt.ioff()
    # plt.show()
else:
    plot_reward_and_running_mean([experiment_dir], save_plot=True)
