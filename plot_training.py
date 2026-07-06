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


experiment_dir = "./experiments/lqr_R_0_5_dB_0_1_2026-03-29_17-58-00_privileged_fine_tuned"

plot_reward_and_running_mean([experiment_dir], save_plot=True)