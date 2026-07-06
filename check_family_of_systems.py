# import gymnasium as gym
# import shimmy
# import shimmy.dm_control_compatibility  # registers dm_control/* env IDs
# from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.logger import configure
from scipy.linalg import solve_discrete_are
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
# from callbacks import LivePlotCallback, PrintProgressCallback, SaveModelCallback
# from wrappers import RewardWrapper, DomainRandomizationWrapper, ForceMujocoFixedCamera # , ParamObsWrapper, SwitchPenalty, HideVelocityObs
# from lqr_model import LinearQuadraticEnv, install_lqr_adapter_for_domain_randomization
# from helper_functions import plot_reward_and_running_mean
# from context_ppo import ContextPPO
# from unified_context_ppo import UnifiedContextPPO
from tqdm import tqdm

A_NOMINAL = np.array([[0.98, 0.0],[0.0, 0.95]], dtype=np.float64)
B_NOMINAL = np.array([[0.1, 0.03],[1.0, 0.2]], dtype=np.float64)
Q = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
R = np.array([[5.0, 0.0],[0.0, 0.2]])# 0.5 * np.eye(1, dtype=np.float64)
DELTA_B = np.array([[0.5, 0.02], [0.15, 1.0]], dtype=np.float64)
P_nom = solve_discrete_are(A_NOMINAL, B_NOMINAL, Q, R)
K_nom = np.linalg.solve(R + B_NOMINAL.T @ P_nom @ B_NOMINAL, B_NOMINAL.T @ P_nom @ A_NOMINAL)
DELTA_A = DELTA_B @ K_nom

K_list = []

def check_system(A, B, Q, R):
    C = np.concatenate([B, A @ B], axis=1)
    rank_C = np.linalg.matrix_rank(C)
    det_C = np.linalg.det(C) if C.shape[0] == C.shape[1] else np.nan
    cond_C = np.linalg.cond(C)

    result = {
        "rank_C": rank_C,
        "det_C": det_C,
        "cond_C": cond_C,
        "controllable": rank_C == A.shape[0],
    }

    try:
        P = solve_discrete_are(A, B, Q, R)
        M = R + B.T @ P @ B
        K = np.linalg.solve(M, B.T @ P @ A)
        Acl = A - B @ K
        result["dare_success"] = True
        result["P_sym_err"] = np.linalg.norm(P - P.T)
        result["P_norm"] = np.linalg.norm(P)
        result["cond_M"] = np.linalg.cond(M)
        result["rho_Acl"] = np.max(np.abs(np.linalg.eigvals(Acl)))
    except Exception as e:
        result["dare_success"] = False
        result["error"] = str(e)

    return result

for theta in tqdm(np.linspace(-1.0, 1.0, num=1000)):

    A = A_NOMINAL + DELTA_A * theta
    B = B_NOMINAL + DELTA_B * theta

    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

    info = check_system(A, B, Q, R)

    if (not info["controllable"]
        or not info["dare_success"]
        or info["cond_C"] > 1e8
        or info.get("cond_M", 1.0) > 1e10):
        print(f"Theta {theta:.3f} looks problematic: {info}")

    K_list.append(K)

    # A_cl = A - B @ K
    # eigvals = np.linalg.eigvals(A_cl)
    # stable = np.all(np.abs(eigvals) < 1.0)
    # print("Eigenvalues: ", eigvals)
    # print("Stable: ", stable)

    # print("Nominal K: ", K_nom)
    # print("Actual K: ", K)



plt.figure(figsize=(10, 6))
K_array = np.stack(K_list, axis=0)
for action_idx in range(K_array.shape[1]):
    for state_idx in range(K_array.shape[2]):
        plt.plot(K_array[:, action_idx, state_idx], label=f'K[{action_idx},{state_idx}]')
plt.xlabel('Theta Index')
plt.ylabel('K Values')
plt.title('LQR Gain K vs Theta')
plt.legend()
plt.grid()
plt.show()




#####

# =========================
# Short performance demo
# =========================

theta_demo = 0.25  # choose one hidden system from the family
T = 500
x0 = np.array([[0.5], [-0.5]], dtype=np.float64)

A_hidden = A_NOMINAL + DELTA_A * theta_demo
B_hidden = B_NOMINAL + DELTA_B * theta_demo

# true optimal gain for the hidden system
P_hidden = solve_discrete_are(A_hidden, B_hidden, Q, R)
K_hidden = np.linalg.solve(R + B_hidden.T @ P_hidden @ B_hidden, B_hidden.T @ P_hidden @ A_hidden)

print("\n===== PERFORMANCE DEMO =====")
print("theta_demo =", theta_demo)
print("K_nom     =", K_nom)
print("K_hidden  =", K_hidden)
print("closed-loop with K_nom on nominal:",
      np.linalg.eigvals(A_NOMINAL - B_NOMINAL @ K_nom))
print("closed-loop with K_nom on hidden :",
      np.linalg.eigvals(A_hidden - B_hidden @ K_nom))
print("closed-loop with K_hidden on hidden:",
      np.linalg.eigvals(A_hidden - B_hidden @ K_hidden))

def rollout(A, B, K, x0, T, Q, R):
    x = x0.copy()
    xs = [x.flatten()]
    us = []
    stage_costs = []
    cumulative_costs = []
    total_cost = 0.0

    for t in range(T):
        u = -K @ x
        cost = float(x.T @ Q @ x + u.T @ R @ u)
        total_cost += cost

        us.append(u.reshape(-1).copy())
        stage_costs.append(cost)
        cumulative_costs.append(total_cost)

        x = A @ x + B @ u #+ np.random.multivariate_normal(mean=np.zeros(x.shape[0]), cov=0.001 * np.eye(x.shape[0])).reshape(-1, 1)
        xs.append(x.flatten())

    return np.array(xs), np.vstack(us), np.array(stage_costs), np.array(cumulative_costs)

# 1) wrong assumption: apply nominal controller to hidden system
xs_wrong, us_wrong, costs_wrong, cum_wrong = rollout(
    A_hidden, B_hidden, K_nom, x0, T, Q, R
)

# 2) correct assumption: apply true optimal controller to hidden system
xs_opt, us_opt, costs_opt, cum_opt = rollout(
    A_hidden, B_hidden, K_hidden, x0, T, Q, R
)

print(f"Total cost on hidden system with K_nom   : {cum_wrong[-1]:.6f}")
print(f"Total cost on hidden system with K_hidden: {cum_opt[-1]:.6f}")
print(f"Cost ratio (wrong / optimal): {cum_wrong[-1] / cum_opt[-1]:.6f}")

time_axis_x = np.arange(T + 1)
time_axis_u = np.arange(T)

plt.figure(figsize=(10, 6))
plt.plot(time_axis_x, xs_wrong[:, 0], label='x1 with K_nom on hidden')
plt.plot(time_axis_x, xs_opt[:, 0], label='x1 with K_hidden on hidden')
plt.plot(time_axis_x, xs_wrong[:, 1], label='x2 with K_nom on hidden', linestyle='--')
plt.plot(time_axis_x, xs_opt[:, 1], label='x2 with K_hidden on hidden', linestyle='--')
plt.xlabel('Time step')
plt.ylabel('State')
plt.title(f'State trajectories on hidden system (theta={theta_demo})')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
for action_idx in range(us_wrong.shape[1]):
    plt.plot(time_axis_u, us_wrong[:, action_idx], label=f'u[{action_idx}] with K_nom on hidden')
    plt.plot(time_axis_u, us_opt[:, action_idx], label=f'u[{action_idx}] with K_hidden on hidden')
plt.xlabel('Time step')
plt.ylabel('Control input')
plt.title(f'Control inputs on hidden system (theta={theta_demo})')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time_axis_u, cum_wrong, label='Cumulative cost with K_nom on hidden')
plt.plot(time_axis_u, cum_opt, label='Cumulative cost with K_hidden on hidden')
plt.xlabel('Time step')
plt.ylabel('Cumulative cost')
plt.title(f'Performance gap on hidden system (theta={theta_demo})')
plt.legend()
plt.grid()
plt.show()
