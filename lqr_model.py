from __future__ import annotations

import math
from typing import Optional, Sequence

import gymnasium as gym
import numpy as np
from scipy.linalg import solve_discrete_are


def _as_1d_array(value: float | Sequence[float], size: int, *, dtype=np.float64) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        return np.full((size,), float(arr), dtype=dtype)
    arr = arr.reshape(-1)
    if arr.shape[0] != size:
        raise ValueError(f"Expected vector of size {size}, got {arr.shape[0]}")
    return arr


class LinearQuadraticEnv(gym.Env):
    """
    Discrete-time linear dynamics with quadratic stage cost:
      x_{t+1} = A x_t + B u_t + w_t
      r_t = -(x_t^T Q x_t + u_t^T R u_t)
    """

    metadata = {"render_modes": []}

    PARAM_ORDER = ("theta", "process_noise_scale", "a", "b")
    PARAM_DENOMS = {"theta": 1.0, "process_noise_scale": 1.0, "a": 1.0, "b": 1.0}

    def __init__(
        self,
        A: Optional[Sequence[Sequence[float]]] = None,
        B: Optional[Sequence[Sequence[float]]] = None,
        delta_B: Optional[Sequence[Sequence[float]]] = None,
        Q: Optional[Sequence[Sequence[float]]] = None,
        R: Optional[Sequence[Sequence[float]]] = None,
        process_noise_std: float | Sequence[float] = 0.05,
        initial_state_low: float | Sequence[float] = -1.0,
        initial_state_high: float | Sequence[float] = 1.0,
        action_low: float | Sequence[float] = -30.0,
        action_high: float | Sequence[float] = 30.0,
        max_episode_steps: int = 200,
        state_termination_bound: Optional[float | Sequence[float]] = 1000000.0,
        reward_cost_mode: str = "log",
        reward_cost_scale: float = 1.0,
        action_cost_type: str = "quadratic",
        smooth_sublinear_power: float = 0.25,
        smooth_sublinear_scale: float = 1.0,
        smooth_sublinear_weight: float = 1.0,
    ):
        super().__init__()

        default_A = np.array([[0.98, 0.0], [0.0, 0.95]], dtype=np.float64) # np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        default_B = np.array([[0.1], [1.0]], dtype=np.float64) # np.array([[0.0], [1.0]], dtype=np.float64)

        self.A_nominal = np.asarray(default_A if A is None else A, dtype=np.float64)
        self.B_nominal = np.asarray(default_B if B is None else B, dtype=np.float64)

        if self.A_nominal.ndim != 2 or self.A_nominal.shape[0] != self.A_nominal.shape[1]:
            raise ValueError(f"A must have shape (n, n), got {self.A_nominal.shape}")
        if self.B_nominal.ndim != 2:
            raise ValueError(f"B must have shape (n, m), got {self.B_nominal.shape}")
        if self.B_nominal.shape[0] != self.A_nominal.shape[0]:
            raise ValueError(
                f"A and B must share state dimension, got A {self.A_nominal.shape}, B {self.B_nominal.shape}"
            )

        self.state_dim = int(self.A_nominal.shape[0])
        self.action_dim = int(self.B_nominal.shape[1])
        default_delta_B = np.array([[0.0], [1.0]], dtype=np.float64)
        self.delta_B = np.asarray(default_delta_B if delta_B is None else delta_B, dtype=np.float64)
        if self.delta_B.shape != self.B_nominal.shape:
            raise ValueError(
                f"delta_B must have shape {self.B_nominal.shape}, got {self.delta_B.shape}"
            )

        self.Q = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64) if Q is None else np.asarray(Q, dtype=np.float64)
        # self.R = 5.0 * np.eye(self.action_dim, dtype=np.float64) if R is None else np.asarray(R, dtype=np.float64)
        self.R = 0.5 * np.eye(self.action_dim, dtype=np.float64) if R is None else np.asarray(R, dtype=np.float64)
        if self.Q.shape != (self.state_dim, self.state_dim):
            raise ValueError(f"Q must have shape ({self.state_dim}, {self.state_dim}), got {self.Q.shape}")
        if self.R.shape != (self.action_dim, self.action_dim):
            raise ValueError(f"R must have shape ({self.action_dim}, {self.action_dim}), got {self.R.shape}")

        self.process_noise_std = _as_1d_array(process_noise_std, self.state_dim, dtype=np.float64)
        self.initial_state_low = _as_1d_array(initial_state_low, self.state_dim, dtype=np.float64)
        self.initial_state_high = _as_1d_array(initial_state_high, self.state_dim, dtype=np.float64)
        if np.any(self.initial_state_high < self.initial_state_low):
            raise ValueError("initial_state_high must be >= initial_state_low elementwise")

        action_low_arr = _as_1d_array(action_low, self.action_dim, dtype=np.float32)
        action_high_arr = _as_1d_array(action_high, self.action_dim, dtype=np.float32)
        if np.any(action_high_arr <= action_low_arr):
            raise ValueError("action_high must be > action_low elementwise")

        self.action_space = gym.spaces.Box(
            low=action_low_arr,
            high=action_high_arr,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        self.max_episode_steps = int(max_episode_steps)
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be > 0")
        self.reward_cost_mode = str(reward_cost_mode)
        if self.reward_cost_mode not in {"log", "raw", "scaled_raw"}:
            raise ValueError(
                "reward_cost_mode must be one of {'log', 'raw', 'scaled_raw'}, "
                f"got {self.reward_cost_mode!r}."
            )
        self.reward_cost_scale = float(reward_cost_scale)
        if self.reward_cost_scale <= 0.0:
            raise ValueError("reward_cost_scale must be positive.")
        self.action_cost_type = str(action_cost_type)
        if self.action_cost_type not in {"quadratic", "smooth_sublinear"}:
            raise ValueError(
                "action_cost_type must be one of {'quadratic', 'smooth_sublinear'}, "
                f"got {self.action_cost_type!r}."
            )
        self.smooth_sublinear_power = float(smooth_sublinear_power)
        if not 0.0 < self.smooth_sublinear_power < 1.0:
            raise ValueError("smooth_sublinear_power must be in (0, 1).")
        self.smooth_sublinear_scale = float(smooth_sublinear_scale)
        if self.smooth_sublinear_scale <= 0.0:
            raise ValueError("smooth_sublinear_scale must be positive.")
        self.smooth_sublinear_weight = float(smooth_sublinear_weight)
        if self.smooth_sublinear_weight <= 0.0:
            raise ValueError("smooth_sublinear_weight must be positive.")

        self.state_termination_bound = (
            None
            if state_termination_bound is None
            else _as_1d_array(state_termination_bound, self.state_dim, dtype=np.float64)
        )

        self.theta = 0.0
        self.process_noise_scale = 1.0
        self.A = self.A_nominal.copy()
        self.B = self.B_nominal.copy()
        self.a = float(self.A[0, 0]) if self.A.shape == (1, 1) else float("nan")
        self.b = float(self.B[0, 0]) if self.B.shape == (1, 1) else float("nan")

        self.state = np.zeros(self.state_dim, dtype=np.float64)
        self._step_count = 0

    def _update_dynamics_matrices(self) -> None:
        try:
            P_nominal = solve_discrete_are(self.A_nominal, self.B_nominal, self.Q, self.R)
            self.K_nominal = np.linalg.solve(self.R + self.B_nominal.T @ P_nominal @ (self.B_nominal), self.B_nominal.T @ P_nominal @ (self.A_nominal))
        except Exception as e:
            print(f"Error occurred while computing LQR gain: {e}")
            self.K_nominal = np.zeros((self.action_dim, self.state_dim), dtype=np.float64)
        
        self.delta_A = self.delta_B @ self.K_nominal
        self.A = self.A_nominal + self.delta_A * self.theta
        self.B = self.B_nominal + self.delta_B * self.theta
        self.a = float(self.A[0, 0]) if self.A.shape == (1, 1) else float("nan")
        self.b = float(self.B[0, 0]) if self.B.shape == (1, 1) else float("nan")

    def set_dynamics_scales(self, theta: float, process_noise_scale: float = 1.0) -> None:
        theta = float(theta)
        process_noise_scale = float(process_noise_scale)
        # if theta <= 0.0 or process_noise_scale <= 0.0:
        #     raise ValueError("Dynamics scales must be strictly positive.")
        self.theta = theta
        self.process_noise_scale = process_noise_scale
        self._update_dynamics_matrices()

    def set_scalar_ab_params(
        self,
        a: float,
        b: float,
        process_noise_scale: float = 1.0,
    ) -> None:
        if self.A_nominal.shape != (1, 1) or self.B_nominal.shape != (1, 1):
            raise ValueError(
                "set_scalar_ab_params requires scalar A and B matrices, "
                f"got A={self.A_nominal.shape}, B={self.B_nominal.shape}."
            )
        self.theta = 0.0
        self.process_noise_scale = float(process_noise_scale)
        self.a = float(a)
        self.b = float(b)
        self.A = np.asarray([[self.a]], dtype=np.float64)
        self.B = np.asarray([[self.b]], dtype=np.float64)

    def get_dynamics_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        return self.A.copy(), self.B.copy()

    def lqr_gain(self) -> np.ndarray:
        # u_t = -K x_t, with K from discrete-time ARE
        from scipy.linalg import solve_discrete_are

        # A_err = self.A_nominal + np.array([[7.59518755, 0.24217761], [0.0, 0.0]], dtype=np.float64) * 2
        # B_err = self.B_nominal + np.array([[1.0], [0.0]], dtype=np.float64) * 2
        # f = np.sqrt(1.0)
        # P = solve_discrete_are(f * A_err, f * B_err, self.Q, self.R)
        # K = np.linalg.solve(self.R + f * B_err.T @ P @ (f * B_err), f * B_err.T @ P @ (f * A_err))

        f = np.sqrt(1.0)
        P = solve_discrete_are(f * self.A, f * self.B, self.Q, self.R)
        K = np.linalg.solve(self.R + f * self.B.T @ P @ (f * self.B), f * self.B.T @ P @ (f * self.A))
        return K

    def lqr_action(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        x = self.state if state is None else np.asarray(state, dtype=np.float64).reshape(-1)
        K = self.lqr_gain()
        u = -(K @ x)
        return np.clip(u, self.action_space.low, self.action_space.high).astype(np.float32)

    def get_true_params_dict(self) -> dict[str, float]:
        return {
            "theta": float(self.theta),
            "process_noise_scale": float(self.process_noise_scale),
            "a": float(self.a),
            "b": float(self.b),
        }

    def get_param_denom_dict(self) -> dict[str, float]:
        return dict(self.PARAM_DENOMS)

    def get_true_params(self, param_names: Sequence[str]) -> np.ndarray:
        params = self.get_true_params_dict()
        return np.asarray([params[name] for name in param_names], dtype=np.float32)

    def get_param_denoms(self, param_names: Sequence[str]) -> np.ndarray:
        denoms = self.get_param_denom_dict()
        return np.asarray([denoms[name] for name in param_names], dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._step_count = 0
        self.state = self.np_random.uniform(self.initial_state_low, self.initial_state_high).astype(np.float64)

        info = {
            "A": self.A.copy(),
            "B": self.B.copy(),
            "true_params": self.get_true_params_dict(),
        }
        return self.state.astype(np.float32), info

    def step(self, action: np.ndarray):
        u = np.asarray(action, dtype=np.float64).reshape(-1)
        if u.shape[0] != self.action_dim:
            raise ValueError(f"Expected action dimension {self.action_dim}, got {u.shape[0]}")
        u = np.clip(u, self.action_space.low, self.action_space.high)

        state_cost = float(self.state.T @ self.Q @ self.state)
        quadratic_action_cost = float(u.T @ self.R @ u)
        if self.action_cost_type == "quadratic":
            action_cost = quadratic_action_cost
        else:
            action_cost = float(
                self.smooth_sublinear_weight
                * (
                    (1.0 + quadratic_action_cost / self.smooth_sublinear_scale)
                    ** self.smooth_sublinear_power
                    - 1.0
                )
            )
        true_cost = float(state_cost + action_cost)
        if self.reward_cost_mode == "log":
            cost = float(np.log1p(true_cost))
        elif self.reward_cost_mode == "raw":
            cost = true_cost
        else:
            cost = float(true_cost / self.reward_cost_scale)
        reward = -cost

        noise = self.np_random.normal(0.0, self.process_noise_std * self.process_noise_scale, size=self.state_dim)
        # noise[0] = 0.0  # only x[1] gets process noise (for 2D state)
        self.state = np.asarray(self.A @ self.state + self.B @ u + noise, dtype=np.float64)

        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps
        terminated = bool(
            self.state_termination_bound is not None
            and np.any(np.abs(self.state) > self.state_termination_bound)
        )

        info = {
            "cost": float(cost),
            "true_cost": float(true_cost),
            "state_cost": float(state_cost),
            "action_cost": float(action_cost),
            "quadratic_action_cost": float(quadratic_action_cost),
            "A": self.A.copy(),
            "B": self.B.copy(),
            "true_params": self.get_true_params_dict(),
        }
        return self.state.astype(np.float32), float(reward), terminated, truncated, info


class StoppingCarEnv(gym.Env):
    """
    Friction-limited obstacle stopping task.

    Observation x = [velocity, distance_to_obstacle, obstacle_visible].
    The unknown parameter is tire-road friction mu. Small braking actions are
    below the low-friction saturation limit and are therefore uninformative;
    strong braking reveals mu through the realized deceleration.
    """

    metadata = {"render_modes": []}

    PARAM_ORDER = ("mu", "process_noise_scale")
    PARAM_DENOMS = {"mu": 1.0, "process_noise_scale": 1.0}

    def __init__(
        self,
        dt: float = 0.05,
        mu: float = 0.5,
        process_noise_std: float | Sequence[float] = 0.0,
        initial_velocity_low: float = 2.4,
        initial_velocity_high: float = 2.6,
        action_low: float | Sequence[float] = -1.0,
        action_high: float | Sequence[float] = 1.0,
        max_episode_steps: int = 200,
        v_ref: float = 2.5,
        sigma_v: float = 0.5,
        d_max: float = 5.0,
        event_time_mode: str = "uniform_once",
        event_time_low: int = 50,
        event_time_high: int = 180,
        visible_distance_low: float = 0.9,
        visible_distance_high: float = 1.2,
        start_visible: bool = False,
        g: float = 9.81,
        brake_accel_max: float = 8.0,
        throttle_accel_max: float = 2.0,
        saturate_throttle_by_friction: bool = False,
        speed_reward_weight: float = 1.0,
        action_cost_weight: float = 0.02,
        safety_cost_weight: float = 5.0,
        safety_alpha: float = 5.0,
        d_buffer: float = 0.15,
        v_safe: float = 0.1,
        crash_penalty: float = 50.0,
        crash_remaining_penalty: float = 50.0,
        success_bonus: float = 5.0,
        terminate_on_success: bool = True,
        terminate_on_crash: bool = True,
    ):
        super().__init__()

        self.state_dim = 3
        self.action_dim = 1
        self.velocity_obs_index = 0
        self.dt = float(dt)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        self.mu = float(mu)
        if self.mu <= 0.0:
            raise ValueError("mu must be positive.")

        noise = np.asarray(process_noise_std, dtype=np.float64)
        if noise.ndim == 0:
            self.process_noise_std = np.asarray([float(noise)], dtype=np.float64)
        else:
            self.process_noise_std = noise.reshape(-1)
            if self.process_noise_std.size not in {1, self.state_dim}:
                raise ValueError(
                    "process_noise_std must be scalar, length 1, or length 3 for StoppingCarEnv."
                )

        self.initial_velocity_low = float(initial_velocity_low)
        self.initial_velocity_high = float(initial_velocity_high)
        if self.initial_velocity_high < self.initial_velocity_low:
            raise ValueError("initial_velocity_high must be >= initial_velocity_low.")

        action_low_arr = _as_1d_array(action_low, self.action_dim, dtype=np.float32)
        action_high_arr = _as_1d_array(action_high, self.action_dim, dtype=np.float32)
        if np.any(action_high_arr <= action_low_arr):
            raise ValueError("action_high must be > action_low elementwise")
        self.action_space = gym.spaces.Box(
            low=action_low_arr,
            high=action_high_arr,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.asarray([0.0, -np.inf, 0.0], dtype=np.float32),
            high=np.asarray([np.inf, np.inf, 1.0], dtype=np.float32),
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        self.max_episode_steps = int(max_episode_steps)
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be > 0")
        self.v_ref = float(v_ref)
        self.sigma_v = float(sigma_v)
        if self.sigma_v <= 0.0:
            raise ValueError("sigma_v must be positive.")
        self.d_max = float(d_max)
        self.event_time_mode = str(event_time_mode)
        if self.event_time_mode not in {"uniform_once", "start_visible"}:
            raise ValueError("event_time_mode must be one of {'uniform_once', 'start_visible'}.")
        self.event_time_low = int(event_time_low)
        self.event_time_high = int(event_time_high)
        if self.event_time_low < 0:
            raise ValueError("event_time_low must be nonnegative.")
        if self.event_time_high < self.event_time_low:
            raise ValueError("event_time_high must be >= event_time_low.")
        self.visible_distance_low = float(visible_distance_low)
        self.visible_distance_high = float(visible_distance_high)
        if self.visible_distance_high < self.visible_distance_low:
            raise ValueError("visible_distance_high must be >= visible_distance_low.")
        self.start_visible = bool(start_visible)
        self.g = float(g)
        if self.g <= 0.0:
            raise ValueError("g must be positive.")
        self.brake_accel_max = float(brake_accel_max)
        self.throttle_accel_max = float(throttle_accel_max)
        if self.brake_accel_max <= 0.0 or self.throttle_accel_max <= 0.0:
            raise ValueError("brake_accel_max and throttle_accel_max must be positive.")
        self.saturate_throttle_by_friction = bool(saturate_throttle_by_friction)

        self.speed_reward_weight = float(speed_reward_weight)
        self.action_cost_weight = float(action_cost_weight)
        self.safety_cost_weight = float(safety_cost_weight)
        self.safety_alpha = float(safety_alpha)
        if self.safety_alpha <= 0.0:
            raise ValueError("safety_alpha must be positive.")
        self.d_buffer = float(d_buffer)
        self.v_safe = float(v_safe)
        self.crash_penalty = float(crash_penalty)
        self.crash_remaining_penalty = float(crash_remaining_penalty)
        self.success_bonus = float(success_bonus)
        self.terminate_on_success = bool(terminate_on_success)
        self.terminate_on_crash = bool(terminate_on_crash)

        self.process_noise_scale = 1.0
        self.theta = self.mu
        self.velocity = 0.0
        self.distance = self.d_max
        self.obstacle_visible = False
        self._step_count = 0
        self._has_crashed = False
        self._success = False
        self._scheduled_event_step: int | None = None
        self.state = np.zeros(self.state_dim, dtype=np.float64)

    def set_friction(self, mu: float, process_noise_scale: float = 1.0) -> None:
        self.mu = float(mu)
        if self.mu <= 0.0:
            raise ValueError(f"StoppingCarEnv requires positive friction mu, got {self.mu}.")
        self.theta = self.mu
        self.process_noise_scale = float(process_noise_scale)

    def set_dynamics_scales(self, theta: float, process_noise_scale: float = 1.0) -> None:
        # Legacy wrapper entry point: in this task theta is interpreted as mu.
        self.set_friction(theta, process_noise_scale=process_noise_scale)

    def get_dynamics_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        a = np.eye(self.state_dim, dtype=np.float64)
        b = np.asarray([[self.dt * self.brake_accel_max], [0.0], [0.0]], dtype=np.float64)
        return a, b

    def get_true_params_dict(self) -> dict[str, float]:
        return {
            "mu": float(self.mu),
            "theta": float(self.theta),
            "process_noise_scale": float(self.process_noise_scale),
        }

    def get_param_denom_dict(self) -> dict[str, float]:
        denoms = dict(self.PARAM_DENOMS)
        denoms["theta"] = 1.0
        return denoms

    def get_true_params(self, param_names: Sequence[str]) -> np.ndarray:
        params = self.get_true_params_dict()
        return np.asarray([params[name] for name in param_names], dtype=np.float32)

    def get_param_denoms(self, param_names: Sequence[str]) -> np.ndarray:
        denoms = self.get_param_denom_dict()
        return np.asarray([denoms[name] for name in param_names], dtype=np.float32)

    def _obs(self) -> np.ndarray:
        self.state = np.asarray(
            [self.velocity, self.distance, 1.0 if self.obstacle_visible else 0.0],
            dtype=np.float64,
        )
        return self.state.astype(np.float32)

    def _sample_visible_distance(self) -> float:
        return float(self.np_random.uniform(self.visible_distance_low, self.visible_distance_high))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._step_count = 0
        self._has_crashed = False
        self._success = False
        self.velocity = float(self.np_random.uniform(self.initial_velocity_low, self.initial_velocity_high))
        start_visible = bool(self.start_visible or self.event_time_mode == "start_visible")
        self.obstacle_visible = start_visible
        self.distance = self._sample_visible_distance() if start_visible else self.d_max
        if start_visible:
            self._scheduled_event_step = 0
        else:
            high = min(self.event_time_high, self.max_episode_steps - 1)
            low = min(self.event_time_low, high)
            self._scheduled_event_step = int(self.np_random.integers(low, high + 1))

        info = {
            "true_params": self.get_true_params_dict(),
            "mu": float(self.mu),
            "event_time": float(self._scheduled_event_step),
            "obstacle_visible": float(self.obstacle_visible),
            "success": 0.0,
            "crashed": 0.0,
        }
        return self._obs(), info

    def _velocity_noise(self) -> float:
        std = float(self.process_noise_std[0]) if self.process_noise_std.size > 0 else 0.0
        return float(self.np_random.normal(0.0, std * self.process_noise_scale))

    def _commanded_accel(self, u: float) -> float:
        return float(self.brake_accel_max * u if u < 0.0 else self.throttle_accel_max * u)

    def realized_accel(self, u: float) -> float:
        a_cmd = self._commanded_accel(float(u))
        if a_cmd < 0.0:
            return float(max(a_cmd, -self.mu * self.g))
        if self.saturate_throttle_by_friction:
            return float(min(a_cmd, self.mu * self.g))
        return float(a_cmd)

    def stopping_distance(self, velocity: Optional[float] = None, mu: Optional[float] = None) -> float:
        v = self.velocity if velocity is None else float(velocity)
        friction = self.mu if mu is None else float(mu)
        return float(max(v, 0.0) ** 2 / max(2.0 * friction * self.g, 1e-8))

    def _speed_reward(self, velocity: float) -> float:
        err = (float(velocity) - self.v_ref) / self.sigma_v
        return float(self.speed_reward_weight * math.exp(-0.5 * err * err))

    def _safety_cost(self, velocity: float, distance: float) -> float:
        margin = self.stopping_distance(velocity=velocity) + self.d_buffer - float(distance)
        softplus = math.log1p(math.exp(self.safety_alpha * margin)) / self.safety_alpha
        return float(self.safety_cost_weight * softplus)

    def step(self, action: np.ndarray):
        u_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if u_arr.shape[0] != self.action_dim:
            raise ValueError(f"Expected action dimension {self.action_dim}, got {u_arr.shape[0]}")
        u_arr = np.clip(u_arr, self.action_space.low, self.action_space.high)
        requested_action = float(u_arr[0])

        visible_before = bool(self.obstacle_visible)
        distance_before = float(self.distance)
        velocity_before = float(self.velocity)
        accel = 0.0
        if not self._has_crashed and not self._success:
            accel = self.realized_accel(requested_action)
            next_velocity = max(0.0, velocity_before + self.dt * accel + self._velocity_noise())
            if visible_before:
                avg_velocity = 0.5 * (velocity_before + next_velocity)
                next_distance = distance_before - self.dt * avg_velocity
            else:
                next_distance = self.d_max
        else:
            next_velocity = 0.0
            next_distance = distance_before

        speed_reward = self._speed_reward(next_velocity) if not visible_before else 0.0
        action_cost = float(self.action_cost_weight * requested_action * requested_action)
        safety_cost = self._safety_cost(next_velocity, next_distance) if visible_before else 0.0

        newly_crashed = bool(visible_before and next_distance <= 0.0 and next_velocity > self.v_safe)
        success = bool(visible_before and next_velocity <= self.v_safe and next_distance > 0.0)
        if newly_crashed:
            self._has_crashed = True
        if success:
            self._success = True

        remaining_fraction = max(0.0, (self.max_episode_steps - self._step_count) / max(self.max_episode_steps, 1))
        crash_cost = (
            float(self.crash_penalty + self.crash_remaining_penalty * remaining_fraction)
            if newly_crashed
            else 0.0
        )
        success_reward = float(self.success_bonus if success else 0.0)
        reward = float(speed_reward + success_reward - action_cost - safety_cost - crash_cost)

        self.velocity = float(next_velocity)
        self.distance = float(next_distance)
        self._step_count += 1

        event_started = False
        if not self.obstacle_visible and self._scheduled_event_step is not None:
            if self._step_count >= self._scheduled_event_step:
                self.obstacle_visible = True
                self.distance = self._sample_visible_distance()
                event_started = True

        terminated = bool((newly_crashed and self.terminate_on_crash) or (success and self.terminate_on_success))
        truncated = bool(self._step_count >= self.max_episode_steps)
        true_cost = float(action_cost + safety_cost + crash_cost - speed_reward - success_reward)
        info = {
            "reward": float(reward),
            "cost": float(-reward),
            "true_cost": float(true_cost),
            "speed_reward": float(speed_reward),
            "success_reward": float(success_reward),
            "action_cost": float(action_cost),
            "safety_cost": float(safety_cost),
            "crash_cost": float(crash_cost),
            "state_cost": float(safety_cost + crash_cost - speed_reward - success_reward),
            "requested_action": float(requested_action),
            "commanded_accel": float(self._commanded_accel(requested_action)),
            "realized_accel": float(accel),
            "velocity": float(self.velocity),
            "distance": float(self.distance),
            "obstacle_visible": float(self.obstacle_visible),
            "event_started": float(event_started),
            "event_time": float(self._scheduled_event_step) if self._scheduled_event_step is not None else float("nan"),
            "pre_obstacle": float(not visible_before),
            "strong_probe": float((not visible_before) and requested_action <= -0.8),
            "brake_saturated": float(self._commanded_accel(requested_action) < -self.mu * self.g),
            "throttle_saturated": float(
                self.saturate_throttle_by_friction
                and self._commanded_accel(requested_action) > self.mu * self.g
            ),
            "accel_saturated": float(abs(accel - self._commanded_accel(requested_action)) > 1e-9),
            "stopping_distance": float(self.stopping_distance()),
            "stopping_margin": float(self.stopping_distance() + self.d_buffer - self.distance),
            "crashed": float(self._has_crashed),
            "newly_crashed": float(newly_crashed),
            "success": float(success),
            "mu": float(self.mu),
            "true_params": self.get_true_params_dict(),
        }
        return self._obs(), float(reward), terminated, truncated, info


class EmergencyBrakeEnv(gym.Env):
    """
    One-dimensional emergency braking task with unknown brake effectiveness.

    Observation x = [velocity, emergency_age / deadline_steps]. While
    emergency_age is zero, the car should cruise near v_ref. Once an emergency
    starts, emergency_age counts up from one and the car must reach
    |velocity| <= safe_speed within deadline_steps actions. Positive action is
    braking, with effectiveness m = m_nominal + theta.
    """

    metadata = {"render_modes": []}

    PARAM_ORDER = ("theta", "process_noise_scale")
    PARAM_DENOMS = {"theta": 1.0, "process_noise_scale": 1.0}

    def __init__(
        self,
        dt: float = 0.1,
        m_nominal: float = 1.0,
        process_noise_std: float | Sequence[float] = 0.0,
        initial_velocity_low: float = -1.1,
        initial_velocity_high: float = -0.9,
        action_low: float | Sequence[float] = -10.0,
        action_high: float | Sequence[float] = 10.0,
        max_episode_steps: int = 120,
        v_ref: float = -1.0,
        event_probability: float = 0.03,
        min_event_step: int = 20,
        event_time_mode: str = "geometric",
        event_time_low: int = 20,
        event_time_high: int = 100,
        deadline_steps: int = 3,
        safe_speed: float = 0.05,
        cruise_cost_weight: float = 1.0,
        action_cost_weight: float = 10.0,
        action_cost_type: str = "sqrt_abs",
        action_cost_eps: float = 1e-4,
        emergency_velocity_cost_weight: float = 0.0,
        emergency_velocity_cost_power: float = 2.0,
        emergency_velocity_cost_scale: float = 1.0,
        failure_cost: float = 300.0,
        terminate_on_success: bool = True,
    ):
        super().__init__()

        self.state_dim = 2
        self.action_dim = 1
        self.velocity_obs_index = 0
        self.dt = float(dt)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        self.m_nominal = float(m_nominal)
        if self.m_nominal <= 0.0:
            raise ValueError("m_nominal must be positive.")

        noise = np.asarray(process_noise_std, dtype=np.float64)
        if noise.ndim == 0:
            self.process_noise_std = np.asarray([float(noise)], dtype=np.float64)
        else:
            self.process_noise_std = noise.reshape(-1)
            if self.process_noise_std.size not in {1, self.state_dim}:
                raise ValueError(
                    "process_noise_std must be scalar, length 1, or length 2 for EmergencyBrakeEnv."
                )

        self.initial_velocity_low = float(initial_velocity_low)
        self.initial_velocity_high = float(initial_velocity_high)
        if self.initial_velocity_high < self.initial_velocity_low:
            raise ValueError("initial_velocity_high must be >= initial_velocity_low.")

        action_low_arr = _as_1d_array(action_low, self.action_dim, dtype=np.float32)
        action_high_arr = _as_1d_array(action_high, self.action_dim, dtype=np.float32)
        if np.any(action_high_arr <= action_low_arr):
            raise ValueError("action_high must be > action_low elementwise")
        self.action_space = gym.spaces.Box(
            low=action_low_arr,
            high=action_high_arr,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        self.max_episode_steps = int(max_episode_steps)
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be > 0")
        self.v_ref = float(v_ref)
        self.event_probability = float(event_probability)
        if not 0.0 <= self.event_probability <= 1.0:
            raise ValueError("event_probability must be in [0, 1].")
        self.min_event_step = int(min_event_step)
        if self.min_event_step < 0:
            raise ValueError("min_event_step must be nonnegative.")
        self.event_time_mode = str(event_time_mode)
        if self.event_time_mode not in {"geometric", "uniform_once"}:
            raise ValueError(
                f"event_time_mode must be 'geometric' or 'uniform_once', got {self.event_time_mode!r}."
            )
        self.event_time_low = int(event_time_low)
        self.event_time_high = int(event_time_high)
        if self.event_time_low < 1:
            raise ValueError("event_time_low must be >= 1.")
        if self.event_time_high < self.event_time_low:
            raise ValueError("event_time_high must be >= event_time_low.")
        self.deadline_steps = int(deadline_steps)
        if self.deadline_steps <= 0:
            raise ValueError("deadline_steps must be positive.")
        self.safe_speed = float(safe_speed)
        if self.safe_speed < 0.0:
            raise ValueError("safe_speed must be nonnegative.")

        self.cruise_cost_weight = float(cruise_cost_weight)
        self.action_cost_weight = float(action_cost_weight)
        self.action_cost_type = str(action_cost_type)
        if self.action_cost_type not in {"quadratic", "abs", "sqrt_abs"}:
            raise ValueError(
                "action_cost_type must be one of {'quadratic', 'abs', 'sqrt_abs'}, "
                f"got {self.action_cost_type!r}."
            )
        self.action_cost_eps = float(action_cost_eps)
        if self.action_cost_eps < 0.0:
            raise ValueError("action_cost_eps must be nonnegative.")
        self.emergency_velocity_cost_weight = float(emergency_velocity_cost_weight)
        if self.emergency_velocity_cost_weight < 0.0:
            raise ValueError("emergency_velocity_cost_weight must be nonnegative.")
        self.emergency_velocity_cost_power = float(emergency_velocity_cost_power)
        if self.emergency_velocity_cost_power <= 0.0:
            raise ValueError("emergency_velocity_cost_power must be positive.")
        self.emergency_velocity_cost_scale = float(emergency_velocity_cost_scale)
        if self.emergency_velocity_cost_scale <= 0.0:
            raise ValueError("emergency_velocity_cost_scale must be positive.")
        self.failure_cost_value = float(failure_cost)
        self.terminate_on_success = bool(terminate_on_success)

        self.theta = 0.0
        self.process_noise_scale = 1.0
        self.m = self.m_nominal
        self.velocity = 0.0
        self.emergency_age = 0
        self._step_count = 0
        self._has_failed = False
        self._event_started = False
        self._scheduled_event_step: int | None = None

    def _update_dynamics(self) -> None:
        self.m = self.m_nominal + float(self.theta)
        if self.m <= 0.0:
            raise ValueError(
                f"EmergencyBrakeEnv requires positive braking effectiveness m, got m={self.m}."
            )

    def set_dynamics_scales(self, theta: float, process_noise_scale: float = 1.0) -> None:
        self.theta = float(theta)
        self.process_noise_scale = float(process_noise_scale)
        self._update_dynamics()

    def get_dynamics_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        b = np.array([[self.dt * self.m], [0.0]], dtype=np.float64)
        return a, b

    def get_true_params_dict(self) -> dict[str, float]:
        return {
            "theta": float(self.theta),
            "m": float(self.m),
            "process_noise_scale": float(self.process_noise_scale),
        }

    def get_param_denom_dict(self) -> dict[str, float]:
        return dict(self.PARAM_DENOMS)

    def get_true_params(self, param_names: Sequence[str]) -> np.ndarray:
        params = self.get_true_params_dict()
        return np.asarray([params[name] for name in param_names], dtype=np.float32)

    def get_param_denoms(self, param_names: Sequence[str]) -> np.ndarray:
        denoms = self.get_param_denom_dict()
        return np.asarray([denoms[name] for name in param_names], dtype=np.float32)

    def _obs(self) -> np.ndarray:
        age_scaled = float(self.emergency_age) / float(self.deadline_steps)
        return np.asarray([self.velocity, age_scaled], dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._step_count = 0
        self._has_failed = False
        self._event_started = False
        self.emergency_age = 0
        if self.event_time_mode == "uniform_once":
            high = min(self.event_time_high, self.max_episode_steps)
            low = min(self.event_time_low, high)
            self._scheduled_event_step = int(self.np_random.integers(low, high + 1))
        else:
            self._scheduled_event_step = None
        self.velocity = float(
            self.np_random.uniform(self.initial_velocity_low, self.initial_velocity_high)
        )
        info = {
            "true_params": self.get_true_params_dict(),
            "m": float(self.m),
            "emergency_age": float(self.emergency_age),
            "deadline_steps": float(self.deadline_steps),
            "event_started": 0.0,
            "event_time": float(self._scheduled_event_step) if self._scheduled_event_step is not None else float("nan"),
            "failed": 0.0,
            "success": 0.0,
        }
        return self._obs(), info

    def _velocity_noise(self) -> float:
        std_raw = self.process_noise_std
        if std_raw.size == 0:
            vel_std = 0.0
        else:
            vel_std = float(std_raw[0])
        return float(self.np_random.normal(0.0, vel_std * self.process_noise_scale))

    def _action_cost(self, brake: float) -> float:
        abs_brake = abs(brake)
        if self.action_cost_type == "quadratic":
            shape = brake * brake
        elif self.action_cost_type == "abs":
            shape = abs_brake
        else:
            shape = math.sqrt(abs_brake + self.action_cost_eps) - math.sqrt(self.action_cost_eps)
        return float(self.action_cost_weight * shape)

    def _emergency_velocity_cost(self, velocity: float) -> float:
        scaled = abs(float(velocity)) / self.emergency_velocity_cost_scale
        return float(self.emergency_velocity_cost_weight * scaled**self.emergency_velocity_cost_power)

    def step(self, action: np.ndarray):
        u = np.asarray(action, dtype=np.float64).reshape(-1)
        if u.shape[0] != self.action_dim:
            raise ValueError(f"Expected action dimension {self.action_dim}, got {u.shape[0]}")
        u = np.clip(u, self.action_space.low, self.action_space.high)
        requested_brake = float(u[0])

        age_before = int(self.emergency_age)
        already_failed = bool(self._has_failed)
        if already_failed:
            brake = 0.0
            next_velocity = 0.0
        else:
            brake = requested_brake
            next_velocity = self.velocity + self.dt * self.m * brake + self._velocity_noise()
        post_action_velocity = float(next_velocity)

        success = False
        newly_failed = False
        if age_before > 0 and not already_failed:
            success = bool(abs(post_action_velocity) <= self.safe_speed)
            if (not success) and age_before >= self.deadline_steps:
                newly_failed = True
                self._has_failed = True
                next_velocity = 0.0

        self.velocity = float(next_velocity)

        # The event starts for the next decision, so the policy observes it
        # before the first emergency action is required.
        if age_before > 0:
            self.emergency_age = age_before + 1
        else:
            starts_now = False
            next_step = self._step_count + 1
            if self.event_time_mode == "uniform_once":
                starts_now = (
                    self._scheduled_event_step is not None
                    and next_step >= self._scheduled_event_step
                )
            else:
                starts_now = (
                    next_step >= self.min_event_step
                    and self.np_random.random() < self.event_probability
                )
            if starts_now:
                self.emergency_age = 1
                self._event_started = True
            else:
                self.emergency_age = 0

        action_cost = self._action_cost(brake)
        cruise_cost = 0.0
        if age_before == 0 and not already_failed:
            cruise_cost = float(self.cruise_cost_weight * (self.velocity - self.v_ref) ** 2)
        emergency_velocity_cost = 0.0
        if age_before > 0 and not already_failed:
            emergency_velocity_cost = self._emergency_velocity_cost(post_action_velocity)
        failure_cost = float(self.failure_cost_value if self._has_failed else 0.0)
        state_cost = float(cruise_cost + emergency_velocity_cost + failure_cost)
        true_cost = float(state_cost + action_cost)
        reward = -true_cost

        self._step_count += 1
        terminated = bool(success and self.terminate_on_success)
        truncated = bool(self._step_count >= self.max_episode_steps)
        info = {
            "cost": float(true_cost),
            "true_cost": float(true_cost),
            "normal_cost": float(cruise_cost + emergency_velocity_cost + action_cost),
            "state_cost": float(state_cost),
            "cruise_cost": float(cruise_cost),
            "emergency_velocity_cost": float(emergency_velocity_cost),
            "action_cost": float(action_cost),
            "failure_cost": float(failure_cost),
            "crash_cost": float(failure_cost),
            "crashed": float(self._has_failed),
            "newly_crashed": float(newly_failed),
            "failed": float(self._has_failed),
            "newly_failed": float(newly_failed),
            "success": float(success),
            "event_started": float(self._event_started),
            "event_time": float(self._scheduled_event_step) if self._scheduled_event_step is not None else float("nan"),
            "emergency_active": float(age_before > 0),
            "emergency_age": float(self.emergency_age),
            "deadline_steps": float(self.deadline_steps),
            "post_action_velocity": float(post_action_velocity),
            "requested_action": float(requested_brake),
            "m": float(self.m),
            "true_params": self.get_true_params_dict(),
        }
        return self._obs(), float(reward), terminated, truncated, info


class LQRDomainRandomizationAdapter:
    """
    Adapter used by existing DomainRandomizationWrapper.
    """

    PARAM_ORDER = LinearQuadraticEnv.PARAM_ORDER
    PARAM_DENOMS = LinearQuadraticEnv.PARAM_DENOMS

    def __init__(
        self,
        env: gym.Env,
        theta_mult_range: Sequence[float] = (-1.0, 1.0),
        mu_range: Sequence[float] = (0.25, 0.5, 0.7),
        a_range: Sequence[float] = (0.75, 1.05),
        b_range: Sequence[float] = (0.4, 1.8),
        process_noise_scale_mult_range: Sequence[float] = (1.0, 1.0),
        randomize_theta: bool = True,
        randomize_mu: bool = False,
        randomize_a: bool = False,
        randomize_b: bool = False,
        randomize_process_noise_scale: bool = True,
        categorical: bool = False,
        **_ignored_kwargs,
    ):
        self.env = self._find_base_env(env)
        if "binomial" in _ignored_kwargs:
            categorical = bool(_ignored_kwargs.pop("binomial")) or bool(categorical)
        _ignored_kwargs.pop("binomial_prob", None)

        self.categorical = bool(categorical)
        self.theta_mult_range = self._normalize_sampling_values("theta_mult_range", theta_mult_range)
        self.mu_range = self._normalize_sampling_values("mu_range", mu_range)
        self.a_range = self._normalize_sampling_values("a_range", a_range)
        self.b_range = self._normalize_sampling_values("b_range", b_range)
        self.process_noise_scale_mult_range = self._normalize_sampling_values(
            "process_noise_scale_mult_range",
            process_noise_scale_mult_range,
        )
        self.randomize_theta = bool(randomize_theta)
        self.randomize_mu = bool(randomize_mu)
        self.randomize_a = bool(randomize_a)
        self.randomize_b = bool(randomize_b)
        self.randomize_process_noise_scale = bool(randomize_process_noise_scale)
        active_param_groups = int(self.randomize_theta) + int(self.randomize_mu) + int(self.randomize_a or self.randomize_b)
        if active_param_groups > 1:
            raise ValueError("Choose only one parameter randomization family: theta, mu, or independent a/b.")
        if self.randomize_mu and not isinstance(self.env, StoppingCarEnv):
            raise ValueError("randomize_mu=True is only supported for StoppingCarEnv.")

    @staticmethod
    def _find_base_env(env: gym.Env) -> LinearQuadraticEnv | StoppingCarEnv | EmergencyBrakeEnv:
        e = env
        while True:
            if isinstance(e, (LinearQuadraticEnv, StoppingCarEnv, EmergencyBrakeEnv)):
                return e
            if not hasattr(e, "env"):
                break
            e = e.env
        raise TypeError(
            "LQRDomainRandomizationAdapter expects a LinearQuadraticEnv, StoppingCarEnv, "
            "or EmergencyBrakeEnv "
            "in the wrapper stack."
        )

    def _normalize_sampling_values(self, name: str, values: Sequence[float]) -> tuple[float, ...]:
        normalized = tuple(float(x) for x in values)
        if not normalized:
            raise ValueError(f"{name} must contain at least one value.")
        if self.categorical:
            return normalized
        if len(normalized) != 2:
            raise ValueError(
                f"{name} must contain exactly two values when categorical=False, got {len(normalized)}."
            )
        lo, hi = normalized
        if lo > hi:
            raise ValueError(f"Invalid range for {name}: low={lo} > high={hi}")
        return normalized

    def _sample_value(self, rng: np.random.Generator, values: tuple[float, ...]) -> float:
        if self.categorical:
            return float(rng.choice(np.asarray(values, dtype=np.float64)))
        lo, hi = values
        return float(rng.uniform(lo, hi))

    def randomize(self, rng: np.random.Generator) -> None:
        theta = getattr(self.env, "theta", 0.0)
        noise_scale = self.env.process_noise_scale

        if self.randomize_a or self.randomize_b:
            if not isinstance(self.env, LinearQuadraticEnv):
                raise ValueError("Independent a/b randomization is only supported for LinearQuadraticEnv.")
            a = self.env.a
            b = self.env.b
            if self.randomize_a:
                a = self._sample_value(rng, self.a_range)
            if self.randomize_b:
                b = self._sample_value(rng, self.b_range)
            if self.randomize_process_noise_scale:
                noise_scale = self._sample_value(rng, self.process_noise_scale_mult_range)
            self.env.set_scalar_ab_params(a=a, b=b, process_noise_scale=noise_scale)
        elif self.randomize_mu:
            mu = self._sample_value(rng, self.mu_range)
            if self.randomize_process_noise_scale:
                noise_scale = self._sample_value(rng, self.process_noise_scale_mult_range)
            self.env.set_friction(mu=mu, process_noise_scale=noise_scale)
        elif self.randomize_theta:
            theta = self._sample_value(rng, self.theta_mult_range)
            if self.randomize_process_noise_scale:
                noise_scale = self._sample_value(rng, self.process_noise_scale_mult_range)
            self.env.set_dynamics_scales(theta=theta, process_noise_scale=noise_scale)
        elif self.randomize_process_noise_scale:
            noise_scale = self._sample_value(rng, self.process_noise_scale_mult_range)
            self.env.set_dynamics_scales(theta=theta, process_noise_scale=noise_scale)

    def get_true_params_dict(self) -> dict[str, float]:
        return self.env.get_true_params_dict()

    def get_param_denom_dict(self) -> dict[str, float]:
        return self.env.get_param_denom_dict()

    def get_true_params(self, param_names: Sequence[str]) -> np.ndarray:
        return self.env.get_true_params(param_names)

    def get_param_denoms(self, param_names: Sequence[str]) -> np.ndarray:
        return self.env.get_param_denoms(param_names)

    def get_current_params_normalized(self) -> np.ndarray:
        params = self.get_true_params_dict()
        return np.asarray(
            [
                params.get("theta", params.get("mu", 0.0)),
                params["process_noise_scale"],
                params.get("a", 0.0),
                params.get("b", 0.0),
                params.get("mu", 0.0),
            ],
            dtype=np.float32,
        )


def install_lqr_adapter_for_domain_randomization() -> None:
    """
    Patch wrappers.DomainRandomizationWrapper to use LQR adapter for LQR env runs.

    DomainRandomizationWrapper currently instantiates `CartPoleSwingUpAdapter`.
    We swap that symbol to point to `LQRDomainRandomizationAdapter`.
    """

    import wrappers

    wrappers.CartPoleSwingUpAdapter = LQRDomainRandomizationAdapter
