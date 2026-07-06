from typing import Sequence

import gymnasium as gym
import numpy as np
import mujoco


class ForceMujocoFixedCamera(gym.Wrapper):
    """ 
    Force a fixed MuJoCo camera when using shimmy dm_control with render_mode="human".
    """

    def __init__(self, env, camera_id: int = 0, width: int | None = None, height: int | None = None):
        super().__init__(env)
        self.camera_id = int(camera_id)
        self.width = None if width is None else int(width)
        self.height = None if height is None else int(height)

    def _apply(self) -> None:
        core = self.env.unwrapped

        # checking whether rendering is turned on
        if not hasattr(core, "viewer"):
            return

        # define width and height of the viewer window (if specified)
        if self.width is not None:
            core.viewer.width = self.width
        if self.height is not None:
            core.viewer.height = self.height

        # force fixed camera
        core.viewer.default_cam_config = {
            "type": mujoco.mjtCamera.mjCAMERA_FIXED,
            "fixedcamid": self.camera_id,
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._apply()
        return obs, info


# costumize environment reward function
class RewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Ensure reward shaping stays scalar even when `action` is a numpy array (e.g. shape (1,)).
        a = float(np.asarray(action).ravel()[0])
        # reward = 1.0 - (0.1 * abs(x) + 1.0 * abs(theta) - 0.01 * abs(action))
        # reward = 1.0 if (abs(x) < 0.25 and abs(theta) < 0.35) else 0.0
        reward = float(reward) - 0.1 * abs(a)
        return obs, reward, terminated, truncated, info


class PreviousActionObservationWrapper(gym.Wrapper):
    """Append the previous action to the current observation for recurrent policies."""

    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("PreviousActionObservationWrapper requires a Box observation space.")
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("PreviousActionObservationWrapper requires a Box action space.")

        self._obs_dim = int(np.prod(env.observation_space.shape))
        self._act_dim = int(np.prod(env.action_space.shape))
        self._last_base_obs = np.zeros((self._obs_dim,), dtype=np.float32)
        self._prev_action = np.zeros((self._act_dim,), dtype=np.float32)

        obs_low = np.asarray(env.observation_space.low, dtype=np.float32).reshape(-1)
        obs_high = np.asarray(env.observation_space.high, dtype=np.float32).reshape(-1)
        act_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
        act_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([obs_low, act_low], axis=0),
            high=np.concatenate([obs_high, act_high], axis=0),
            dtype=np.float32,
        )

    def _augment(self, obs) -> np.ndarray:
        self._last_base_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        return np.concatenate([self._last_base_obs, self._prev_action], axis=0).astype(np.float32, copy=False)

    def get_current_control_observation(self, _obs=None) -> np.ndarray:
        return self._last_base_obs.copy()

    def lqr_action(self, obs=None) -> np.ndarray:
        base_obs = self._last_base_obs if obs is None else np.asarray(obs, dtype=np.float32).reshape(-1)[: self._obs_dim]
        return self.env.lqr_action(base_obs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action.fill(0.0)
        return self._augment(obs), info

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        self._prev_action = np.clip(
            action_arr,
            self.env.action_space.low.reshape(-1),
            self.env.action_space.high.reshape(-1),
        ).astype(np.float32, copy=False)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info


# CartPole-specific parameter access and mutation for the dm_control swing-up task.
class CartPoleSwingUpAdapter:
    PARAM_ORDER = ("cart_mass", "pole_mass", "pole_length", "actuator_scale", "cart_friction")
    PARAM_DENOMS = {
        "cart_mass": 1.0,
        "pole_mass": 0.1,
        "pole_length": 1.0,
        "actuator_scale": 10.0,
        "cart_friction": 5e-4,
    }

    def __init__(
        self,
        env,
        cart_body_name: str = "cart",
        pole_body_name: str = "pole_1",
        pole_geom_name: str = "pole_1",
        actuator_name: str = "slide",
        cart_mass_mult_range: Sequence[float] = (0.2, 3.0),
        pole_mass_mult_range: Sequence[float] = (0.2, 3.0),
        pole_length_mult_range: Sequence[float] = (0.2, 3.0),
        actuator_scale_mult_range: Sequence[float] = (0.2, 3.0),
        cart_friction_mult_range: Sequence[float] = (0.2, 3.0),
        randomize_cart_mass: bool = True,
        randomize_pole_mass: bool = True,
        randomize_pole_length: bool = True,
        randomize_actuator_scale: bool = True,
        randomize_cart_friction: bool = False,
        categorical: bool = False,
        **_ignored_kwargs,
    ):
        self.env = env
        self.cart_body_name = cart_body_name
        self.pole_body_name = pole_body_name
        self.pole_geom_name = pole_geom_name
        self.actuator_name = actuator_name

        if "binomial" in _ignored_kwargs:
            categorical = bool(_ignored_kwargs.pop("binomial")) or bool(categorical)
        _ignored_kwargs.pop("binomial_prob", None)

        self.categorical = bool(categorical)
        self.cart_mass_mult_range = self._normalize_sampling_values("cart_mass_mult_range", cart_mass_mult_range)
        self.pole_mass_mult_range = self._normalize_sampling_values("pole_mass_mult_range", pole_mass_mult_range)
        self.pole_length_mult_range = self._normalize_sampling_values("pole_length_mult_range", pole_length_mult_range)
        self.actuator_scale_mult_range = self._normalize_sampling_values("actuator_scale_mult_range", actuator_scale_mult_range)
        self.cart_friction_mult_range = self._normalize_sampling_values("cart_friction_mult_range", cart_friction_mult_range)
        self.randomize_cart_mass = bool(randomize_cart_mass)
        self.randomize_pole_mass = bool(randomize_pole_mass)
        self.randomize_pole_length = bool(randomize_pole_length)
        self.randomize_actuator_scale = bool(randomize_actuator_scale)
        self.randomize_cart_friction = bool(randomize_cart_friction)

        # initialize component ids + original values.
        self._physics = None
        self._model = None
        self.cart_id = None
        self.pole_id = None
        self.pole_geom_id = None
        self.act_id = None
        self.slider_jnt_id = None
        self.slider_dof_id = None
        self._orig_cart_mass = None
        self._orig_pole_mass = None
        self._orig_pole_half_len = None
        self._orig_pole_geom_pos = None
        self._pole_axis_body = None
        self._pole_anchor_pos = None
        self._pole_anchor_sign = None
        self._orig_pole_radius = None
        self._orig_pole_ipos = None
        self._orig_pole_inertia = None
        self._orig_pole_iquat = None
        self._orig_gear = None
        self._orig_cart_friction = None

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

    def _sample_multiplier(self, rng: np.random.Generator, values: tuple[float, ...]) -> float:
        if self.categorical:
            return float(rng.choice(np.asarray(values, dtype=np.float64)))
        lo, hi = values
        return float(rng.uniform(lo, hi))

    def randomize(self, rng: np.random.Generator) -> None:
        self._maybe_init_dmcontrol_handles()
        m = self._model

        if self.randomize_cart_mass:
            scale = self._sample_multiplier(rng, self.cart_mass_mult_range)
            m.body_mass[self.cart_id] = scale * self._orig_cart_mass


        if self.randomize_pole_mass:
            scale = self._sample_multiplier(rng, self.pole_mass_mult_range)
            m.body_mass[self.pole_id] = scale * self._orig_pole_mass


        new_half_len = float(m.geom_size[self.pole_geom_id][1])
        if self.randomize_pole_length:
            scale = self._sample_multiplier(rng, self.pole_length_mult_range)
            new_half_len = scale * self._orig_pole_half_len

                
        m.geom_size[self.pole_geom_id][1] = new_half_len
        new_pos = self._pole_anchor_pos - (self._pole_anchor_sign * self._pole_axis_body * new_half_len)
        m.geom_pos[self.pole_geom_id] = new_pos

        # Update inertia so length changes affect physics.
        pole_mass = float(m.body_mass[self.pole_id])
        r = float(self._orig_pole_radius)
        L = 2.0 * float(new_half_len)
        I_perp = (pole_mass / 12.0) * (3.0 * r * r + L * L)
        I_axis = 0.5 * pole_mass * r * r
        m.body_ipos[self.pole_id] = new_pos
        m.body_inertia[self.pole_id] = np.array([I_perp, I_perp, I_axis], dtype=np.float64)

        if self.randomize_actuator_scale:
            scale = self._sample_multiplier(rng, self.actuator_scale_mult_range)
            m.actuator_gear[self.act_id][0] = scale * self._orig_gear

        if self.randomize_cart_friction:
            scale = self._sample_multiplier(rng, self.cart_friction_mult_range)
            m.dof_damping[self.slider_dof_id] = scale * self._orig_cart_friction

        self._physics.forward()

    def get_true_params_dict(self) -> dict[str, float]:
        self._maybe_init_dmcontrol_handles()
        m = self._model
        return {
            "cart_mass": float(m.body_mass[self.cart_id]),
            "pole_mass": float(m.body_mass[self.pole_id]),
            "pole_length": 2.0 * float(m.geom_size[self.pole_geom_id][1]),
            "actuator_scale": float(m.actuator_gear[self.act_id][0]),
            "cart_friction": float(m.dof_damping[self.slider_dof_id]),
        }

    def get_param_denom_dict(self) -> dict[str, float]:
        return dict(self.PARAM_DENOMS)

    def get_true_params(self, param_names) -> np.ndarray:
        params = self.get_true_params_dict()
        return np.array([float(params[name]) for name in param_names], dtype=np.float32)

    def get_param_denoms(self, param_names) -> np.ndarray:
        denoms = self.get_param_denom_dict()
        return np.array([float(denoms[name]) for name in param_names], dtype=np.float32)

    def get_current_params_normalized(self) -> np.ndarray:
        """Return normalized (cart_mass, pole_mass, pole_length, actuator_scale, cart_friction)."""
        params = self.get_true_params_dict()
        return np.array(
            [
                params["cart_mass"] / max(self._orig_cart_mass, 1e-8),
                params["pole_mass"] / max(self._orig_pole_mass, 1e-8),
                params["pole_length"] / max(2.0 * self._orig_pole_half_len, 1e-8),
                params["actuator_scale"] / max(self._orig_gear, 1e-8),
                params["cart_friction"] / max(self._orig_cart_friction, 1e-8),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector v by quaternion q=(w,x,y,z)."""
        w, x, y, z = q
        q_vec = np.array([x, y, z], dtype=np.float64)
        v = v.astype(np.float64)
        t = 2.0 * np.cross(q_vec, v)
        return v + w * t + np.cross(q_vec, t)

    def _maybe_init_dmcontrol_handles(self) -> None:
        if self._model is not None:
            return

        e = self.env.unwrapped

        if not (hasattr(e, "_env") and hasattr(e._env, "physics")):
            raise AttributeError("CartPoleSwingUpAdapter expects a shimmy dm_control env with core._env.physics")

        self._physics = e._env.physics
        self._model = self._physics.model

        # store component IDs to access params later
        m = self._model
        self.cart_id = int(m.name2id(self.cart_body_name, "body"))
        self.pole_id = int(m.name2id(self.pole_body_name, "body"))
        self.pole_geom_id = int(m.name2id(self.pole_geom_name, "geom"))
        self.act_id = int(m.name2id(self.actuator_name, "actuator"))
        self.slider_jnt_id = int(m.name2id("slider", "joint"))
        self.slider_dof_id = int(m.jnt_dofadr[self.slider_jnt_id])

        # Store original values once
        self._orig_cart_mass = float(m.body_mass[self.cart_id])
        self._orig_pole_mass = float(m.body_mass[self.pole_id])

        # pole geom is size=[radius, half_length, 0]
        self._orig_pole_half_len = float(m.geom_size[self.pole_geom_id][1])
        self._orig_pole_radius = float(m.geom_size[self.pole_geom_id][0])

        # get position and orientation (!) of the pole
        self._orig_pole_geom_pos = np.array(m.geom_pos[self.pole_geom_id], dtype=np.float64)
        q = np.array(m.geom_quat[self.pole_geom_id], dtype=np.float64)
        axis = self._quat_rotate_wxyz(q, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            axis = axis / axis_norm
        self._pole_axis_body = axis

        # get hinge position
        pole_joint_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        jnt_adr = int(m.body_jntadr[self.pole_id])
        jnt_num = int(m.body_jntnum[self.pole_id])
        if jnt_num > 0:
            pole_joint_pos = np.array(m.jnt_pos[jnt_adr], dtype=np.float64)

        # Determine which end of the pole is anchored at the hinge by comparing distances to the geom center.
        half = float(self._orig_pole_half_len)
        end_plus = self._orig_pole_geom_pos + axis * half
        end_minus = self._orig_pole_geom_pos - axis * half
        if np.linalg.norm(end_plus - pole_joint_pos) <= np.linalg.norm(end_minus - pole_joint_pos):
            self._pole_anchor_pos = end_plus
            self._pole_anchor_sign = +1.0
        else:
            self._pole_anchor_pos = end_minus
            self._pole_anchor_sign = -1.0

        # store position of center of mass, principal moments of inertia and inertial frame orientation (quaternion)
        self._orig_pole_ipos = np.array(m.body_ipos[self.pole_id], dtype=np.float64)
        self._orig_pole_inertia = np.array(m.body_inertia[self.pole_id], dtype=np.float64)
        self._orig_pole_iquat = np.array(m.body_iquat[self.pole_id], dtype=np.float64)

        # store actuation scale
        self._orig_gear = float(m.actuator_gear[self.act_id][0])
        self._orig_cart_friction = float(m.dof_damping[self.slider_dof_id])


class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        change_prob=0.01,
        only_at_episode_end=True,
        randomize_on_reset: bool = True,
        rng_seed: int | None = None,
        reseed_on_reset: bool = False,
        **adapter_kwargs,
    ):
        super().__init__(env)
        self.change_prob = float(change_prob)
        self.only_at_episode_end = bool(only_at_episode_end)
        self.randomize_on_reset = bool(randomize_on_reset)
        self.reseed_on_reset = bool(reseed_on_reset)
        self._rng = np.random.default_rng(None if rng_seed is None else int(rng_seed))
        self.adapter = CartPoleSwingUpAdapter(env, **adapter_kwargs)

    def get_true_params_dict(self) -> dict[str, float]:
        return self.adapter.get_true_params_dict()

    def get_current_params_normalized(self) -> np.ndarray:
        return self.adapter.get_current_params_normalized()

    def get_param_denom_dict(self) -> dict[str, float]:
        return self.adapter.get_param_denom_dict()

    def get_true_params(self, param_names) -> np.ndarray:
        return self.adapter.get_true_params(param_names)

    def get_param_denoms(self, param_names) -> np.ndarray:
        return self.adapter.get_param_denoms(param_names)

    def _randomize_params(self) -> None:
        self.adapter.randomize(self._rng)

    def reset(self, **kwargs):
        # reseeding the random number generator is only used in evaluation
        # to ensure deterministic seeds in the evaluation runs
        if self.reseed_on_reset and "seed" in kwargs and kwargs["seed"] is not None:
            self._rng = np.random.default_rng(int(kwargs["seed"]))

        obs, info = self.env.reset(**kwargs)

        if self.randomize_on_reset:
            self._randomize_params()

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        episode_ended = bool(terminated or truncated)
        should_skip_terminal_randomization = episode_ended and self.randomize_on_reset

        if not should_skip_terminal_randomization and (not self.only_at_episode_end or episode_ended):
            if float(self._rng.random()) < self.change_prob:
                self._randomize_params()

        return obs, reward, terminated, truncated, info


# Backwards compatibility for saved configs and imports.
ChangingCartPoleDynamics = DomainRandomizationWrapper
