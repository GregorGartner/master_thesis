from __future__ import annotations

import json
import os
from pathlib import Path

import torch as th
import yaml

from scalar_ab_lqr_utils import (
    DEFAULT_RANGE,
    PRIOR_MEAN,
    PRIOR_VAR,
    SELECTION_ROOT,
    base_training_cfg,
    evaluate_controller_grid,
    run_training_stage,
    save_json,
    set_load,
    smoke_enabled,
    snapshot_phase_weights,
    timestamp,
)
from run_two_action_pipeline import CONFIG_PATH, _resolve_exp_root
from unified_context_ppo import UnifiedContextPPO


SEED = 1
PRIVILEGED_STEPS = 2_000_000
GRADUAL_PHASES = [
    ("P01", 0.25, 0.08, 3e-4, 1_000_000),
    ("P02", 0.50, 0.05, 3e-4, 1_000_000),
    ("P03", 0.75, 0.03, 2e-4, 1_000_000),
    ("P04", 1.00, 0.01, 1e-4, 1_500_000),
    ("P05", 1.00, 0.005, 1e-4, 1_500_000),
    ("P06", 1.00, 0.002, 1e-4, 1_000_000),
]


def _smoke() -> bool:
    return smoke_enabled("SCALAR_AB_BLR_PPO_SMOKE")


def _step_multiplier() -> float:
    if _smoke():
        return 1.0
    raw = os.environ.get("SCALAR_AB_BLR_PPO_STEP_MULT", os.environ.get("SCALAR_AB_STEP_MULT"))
    if raw is not None and raw != "":
        return float(raw)
    return 2.0 if os.environ.get("SCALAR_AB_LONG", "0").lower() in {"1", "true", "yes"} else 1.0


def _steps(steps: int) -> int:
    return 16_384 if _smoke() else int(round(float(steps) * _step_multiplier()))


def _checkpoint(exp_dir: Path, weights_name: str = "weights") -> Path:
    path = exp_dir / weights_name
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def _set_privileged(cfg: dict) -> None:
    params = cfg["model"]["params"]
    params["context_mode"] = "privileged"
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["condition_on_uncertainty"] = False
    params["privileged_uncertainty_mode"] = "zeros"
    params["privileged_context_probability"] = 0.0
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["learning_rate"] = 3e-4
    params["ent_coef"] = 0.01


def _set_blr(cfg: dict, *, condition_on_uncertainty: bool, blr_probability: float, ent: float, lr: float) -> None:
    params = cfg["model"]["params"]
    params["context_mode"] = "closed_form"
    params["closed_form_system"] = "scalar_ab_lqr"
    params["closed_form_prior_mean"] = list(PRIOR_MEAN)
    params["closed_form_prior_var"] = list(PRIOR_VAR)
    params["closed_form_obs_noise_var_floor"] = 1e-5
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["condition_on_uncertainty"] = bool(condition_on_uncertainty)
    params["uncertainty_context_dim"] = 3 if condition_on_uncertainty else 2
    params["privileged_uncertainty_mode"] = "zeros"
    params["privileged_context_probability"] = 1.0 - float(blr_probability)
    params["uncertainty_regularization_coef"] = 0.0
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["detach_context_for_rl"] = True
    params["naive_action_noise_std"] = 0.0
    params["learning_rate"] = float(lr)
    params["ent_coef"] = float(ent)


def _train_blr(base_cfg: dict, exp_root: Path, exp_name: str, exp_dir: Path, *, condition_on_uncertainty: bool) -> None:
    for phase_idx, (phase, blr_prob, ent, lr, steps) in enumerate(GRADUAL_PHASES):
        cfg = base_training_cfg(base_cfg, exp_root, exp_name, _steps(steps), DEFAULT_RANGE)
        _set_blr(cfg, condition_on_uncertainty=condition_on_uncertainty, blr_probability=blr_prob, ent=ent, lr=lr)
        set_load(cfg, exp_dir if phase_idx > 0 else None)
        run_training_stage(f"{exp_name}/{phase}", cfg)
        snapshot_phase_weights(exp_dir, phase)


def _validate_scalar_ab_blr_math() -> None:
    model = object.__new__(UnifiedContextPPO)
    model.regression_param_names = ["a", "b"]
    model._latent_dim = 2
    model._flat_observation_space = __import__("gymnasium").spaces.Box(-10.0, 10.0, shape=(1,))
    model.action_space = __import__("gymnasium").spaces.Box(-10.0, 10.0, shape=(1,))
    model._pair_traj_dim = 2
    model.transition_type = "s_next"
    model._closed_form_reg_eps = 1e-8
    model.closed_form_prior_mean = [0.0, 0.0]
    model.closed_form_prior_var = [1.0, 1.0]
    model.device = th.device("cpu")
    model._get_closed_form_noise_inv_diag = lambda env, n_envs, obs_dim, env_indices=None: th.ones((n_envs, obs_dim))
    model._ensure_true_param_denoms = lambda env: setattr(model, "_true_param_denoms", __import__("numpy").asarray([1.0, 1.0], dtype="float32"))
    model._resolve_closed_form_system = lambda env: "scalar_ab_lqr"
    informative = th.tensor([[[1.0, 0.0, 0.8], [0.5, 1.0, 1.6], [2.0, -1.0, 0.6]]])
    mean, cov = model._estimate_closed_form_scalar_ab_posterior(informative, env=None)
    z = th.tensor([[1.0, 0.0], [0.5, 1.0], [2.0, -1.0]])
    y = th.tensor([0.8, 1.6, 0.6])
    precision = th.eye(2) + z.T @ z
    expected_mean = th.linalg.solve(precision, z.T @ y)
    expected_cov = th.linalg.inv(precision)
    if not th.allclose(mean[0], expected_mean, atol=1e-6):
        raise RuntimeError("scalar_ab BLR posterior mean does not match the reference equation.")
    if not th.allclose(cov[0], expected_cov, atol=1e-6):
        raise RuntimeError("scalar_ab BLR posterior covariance does not match the reference equation.")
    rank_def = th.tensor([[[1.0, -0.5, 0.6], [2.0, -1.0, 1.2], [3.0, -1.5, 1.8]]])
    _, cov_rank_def = model._estimate_closed_form_scalar_ab_posterior(rank_def, env=None)
    eig = th.linalg.eigvalsh(cov_rank_def[0])
    if not bool((eig[-1] / eig[0]) > 5.0):
        raise RuntimeError("Rank-deficient closed-loop data did not produce elongated posterior covariance.")


def main() -> None:
    _validate_scalar_ab_blr_math()
    original_text = CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_text)
    exp_root = _resolve_exp_root(base_cfg)
    stamp = timestamp()
    suffix = "_smoke" if _smoke() else ""
    names = {
        "privileged": f"s_{stamp}_scalar_ab_privileged{suffix}",
        "blr_mean": f"s_{stamp}_scalar_ab_blr_mean{suffix}",
        "blr_mean_cov": f"s_{stamp}_scalar_ab_blr_mean_cov{suffix}",
    }
    dirs = {key: exp_root / value for key, value in names.items()}
    run_root = SELECTION_ROOT / f"blr_ppo_{stamp}_stable_wide_b{suffix}"
    manifest = {
        "timestamp": stamp,
        "smoke": _smoke(),
        "seed": SEED,
        "param_range": DEFAULT_RANGE,
        "step_multiplier": _step_multiplier(),
        "dirs": {key: str(value.resolve()) for key, value in dirs.items()},
        "phases": [
            {"phase": phase, "blr_probability": prob, "ent_coef": ent, "learning_rate": lr, "timesteps": _steps(steps)}
            for phase, prob, ent, lr, steps in GRADUAL_PHASES
        ],
    }
    save_json(run_root / "scalar_ab_blr_ppo.json", manifest)

    try:
        cfg = base_training_cfg(base_cfg, exp_root, names["privileged"], _steps(PRIVILEGED_STEPS), DEFAULT_RANGE)
        _set_privileged(cfg)
        run_training_stage("scalar_ab_privileged", cfg)
        _train_blr(base_cfg, exp_root, names["blr_mean"], dirs["blr_mean"], condition_on_uncertainty=False)
        _train_blr(base_cfg, exp_root, names["blr_mean_cov"], dirs["blr_mean_cov"], condition_on_uncertainty=True)
    finally:
        CONFIG_PATH.write_text(original_text)

    for key, phase in [("privileged", "weights"), ("blr_mean", "weights_P06"), ("blr_mean_cov", "weights_P06")]:
        path = _checkpoint(dirs[key], phase)
        if not path.exists():
            raise FileNotFoundError(f"Missing expected checkpoint: {path}")

    specs = [
        {"label": "privileged", "kind": "ppo", "experiment": str(dirs["privileged"]), "weights_name": "weights"},
        {"label": "blr_mean_P06", "kind": "ppo", "experiment": str(dirs["blr_mean"]), "weights_name": "weights_P06"},
        {"label": "blr_mean_cov_P06", "kind": "ppo", "experiment": str(dirs["blr_mean_cov"]), "weights_name": "weights_P06"},
        {"label": "oracle_lqr", "kind": "oracle_lqr"},
    ]
    compact_n = 3 if _smoke() else 31
    compact_seeds = 1 if _smoke() else 3
    final_n = 3 if _smoke() else 51
    final_seeds = 1 if _smoke() else 5
    compact = evaluate_controller_grid(run_root / "scalar_ab_grid_compact", specs, grid_n=compact_n, seeds_per_pair=compact_seeds)
    final = evaluate_controller_grid(run_root / "scalar_ab_grid_final", specs, grid_n=final_n, seeds_per_pair=final_seeds)
    manifest["sweeps"] = {"compact": str(compact.resolve()), "final": str(final.resolve())}
    manifest["validation"] = {
        "scalar_ab_blr_math": True,
        "checkpoints_produced": True,
        "full_cov_context_dim": 5,
        "scorecards_produced": True,
    }
    save_json(run_root / "scalar_ab_blr_ppo.json", manifest)
    if not _smoke():
        (SELECTION_ROOT / "latest_scalar_ab_blr_ppo.txt").write_text(str(run_root.resolve()) + "\n")
    print(f"Saved scalar-ab BLR+PPO run to: {run_root}", flush=True)


if __name__ == "__main__":
    main()
