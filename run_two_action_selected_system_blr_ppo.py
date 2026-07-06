from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch
import yaml

from run_two_action_pipeline import (
    CONFIG_PATH,
    _resolve_exp_root,
    _snapshot_phase_weights,
)
from run_two_action_selected_system_final_methods import (
    GRADUAL_PHASES,
    SEED,
    SELECTION_ROOT,
    SELECTED_SYSTEM_LABEL,
    _base_cfg,
    _checkpoint,
    _load_confirmation,
    _lqr_specs,
    _policy_group_digest,
    _ppo_spec,
    _require_checkpoint,
    _run_stage,
    _run_sweep,
    _set_load,
)
from run_two_action_system_neural_screening import _set_common
from unified_context_ppo import UnifiedContextPPO


PRIOR_MEAN = 0.0
PRIOR_VAR = 0.25**2 / 3.0


def _smoke_enabled() -> bool:
    return os.environ.get("TWO_ACTION_BLR_PPO_SMOKE", "0").lower() in {"1", "true", "yes"}


def _steps(steps: int) -> int:
    return 16_384 if _smoke_enabled() else steps


def _load_final_methods() -> tuple[Path, dict]:
    pointer = SELECTION_ROOT / "latest_final_methods.txt"
    if not pointer.exists():
        raise FileNotFoundError("No latest final-method pointer found. Run final selected-system methods first.")

    run_root = Path(pointer.read_text().strip()).resolve()
    manifest_path = run_root / "final_methods.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Final-method manifest is missing: {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if manifest["system"]["label"] != SELECTED_SYSTEM_LABEL or int(manifest["seed"]) != SEED:
        raise RuntimeError("Final-method manifest does not match the frozen selected system and seed.")
    for key in ["staged_policy", "gradual_mle", "gradual_nll"]:
        if key not in manifest["dirs"]:
            raise RuntimeError(f"Final-method manifest is missing experiment directory {key!r}.")
    _require_checkpoint(Path(manifest["dirs"]["staged_policy"]), "weights_S02")
    for key in ["gradual_mle", "gradual_nll"]:
        _require_checkpoint(Path(manifest["dirs"][key]), "weights_P05")
        _require_checkpoint(Path(manifest["dirs"][key]), "weights_P06")
    return run_root, manifest


def _set_blr(
    cfg: dict,
    *,
    condition_on_uncertainty: bool,
    blr_probability: float,
    ent: float,
    lr: float,
) -> None:
    _set_common(cfg, SEED)
    params = cfg["model"]["params"]
    params["context_mode"] = "closed_form"
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["condition_on_uncertainty"] = bool(condition_on_uncertainty)
    params["privileged_uncertainty_mode"] = "zeros"
    params["privileged_context_probability"] = 1.0 - float(blr_probability)
    params["closed_form_prior_mean"] = PRIOR_MEAN
    params["closed_form_prior_var"] = PRIOR_VAR
    params["uncertainty_regularization_coef"] = 0.0
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["uncertainty_penalty_metric"] = "std"
    params["detach_context_for_rl"] = True
    params["naive_action_noise_std"] = 0.0
    params["learning_rate"] = float(lr)
    params["ent_coef"] = float(ent)


def _train_blr(
    *,
    condition_on_uncertainty: bool,
    base_cfg: dict,
    exp_root: Path,
    system: dict,
    exp_name: str,
    exp_dir: Path,
) -> None:
    label = "mean_std" if condition_on_uncertainty else "mean"
    for phase_idx, (phase, blr_probability, ent, lr, steps) in enumerate(GRADUAL_PHASES):
        cfg = _base_cfg(base_cfg, exp_root, exp_name, _steps(steps), system)
        _set_blr(
            cfg,
            condition_on_uncertainty=condition_on_uncertainty,
            blr_probability=blr_probability,
            ent=ent,
            lr=lr,
        )
        _set_load(cfg, exp_dir if phase_idx > 0 else None)
        _run_stage(f"gradual_blr_{label}/{phase}", cfg)
        _snapshot_phase_weights(exp_dir, phase)


def _validate_blr_math() -> None:
    model = object.__new__(UnifiedContextPPO)
    model.regression_param_names = ["theta"]
    model._latent_dim = 1
    model._flat_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
    model._pair_traj_dim = 2
    model.transition_type = "s_next"
    model._closed_form_reg_eps = 1e-8
    model.closed_form_prior_mean = 0.1
    model.closed_form_prior_var = 0.25
    model.device = torch.device("cpu")
    model._get_closed_form_lqr_tensors = lambda env: {
        "A0": torch.tensor([[0.0]]),
        "B0": torch.tensor([[0.0]]),
        "dA": torch.tensor([[1.0]]),
        "dB": torch.tensor([[0.0]]),
    }
    model._get_closed_form_noise_inv_diag = lambda env, n_envs, obs_dim, env_indices=None: torch.full(
        (n_envs, obs_dim), 4.0
    )

    zero_traj = torch.zeros((1, 2, 3))
    zero_mean, zero_var = model._estimate_closed_form_theta_posterior(zero_traj, env=None)
    if not torch.allclose(zero_mean, torch.tensor([[0.1]]), atol=1e-7):
        raise RuntimeError("Zero-information BLR posterior did not return the configured prior mean.")
    if not torch.allclose(zero_var, torch.tensor([[0.25]]), atol=1e-7):
        raise RuntimeError("Zero-information BLR posterior did not return the configured prior variance.")

    informative_traj = torch.tensor([[[1.0, 0.0, 0.5], [2.0, 0.0, 1.0]]])
    mean, var = model._estimate_closed_form_theta_posterior(informative_traj, env=None)
    expected_var = torch.tensor([[1.0 / 24.0]])
    expected_mean = torch.tensor([[(4.0 * 0.1 + 10.0) / 24.0]])
    if not torch.allclose(mean, expected_mean, atol=1e-7):
        raise RuntimeError("Windowed BLR posterior mean does not match the scalar reference calculation.")
    if not torch.allclose(var, expected_var, atol=1e-7):
        raise RuntimeError("Windowed BLR posterior variance does not match the scalar reference calculation.")
    if not bool((var < zero_var).all()):
        raise RuntimeError("Windowed BLR posterior variance did not decrease with informative data.")


def _validate_configs(mean_dir: Path, mean_std_dir: Path) -> None:
    with open(mean_dir / "config.yaml", "r") as f:
        mean = yaml.safe_load(f)["model"]["params"]
    with open(mean_std_dir / "config.yaml", "r") as f:
        mean_std = yaml.safe_load(f)["model"]["params"]

    matched_keys = [
        "seed",
        "context_mode",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "learning_rate",
        "ent_coef",
        "regression_coef",
        "privileged_context_probability",
        "detach_context_for_rl",
        "actor_net_arch",
        "critic_net_arch",
        "uncertainty_reward_penalty_coef",
        "closed_form_prior_mean",
        "closed_form_prior_var",
    ]
    mismatched = [key for key in matched_keys if mean.get(key) != mean_std.get(key)]
    if mismatched:
        raise RuntimeError(f"Matched BLR mean/mean+std configs differ unexpectedly: {mismatched}")
    if mean["condition_on_uncertainty"] or not mean_std["condition_on_uncertainty"]:
        raise RuntimeError("BLR policies do not differ only by uncertainty conditioning.")
    if mean["context_mode"] != "closed_form" or mean["uncertainty_reward_penalty_coef"] != 0.0:
        raise RuntimeError("BLR experiment is not fixed closed-form identification with zero uncertainty penalty.")

    for exp_dir in [mean_dir, mean_std_dir]:
        first = _policy_group_digest(_checkpoint(exp_dir, "weights_P01"), encoder=True)
        last = _policy_group_digest(_checkpoint(exp_dir, "weights_P06"), encoder=True)
        if first != last:
            raise RuntimeError(f"Unused context encoder changed during BLR training: {exp_dir}")


def _validate_phase_checkpoints(dirs: dict[str, Path]) -> None:
    for exp_dir in dirs.values():
        for phase, *_ in GRADUAL_PHASES:
            _require_checkpoint(exp_dir, f"weights_{phase}")


def _compact_specs(dirs: dict[str, Path]) -> list[dict]:
    specs = []
    for method in ["mean", "mean_std"]:
        specs.extend(
            _ppo_spec(f"blr_{method}_{phase}", dirs[method], f"weights_{phase}")
            for phase, *_ in GRADUAL_PHASES
        )
    return specs + _lqr_specs(dirs["mean"])


def _full_specs(
    dirs: dict[str, Path],
    confirmation_run: dict,
    final_methods: dict,
) -> list[dict]:
    final_dirs = {key: Path(value) for key, value in final_methods["dirs"].items()}
    return [
        _ppo_spec("privileged", Path(confirmation_run["dirs"]["privileged"])),
        _ppo_spec("fully_trained_staged_S02", final_dirs["staged_policy"], "weights_S02"),
        _ppo_spec("extended_mle_P05", final_dirs["gradual_mle"], "weights_P05"),
        _ppo_spec("extended_mle_P06", final_dirs["gradual_mle"], "weights_P06"),
        _ppo_spec("extended_nll_P05", final_dirs["gradual_nll"], "weights_P05"),
        _ppo_spec("extended_nll_P06", final_dirs["gradual_nll"], "weights_P06"),
        _ppo_spec("blr_mean_P05", dirs["mean"], "weights_P05"),
        _ppo_spec("blr_mean_P06", dirs["mean"], "weights_P06"),
        _ppo_spec("blr_mean_std_P05", dirs["mean_std"], "weights_P05"),
        _ppo_spec("blr_mean_std_P06", dirs["mean_std"], "weights_P06"),
    ] + _lqr_specs(dirs["mean"])


def _write_manifest(run_root: Path, payload: dict) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    with open(run_root / "blr_ppo.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(run_root / "blr_ppo.yaml", "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main() -> None:
    _validate_blr_math()
    confirmation_dir, system, confirmation_run = _load_confirmation()
    final_methods_root, final_methods = _load_final_methods()

    original_text = CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_text)
    exp_root = _resolve_exp_root(base_cfg)
    stamp = datetime.now().strftime("%m-%d__%H-%M")
    smoke_suffix = "_smoke" if _smoke_enabled() else ""
    run_root = SELECTION_ROOT / f"blr_ppo_{stamp}_{system['label']}_seed{SEED}{smoke_suffix}"
    names = {
        "mean": f"s_{stamp}_{system['label']}_gradual_blr_mean_ppo{smoke_suffix}",
        "mean_std": f"s_{stamp}_{system['label']}_gradual_blr_mean_std_ppo{smoke_suffix}",
    }
    dirs = {key: exp_root / name for key, name in names.items()}
    payload = {
        "confirmation_dir": str(confirmation_dir),
        "final_methods_root": str(final_methods_root),
        "system": system,
        "seed": SEED,
        "smoke": _smoke_enabled(),
        "prior": {"mean": PRIOR_MEAN, "variance": PRIOR_VAR},
        "experiments": names,
        "dirs": {key: str(value.resolve()) for key, value in dirs.items()},
        "phases": [
            {
                "phase": phase,
                "blr_probability": probability,
                "privileged_probability": 1.0 - probability,
                "ent_coef": ent,
                "learning_rate": lr,
                "timesteps": _steps(steps),
            }
            for phase, probability, ent, lr, steps in GRADUAL_PHASES
        ],
    }
    _write_manifest(run_root, payload)

    try:
        _train_blr(
            condition_on_uncertainty=False,
            base_cfg=base_cfg,
            exp_root=exp_root,
            system=system,
            exp_name=names["mean"],
            exp_dir=dirs["mean"],
        )
        _train_blr(
            condition_on_uncertainty=True,
            base_cfg=base_cfg,
            exp_root=exp_root,
            system=system,
            exp_name=names["mean_std"],
            exp_dir=dirs["mean_std"],
        )
        _validate_phase_checkpoints(dirs)
        _validate_configs(dirs["mean"], dirs["mean_std"])
    finally:
        CONFIG_PATH.write_text(original_text)

    compact_dir = _run_sweep(
        target=dirs["mean"],
        specs=_compact_specs(dirs),
        output_subdir="theta_sweep_blr_compact_all_phases",
        theta_points=3 if _smoke_enabled() else 21,
        episodes_per_theta=1 if _smoke_enabled() else 10,
    )
    full_dir = _run_sweep(
        target=dirs["mean"],
        specs=_full_specs(dirs, confirmation_run, final_methods),
        output_subdir="theta_sweep_blr_full_final_comparison",
        theta_points=3 if _smoke_enabled() else 41,
        episodes_per_theta=1 if _smoke_enabled() else 20,
    )
    payload["sweeps"] = {
        "compact": str(compact_dir.resolve()),
        "full_final_comparison": str(full_dir.resolve()),
    }
    payload["validation"] = {
        "selected_system_and_seed_match": True,
        "zero_information_returns_prior": True,
        "informative_window_matches_scalar_blr": True,
        "posterior_variance_decreases": True,
        "mean_and_mean_std_schedules_matched": True,
        "unused_context_encoders_unchanged": True,
        "zero_uncertainty_reward_penalty": True,
        "all_phase_checkpoints_and_scorecards_produced": True,
    }
    _write_manifest(run_root, payload)
    if not _smoke_enabled():
        (SELECTION_ROOT / "latest_blr_ppo.txt").write_text(str(run_root.resolve()) + "\n")
    print(f"Saved BLR+PPO run to: {run_root}", flush=True)


if __name__ == "__main__":
    main()
