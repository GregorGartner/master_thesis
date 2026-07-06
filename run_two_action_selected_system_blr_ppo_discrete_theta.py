from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import yaml

import run_two_action_selected_system_blr_ppo as base
import run_two_action_selected_system_final_methods_discrete_theta as discrete
from run_two_action_pipeline import CONFIG_PATH, _resolve_exp_root


DISCRETE_PRIOR_MEAN = 0.0
DISCRETE_PRIOR_VAR = (0.25**2 + 0.0 + 0.25**2) / 3.0
POINTER = base.SELECTION_ROOT / "latest_blr_ppo_discrete_theta.txt"


def _load_discrete_final_methods() -> tuple[Path, dict]:
    pointer = base.SELECTION_ROOT / "latest_final_methods_discrete_theta.txt"
    if not pointer.exists():
        raise FileNotFoundError(
            "No latest discrete-theta final-method pointer found. "
            "Run run_two_action_selected_system_final_methods_discrete_theta.py first."
        )
    run_root = Path(pointer.read_text().strip()).resolve()
    manifest_path = run_root / "final_methods.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Discrete final-method manifest is missing: {manifest_path}")
    with manifest_path.open("r") as f:
        manifest = json.load(f)
    if manifest["system"]["label"] != base.SELECTED_SYSTEM_LABEL or int(manifest["seed"]) != base.SEED:
        raise RuntimeError("Discrete final-method manifest does not match the frozen selected system and seed.")
    for key in ["staged_policy", "gradual_mle", "gradual_nll"]:
        if key not in manifest["dirs"]:
            raise RuntimeError(f"Discrete final-method manifest is missing experiment directory {key!r}.")
    return run_root, manifest


def _base_cfg_discrete(base_cfg: dict, exp_root: Path, exp_name: str, steps: int, system: dict) -> dict:
    cfg = discrete._base_cfg(base_cfg, exp_root, exp_name, steps, system)
    discrete.apply_r22_variant_overrides(cfg)
    return cfg


def _run_discrete_sweep(
    *,
    target: Path,
    specs: list[dict],
    output_subdir: str,
    theta_points: int,
    episodes_per_theta: int,
) -> Path:
    del theta_points
    return discrete._run_sweep(
        target=target,
        specs=specs,
        output_subdir=output_subdir,
        episodes_per_theta=episodes_per_theta,
    )


def _write_manifest(run_root: Path, payload: dict) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "blr_ppo_discrete_theta.json").open("w") as f:
        json.dump(payload, f, indent=2)
    with (run_root / "blr_ppo_discrete_theta.yaml").open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main() -> None:
    base._validate_blr_math()
    confirmation_dir, system, confirmation_run = base._load_confirmation()
    final_methods_root, final_methods = _load_discrete_final_methods()

    original_base_cfg_fn = base._base_cfg
    original_run_sweep_fn = base._run_sweep
    original_set_blr_fn = base._set_blr
    original_prior_mean = base.PRIOR_MEAN
    original_prior_var = base.PRIOR_VAR

    base._base_cfg = _base_cfg_discrete
    base._run_sweep = _run_discrete_sweep

    def _set_blr_with_overrides(cfg: dict, **kwargs) -> None:
        original_set_blr_fn(cfg, **kwargs)
        discrete.apply_r22_variant_overrides(cfg)

    base._set_blr = _set_blr_with_overrides
    base.PRIOR_MEAN = DISCRETE_PRIOR_MEAN
    base.PRIOR_VAR = DISCRETE_PRIOR_VAR

    original_text = CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_text)
    exp_root = _resolve_exp_root(base_cfg)
    stamp = datetime.now().strftime("%m-%d__%H-%M")
    smoke_suffix = "_smoke" if base._smoke_enabled() else ""
    run_root = base.SELECTION_ROOT / f"blr_ppo_discrete_theta_{stamp}_{system['label']}_seed{base.SEED}{smoke_suffix}"
    names = {
        "mean": f"s_{stamp}_{system['label']}_discrete_theta_gradual_blr_mean_ppo{smoke_suffix}",
        "mean_std": f"s_{stamp}_{system['label']}_discrete_theta_gradual_blr_mean_std_ppo{smoke_suffix}",
    }
    dirs = {key: exp_root / name for key, name in names.items()}
    payload = {
        "confirmation_dir": str(confirmation_dir),
        "discrete_final_methods_root": str(final_methods_root),
        "system": system,
        "seed": base.SEED,
        "smoke": base._smoke_enabled(),
        "theta_distribution": {
            "type": "categorical",
            "values": list(discrete.DISCRETE_THETA_VALUES),
        },
        "r22_variant_overrides": discrete.r22_variant_overrides(),
        "prior": {"mean": DISCRETE_PRIOR_MEAN, "variance": DISCRETE_PRIOR_VAR},
        "experiments": names,
        "dirs": {key: str(value.resolve()) for key, value in dirs.items()},
        "phases": [
            {
                "phase": phase,
                "blr_probability": probability,
                "privileged_probability": 1.0 - probability,
                "ent_coef": ent,
                "learning_rate": lr,
                "timesteps": base._steps(steps),
            }
            for phase, probability, ent, lr, steps in base.GRADUAL_PHASES
        ],
    }
    _write_manifest(run_root, payload)

    try:
        try:
            base._train_blr(
                condition_on_uncertainty=False,
                base_cfg=base_cfg,
                exp_root=exp_root,
                system=system,
                exp_name=names["mean"],
                exp_dir=dirs["mean"],
            )
            base._train_blr(
                condition_on_uncertainty=True,
                base_cfg=base_cfg,
                exp_root=exp_root,
                system=system,
                exp_name=names["mean_std"],
                exp_dir=dirs["mean_std"],
            )
            base._validate_phase_checkpoints(dirs)
            base._validate_configs(dirs["mean"], dirs["mean_std"])
        finally:
            CONFIG_PATH.write_text(original_text)

        compact_dir = base._run_sweep(
            target=dirs["mean"],
            specs=base._compact_specs(dirs),
            output_subdir="theta_sweep_discrete_blr_compact_all_phases",
            theta_points=len(discrete.DISCRETE_THETA_VALUES),
            episodes_per_theta=1 if base._smoke_enabled() else 10,
        )
        full_dir = base._run_sweep(
            target=dirs["mean"],
            specs=base._full_specs(dirs, confirmation_run, final_methods),
            output_subdir="theta_sweep_discrete_blr_full_final_comparison",
            theta_points=len(discrete.DISCRETE_THETA_VALUES),
            episodes_per_theta=1 if base._smoke_enabled() else 20,
        )
        payload["sweeps"] = {
            "compact": str(compact_dir.resolve()),
            "full_final_comparison": str(full_dir.resolve()),
        }
        payload["validation"] = {
            "selected_system_and_seed_match": True,
            "categorical_theta_values": list(discrete.DISCRETE_THETA_VALUES),
            "zero_information_returns_prior": True,
            "informative_window_matches_scalar_blr": True,
            "posterior_variance_decreases": True,
            "mean_and_mean_std_schedules_matched": True,
            "unused_context_encoders_unchanged": True,
            "zero_uncertainty_reward_penalty": True,
            "all_phase_checkpoints_and_scorecards_produced": True,
        }
        _write_manifest(run_root, payload)
        if not base._smoke_enabled():
            POINTER.write_text(str(run_root.resolve()) + "\n")
        print(f"Saved discrete-theta BLR+PPO run to: {run_root}", flush=True)
    finally:
        base._base_cfg = original_base_cfg_fn
        base._run_sweep = original_run_sweep_fn
        base._set_blr = original_set_blr_fn
        base.PRIOR_MEAN = original_prior_mean
        base.PRIOR_VAR = original_prior_var


if __name__ == "__main__":
    main()
