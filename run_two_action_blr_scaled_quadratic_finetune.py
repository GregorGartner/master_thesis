from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import yaml

from run_two_action_pipeline import CONFIG_PATH, _resolve_exp_root, _snapshot_phase_weights
from run_two_action_selected_system_blr_ppo import (
    PRIOR_MEAN,
    PRIOR_VAR,
    _load_final_methods,
    _validate_blr_math,
)
from run_two_action_selected_system_final_methods import (
    SEED,
    SELECTION_ROOT,
    SELECTED_SYSTEM_LABEL,
    _base_cfg,
    _checkpoint,
    _load_confirmation,
    _lqr_specs,
    _ppo_spec,
    _require_checkpoint,
    _run_stage,
    _run_sweep,
    _set_load,
)
from run_two_action_system_neural_screening import _set_common


PHASES = [
    # phase, steps, learning_rate, ent_coef, policy_loss_coef, target_kl, gae_lambda
    ("Q00", 750_000, 1e-5, 0.0, 0.0, None, 1.0),
    ("Q01", 2_000_000, 1e-5, 0.0, 1.0, 0.003, 0.95),
    ("Q02", 3_000_000, 3e-5, 0.0, 1.0, 0.01, 0.95),
]


def _smoke_enabled() -> bool:
    return os.environ.get("TWO_ACTION_BLR_SCALED_QUAD_SMOKE", "0").lower() in {
        "1",
        "true",
        "yes",
    }


def _steps(steps: int) -> int:
    return 16_384 if _smoke_enabled() else int(steps)


def _load_blr_run() -> tuple[Path, dict]:
    pointer = SELECTION_ROOT / "latest_blr_ppo.txt"
    if not pointer.exists():
        raise FileNotFoundError("No BLR+PPO pointer found. Run run_two_action_selected_system_blr_ppo.py first.")
    run_root = Path(pointer.read_text().strip()).resolve()
    manifest_path = run_root / "blr_ppo.json"
    if not manifest_path.exists():
        raise RuntimeError(f"BLR+PPO manifest is missing: {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    if manifest["system"]["label"] != SELECTED_SYSTEM_LABEL or int(manifest["seed"]) != SEED:
        raise RuntimeError("BLR+PPO manifest does not match the frozen selected system and seed.")
    return run_root, manifest


def _read_scorecard(path: Path) -> list[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _best_blr_checkpoint(blr_manifest: dict, *, method: str) -> dict:
    mean_dir = Path(blr_manifest["dirs"]["mean"])
    scorecard = mean_dir / "theta_sweep_blr_compact_all_phases" / "controller_scorecard.csv"
    if not scorecard.exists():
        raise FileNotFoundError(f"Missing BLR compact scorecard: {scorecard}")
    if method == "mean":
        prefix = "blr_mean_P"
    elif method == "mean_std":
        prefix = "blr_mean_std_P"
    else:
        raise ValueError(f"Unknown BLR method: {method}")

    rows = [row for row in _read_scorecard(scorecard) if row["controller"].startswith(prefix)]
    if not rows:
        raise RuntimeError(f"No {method} BLR checkpoints found in {scorecard}")
    best = max(rows, key=lambda row: float(row["mean_return"]))
    phase = best["controller"].rsplit("_", 1)[-1]
    exp_dir = Path(blr_manifest["dirs"][method])
    weights_name = f"weights_{phase}"
    _require_checkpoint(exp_dir, weights_name)
    return {
        "controller": best["controller"],
        "phase": phase,
        "weights_name": weights_name,
        "experiment_dir": str(exp_dir.resolve()),
        "mean_return": float(best["mean_return"]),
        "tail_mean_return": float(best["tail_mean_return"]),
        "center_mean_return": float(best["center_mean_return"]),
    }


def _set_scaled_quad_blr(
    cfg: dict,
    *,
    condition_on_uncertainty: bool,
    lr: float,
    ent: float,
    policy_loss_coef: float,
    target_kl: float | None,
    gae_lambda: float,
) -> None:
    _set_common(cfg, SEED)
    lqr = cfg.setdefault("lqr_env", {})
    lqr["reward_cost_mode"] = "raw"
    lqr["reward_cost_scale"] = 1.0
    lqr["action_cost_type"] = "quadratic"

    params = cfg["model"]["params"]
    params["context_mode"] = "closed_form"
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["condition_on_uncertainty"] = bool(condition_on_uncertainty)
    params["privileged_uncertainty_mode"] = "zeros"
    params["privileged_context_probability"] = 0.0
    params["closed_form_prior_mean"] = PRIOR_MEAN
    params["closed_form_prior_var"] = PRIOR_VAR
    params["uncertainty_regularization_coef"] = 0.0
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["detach_context_for_rl"] = True
    params["naive_action_noise_std"] = 0.0
    params["learning_rate"] = float(lr)
    params["ent_coef"] = float(ent)
    params["policy_loss_coef"] = float(policy_loss_coef)
    params["target_kl"] = target_kl
    params["gae_lambda"] = float(gae_lambda)


def _train_method(
    *,
    method: str,
    condition_on_uncertainty: bool,
    base_cfg: dict,
    exp_root: Path,
    system: dict,
    exp_name: str,
    exp_dir: Path,
    start: dict,
) -> None:
    for phase_idx, (phase, steps, lr, ent, policy_loss_coef, target_kl, gae_lambda) in enumerate(PHASES):
        cfg = _base_cfg(base_cfg, exp_root, exp_name, _steps(steps), system)
        _set_scaled_quad_blr(
            cfg,
            condition_on_uncertainty=condition_on_uncertainty,
            lr=lr,
            ent=ent,
            policy_loss_coef=policy_loss_coef,
            target_kl=target_kl,
            gae_lambda=gae_lambda,
        )
        if phase_idx == 0:
            _set_load(cfg, Path(start["experiment_dir"]), start["weights_name"])
        else:
            _set_load(cfg, exp_dir, f"weights_{PHASES[phase_idx - 1][0]}")
        _run_stage(f"scaled_quadratic_finetune/{method}/{phase}", cfg)
        _snapshot_phase_weights(exp_dir, phase)


def _compact_specs(dirs: dict[str, Path], starts: dict[str, dict]) -> list[dict]:
    specs = []
    for method in ["mean", "mean_std"]:
        specs.append(
            _ppo_spec(
                f"start_{starts[method]['controller']}",
                Path(starts[method]["experiment_dir"]),
                starts[method]["weights_name"],
            )
        )
        specs.extend(_ppo_spec(f"{method}_{phase}", dirs[method], f"weights_{phase}") for phase, *_ in PHASES)
    return specs + _lqr_specs(dirs["mean"])


def _full_specs(
    dirs: dict[str, Path],
    starts: dict[str, dict],
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
        _ppo_spec(
            f"start_{starts['mean']['controller']}",
            Path(starts["mean"]["experiment_dir"]),
            starts["mean"]["weights_name"],
        ),
        _ppo_spec(
            f"start_{starts['mean_std']['controller']}",
            Path(starts["mean_std"]["experiment_dir"]),
            starts["mean_std"]["weights_name"],
        ),
        _ppo_spec("scaled_quad_mean_Q02", dirs["mean"], "weights_Q02"),
        _ppo_spec("scaled_quad_mean_std_Q02", dirs["mean_std"], "weights_Q02"),
    ] + _lqr_specs(dirs["mean"])


def _write_manifest(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "blr_scaled_quadratic_finetune.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(path / "blr_scaled_quadratic_finetune.yaml", "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main() -> None:
    _validate_blr_math()
    confirmation_dir, system, confirmation_run = _load_confirmation()
    final_methods_root, final_methods = _load_final_methods()
    blr_root, blr_manifest = _load_blr_run()

    starts = {
        "mean": _best_blr_checkpoint(blr_manifest, method="mean"),
        "mean_std": _best_blr_checkpoint(blr_manifest, method="mean_std"),
    }

    original_text = CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_text)
    exp_root = _resolve_exp_root(base_cfg)
    stamp = datetime.now().strftime("%m-%d__%H-%M")
    smoke_suffix = "_smoke" if _smoke_enabled() else ""
    run_root = SELECTION_ROOT / f"blr_scaled_quad_ft_{stamp}_{system['label']}_seed{SEED}{smoke_suffix}"
    names = {
        "mean": f"s_{stamp}_{system['label']}_blr_mean_scaled_quad_ft{smoke_suffix}",
        "mean_std": f"s_{stamp}_{system['label']}_blr_mean_std_scaled_quad_ft{smoke_suffix}",
    }
    dirs = {key: exp_root / value for key, value in names.items()}
    payload = {
        "confirmation_dir": str(confirmation_dir),
        "final_methods_root": str(final_methods_root),
        "source_blr_root": str(blr_root),
        "system": system,
        "seed": SEED,
        "smoke": _smoke_enabled(),
        "starts": starts,
        "experiments": names,
        "dirs": {key: str(value.resolve()) for key, value in dirs.items()},
        "reward_cost_mode": "raw",
        "reward_cost_scale": 1.0,
        "action_cost_type": "quadratic",
        "phases": [
            {
                "phase": phase,
                "timesteps": _steps(steps),
                "learning_rate": lr,
                "ent_coef": ent,
                "policy_loss_coef": policy_loss_coef,
                "target_kl": target_kl,
                "gae_lambda": gae_lambda,
            }
            for phase, steps, lr, ent, policy_loss_coef, target_kl, gae_lambda in PHASES
        ],
    }
    _write_manifest(run_root, payload)

    try:
        _train_method(
            method="mean",
            condition_on_uncertainty=False,
            base_cfg=base_cfg,
            exp_root=exp_root,
            system=system,
            exp_name=names["mean"],
            exp_dir=dirs["mean"],
            start=starts["mean"],
        )
        _train_method(
            method="mean_std",
            condition_on_uncertainty=True,
            base_cfg=base_cfg,
            exp_root=exp_root,
            system=system,
            exp_name=names["mean_std"],
            exp_dir=dirs["mean_std"],
            start=starts["mean_std"],
        )
    finally:
        CONFIG_PATH.write_text(original_text)

    for exp_dir in dirs.values():
        for phase, *_ in PHASES:
            _require_checkpoint(exp_dir, f"weights_{phase}")

    compact_dir = _run_sweep(
        target=dirs["mean"],
        specs=_compact_specs(dirs, starts),
        output_subdir="theta_sweep_scaled_quad_compact",
        theta_points=3 if _smoke_enabled() else 21,
        episodes_per_theta=1 if _smoke_enabled() else 10,
    )
    full_dir = _run_sweep(
        target=dirs["mean"],
        specs=_full_specs(dirs, starts, confirmation_run, final_methods),
        output_subdir="theta_sweep_scaled_quad_full_comparison",
        theta_points=3 if _smoke_enabled() else 41,
        episodes_per_theta=1 if _smoke_enabled() else 20,
    )
    payload["sweeps"] = {
        "compact": str(compact_dir.resolve()),
        "full_comparison": str(full_dir.resolve()),
    }
    payload["validation"] = {
        "selected_system_and_seed_match": True,
        "source_best_checkpoints_selected_by_mean_return": True,
        "value_only_phase_policy_loss_coef_zero": True,
        "raw_quadratic_training_objective": True,
        "all_phase_checkpoints_and_scorecards_produced": True,
    }
    _write_manifest(run_root, payload)
    if not _smoke_enabled():
        (SELECTION_ROOT / "latest_blr_scaled_quad_ft.txt").write_text(str(run_root.resolve()) + "\n")
    print(f"Saved BLR scaled-quadratic fine-tune run to: {run_root}", flush=True)


if __name__ == "__main__":
    main()
