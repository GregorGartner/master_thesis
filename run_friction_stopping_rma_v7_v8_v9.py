from __future__ import annotations

import csv
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from friction_stopping_utils import (
    CONFIG_PATH,
    OUT_ROOT,
    evaluate_handcoded,
    evaluate_model,
    make_base_cfg,
    run_sanity_checks,
    run_training_stage,
    save_json,
    set_gradual_policy,
    set_load,
    set_privileged,
    smoke_enabled,
    steps,
    timestamp,
    write_final_comparison_diagnostics,
    write_named_comparison_diagnostics,
    write_probe_fraction_by_context,
)


@dataclass(frozen=True)
class Variant:
    version: str
    action_cost_weight: float
    safety_cost_weight: float
    crash_penalty: float
    crash_remaining_penalty: float
    calibration_noise_std: list[float]
    calibration_step_overrides: dict[str, int] = field(default_factory=dict)


VARIANTS = [
    Variant(
        version="v7",
        action_cost_weight=0.20,
        safety_cost_weight=10.0,
        crash_penalty=200.0,
        crash_remaining_penalty=200.0,
        calibration_noise_std=[0.15, 0.60],
    ),
    Variant(
        version="v8",
        action_cost_weight=0.15,
        safety_cost_weight=10.0,
        crash_penalty=200.0,
        crash_remaining_penalty=200.0,
        calibration_noise_std=[0.15, 0.60],
    ),
    Variant(
        version="v9",
        action_cost_weight=0.20,
        safety_cost_weight=10.0,
        crash_penalty=200.0,
        crash_remaining_penalty=200.0,
        calibration_noise_std=[0.25, 0.75],
        calibration_step_overrides={"CAL02": 900_000, "CAL03": 1_000_000, "CAL04": 1_200_000},
    ),
]


BASE_ENV_OVERRIDES = {
    "max_episode_steps": 300,
    "event_time_low": 80,
    "event_time_high": 260,
    "initial_velocity_low": 2.0,
    "initial_velocity_high": 3.0,
    "brake_accel_max": 8.0,
    "throttle_accel_max": 8.0,
    "saturate_throttle_by_friction": False,
}

BASE_MODEL_OVERRIDES = {
    "window_length": 30,
    "nominal_warmup_steps": 0,
    "id_update_interval": 1,
    "action_log_std_init": -1.2,
}

PHASE_DISTANCE_RANGES = {
    "start_visible": (2.0, 2.8),
    "easy": (0.9, 1.8),
    "medium": (0.9, 1.5),
    "final": (0.9, 1.2),
}

PRIVILEGED_PHASES = [
    ("A_start_visible", "start_visible", True, 1_000_000, 0.004, 3e-4),
    ("B_mixed_easy", "easy", False, 1_800_000, 0.003, 3e-4),
    ("C_mixed_medium", "medium", False, 2_200_000, 0.0015, 2e-4),
    ("D_hard_final", "final", False, 4_000_000, 0.0005, 1e-4),
]

ADAPTIVE_PHASES = [
    ("P01", "policy", "easy", 0.25, 0.004, 3e-4, 1_200_000, False),
    ("P02", "policy", "medium", 0.50, 0.003, 3e-4, 1_400_000, False),
    ("CAL02", "calibration", "medium", 0.50, 0.0, 1e-4, 600_000, False),
    ("P03", "policy", "final", 0.75, 0.0015, 2e-4, 1_700_000, False),
    ("CAL03", "calibration", "final", 0.75, 0.0, 1e-4, 700_000, False),
    ("P04", "policy", "final", 1.00, 0.0005, 1e-4, 2_000_000, False),
    ("CAL04", "calibration", "final", 1.00, 0.0, 1e-4, 800_000, False),
    ("FT01", "policy", "final", 1.00, 0.0010, 5e-5, 1_200_000, True),
    ("FT02", "policy", "final", 1.00, 0.0002, 3e-5, 1_200_000, True),
]

SMOKE_ADAPTIVE_PHASES = [
    ("P01", "policy", "easy", 0.25, 0.004, 3e-4, 1_200_000, False),
    ("CAL01", "calibration", "easy", 0.25, 0.0, 1e-4, 300_000, False),
    ("FT01", "policy", "final", 1.00, 0.0010, 5e-5, 300_000, True),
]

CALIBRATION_NOISE_DIST = ["gaussian", "uniform"]

PRIVILEGED_GATE_MAX_CRASH_RATE = 0.05
PRIVILEGED_GATE_MIN_SUCCESS_RATE = 0.95
PRIVILEGED_GATE_MAX_PRE_BOUND_FRAC = 0.30


def _exp_dir(run_dir: Path, name: str) -> Path:
    return run_dir / name


def _snapshot(exp_dir: Path, phase: str) -> None:
    for name in ["weights", "weights_best"]:
        src = exp_dir / f"{name}.zip"
        if src.exists():
            shutil.copy2(src, exp_dir / f"{name}_{phase}.zip")
    metric = exp_dir / "weights_best.metric"
    if metric.exists():
        shutil.copy2(metric, exp_dir / f"weights_best_{phase}.metric")


def _env_overrides(variant: Variant) -> dict[str, Any]:
    return {
        **BASE_ENV_OVERRIDES,
        "action_cost_weight": variant.action_cost_weight,
        "safety_cost_weight": variant.safety_cost_weight,
        "crash_penalty": variant.crash_penalty,
        "crash_remaining_penalty": variant.crash_remaining_penalty,
    }


def _privileged_phases() -> list[tuple[str, str, bool, int, float, float]]:
    if smoke_enabled():
        return [PRIVILEGED_PHASES[0], PRIVILEGED_PHASES[-1]]
    return PRIVILEGED_PHASES


def _adaptive_phases(variant: Variant) -> list[tuple[str, str, str, float, float, float, int, bool]]:
    source = SMOKE_ADAPTIVE_PHASES if smoke_enabled() else ADAPTIVE_PHASES
    out = []
    for phase_name, phase_kind, env_phase, enc_prob, ent, lr, raw_steps, freeze_encoder in source:
        raw_steps = variant.calibration_step_overrides.get(phase_name, raw_steps)
        out.append((phase_name, phase_kind, env_phase, enc_prob, ent, lr, raw_steps, freeze_encoder))
    return out


def _stage_cfg(
    base_cfg: dict,
    run_dir: Path,
    exp_name: str,
    variant: Variant,
    *,
    total_timesteps: int,
    phase: str,
    start_visible: bool = False,
) -> dict:
    lo, hi = PHASE_DISTANCE_RANGES[phase]
    env_overrides = {
        **_env_overrides(variant),
        "visible_distance_low": lo,
        "visible_distance_high": hi,
    }
    return make_base_cfg(
        base_cfg,
        exp_root=run_dir,
        exp_name=exp_name,
        total_timesteps=steps(total_timesteps),
        phase=phase,
        start_visible=start_visible,
        env_overrides=env_overrides,
        model_overrides=BASE_MODEL_OVERRIDES,
    )


def _train_privileged(base_cfg: dict, run_dir: Path, variant: Variant) -> Path:
    exp_name = f"privileged_{variant.version}"
    exp_dir = _exp_dir(run_dir, exp_name)
    for idx, (phase_name, env_phase, start_visible, raw_steps, ent, lr) in enumerate(_privileged_phases()):
        cfg = _stage_cfg(
            base_cfg,
            run_dir,
            exp_name,
            variant,
            total_timesteps=raw_steps,
            phase=env_phase,
            start_visible=start_visible,
        )
        set_privileged(cfg, ent=ent, lr=lr)
        set_load(cfg, exp_dir if idx > 0 else None)
        run_training_stage(f"{variant.version} privileged/{phase_name}", cfg)
        _snapshot(exp_dir, phase_name)
        evaluate_model(
            exp_dir,
            run_dir / "evaluations" / f"privileged_{phase_name}",
            label=f"privileged_{phase_name}",
            seeds=3 if smoke_enabled() else 12,
        )
    return exp_dir


def _privileged_gate(run_dir: Path) -> dict:
    score_path = run_dir / "evaluations" / "privileged_D_hard_final" / "scorecard.csv"
    payload = {
        "scorecard": str(score_path),
        "max_crash_rate": PRIVILEGED_GATE_MAX_CRASH_RATE,
        "min_success_rate": PRIVILEGED_GATE_MIN_SUCCESS_RATE,
        "max_pre_bound_frac": PRIVILEGED_GATE_MAX_PRE_BOUND_FRAC,
        "passed": False,
        "reason": "missing_scorecard",
    }
    if not score_path.exists():
        return payload
    with score_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        payload["reason"] = "empty_scorecard"
        return payload

    row = rows[0]
    crash_rate = float(row.get("crashed_mean", "nan"))
    success_rate = float(row.get("success_mean", "nan"))
    pre_bound_frac = float(row.get("pre_bound_frac_mean", "nan"))
    reliability_passed = crash_rate <= PRIVILEGED_GATE_MAX_CRASH_RATE and success_rate >= PRIVILEGED_GATE_MIN_SUCCESS_RATE
    bound_passed = pre_bound_frac <= PRIVILEGED_GATE_MAX_PRE_BOUND_FRAC
    passed = reliability_passed and bound_passed
    if passed:
        reason = "passed"
    elif not reliability_passed:
        reason = "privileged_not_reliable_enough"
    else:
        reason = "privileged_too_much_pre_obstacle_bang_bang"
    payload.update(
        {
            "crash_rate": crash_rate,
            "success_rate": success_rate,
            "pre_bound_frac": pre_bound_frac,
            "reliability_passed": bool(reliability_passed),
            "bound_passed": bool(bound_passed),
            "passed": bool(passed),
            "reason": reason,
        }
    )
    return payload


def _set_calibration_policy(
    cfg: dict,
    variant: Variant,
    *,
    condition_on_uncertainty: bool,
    encoder_probability: float,
    lr: float,
) -> None:
    if len(variant.calibration_noise_std) != 2:
        raise ValueError("calibration_noise_std must contain exactly two values; the PPO code samples uniformly between them.")
    set_gradual_policy(
        cfg,
        method="nll",
        condition_on_uncertainty=condition_on_uncertainty,
        encoder_probability=encoder_probability,
        ent=0.0,
        lr=lr,
        freeze_encoder=False,
    )
    params = cfg["model"]["params"]
    params["freeze_ppo"] = True
    params["detach_context_for_rl"] = True
    params["regression_coef"] = 1.0
    params["policy_loss_coef"] = 0.0
    params["naive_action_noise_std"] = list(variant.calibration_noise_std)
    params["naive_action_noise_dist"] = list(CALIBRATION_NOISE_DIST)


def _set_policy_phase(
    cfg: dict,
    *,
    condition_on_uncertainty: bool,
    encoder_probability: float,
    ent: float,
    lr: float,
    freeze_encoder: bool,
) -> None:
    set_gradual_policy(
        cfg,
        method="nll",
        condition_on_uncertainty=condition_on_uncertainty,
        encoder_probability=encoder_probability,
        ent=ent,
        lr=lr,
        freeze_encoder=freeze_encoder,
    )
    params = cfg["model"]["params"]
    params["freeze_ppo"] = False
    params["detach_context_for_rl"] = True
    params["policy_loss_coef"] = 1.0
    params["naive_action_noise_std"] = 0.0
    params["naive_action_noise_dist"] = "gaussian"


def _train_adaptive_nll(
    base_cfg: dict,
    run_dir: Path,
    privileged_dir: Path,
    variant: Variant,
    *,
    name: str,
    condition_on_uncertainty: bool,
) -> Path:
    exp_dir = _exp_dir(run_dir, name)
    for idx, (phase_name, phase_kind, env_phase, enc_prob, ent, lr, raw_steps, freeze_encoder) in enumerate(
        _adaptive_phases(variant)
    ):
        cfg = _stage_cfg(base_cfg, run_dir, name, variant, total_timesteps=raw_steps, phase=env_phase)
        if phase_kind == "calibration":
            _set_calibration_policy(
                cfg,
                variant,
                condition_on_uncertainty=condition_on_uncertainty,
                encoder_probability=enc_prob,
                lr=lr,
            )
        else:
            _set_policy_phase(
                cfg,
                condition_on_uncertainty=condition_on_uncertainty,
                encoder_probability=enc_prob,
                ent=ent,
                lr=lr,
                freeze_encoder=freeze_encoder,
            )
        params = cfg["model"]["params"]
        params["initial_context"] = [0.5, 0.2] if condition_on_uncertainty else [0.5]
        set_load(cfg, privileged_dir if idx == 0 else exp_dir)
        run_training_stage(f"{variant.version} {name}/{phase_name}", cfg)
        _snapshot(exp_dir, phase_name)
        eval_dir = run_dir / "evaluations" / f"{name}_{phase_name}"
        evaluate_model(
            exp_dir,
            eval_dir,
            label=f"{name}_{phase_name}",
            seeds=3 if smoke_enabled() else 12,
        )
        if condition_on_uncertainty and phase_kind == "calibration":
            # Keep a top-level probe-context view for calibration checkpoints too.
            # Mean-only may not have a matching checkpoint yet, so this is best-effort.
            pass
    return exp_dir


def _score_return(eval_dir: Path) -> float:
    score_path = eval_dir / "scorecard.csv"
    if not score_path.exists():
        return float("-inf")
    with score_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return float("-inf")
    return float(rows[0].get("return_mean", "-inf"))


def _existing_evaluations(run_dir: Path, names: list[tuple[str, str]]) -> dict[str, Path]:
    evaluations = run_dir / "evaluations"
    return {label: evaluations / eval_name for label, eval_name in names if (evaluations / eval_name / "scorecard.csv").exists()}


def _write_checkpoint_comparisons(run_dir: Path, variant: Variant) -> None:
    evaluations = run_dir / "evaluations"
    final_phase = "FT01" if smoke_enabled() else "FT02"
    mean_only_prefix = f"gradual_nll_mean_only_{variant.version}"
    mean_std_prefix = f"gradual_nll_mean_std_{variant.version}"
    mean_only_final_eval = evaluations / f"{mean_only_prefix}_{final_phase}"
    mean_std_final_eval = evaluations / f"{mean_std_prefix}_{final_phase}"

    write_final_comparison_diagnostics(
        run_dir,
        model_evaluations={
            "privileged": evaluations / "privileged_D_hard_final",
            "mean-only": mean_only_final_eval,
            "mean+std": mean_std_final_eval,
        },
        mean_only_eval_dir=mean_only_final_eval,
        mean_std_eval_dir=mean_std_final_eval,
    )

    if mean_only_final_eval.exists() and mean_std_final_eval.exists():
        write_probe_fraction_by_context(mean_only_final_eval, mean_std_final_eval, evaluations)

    phases = ["P01", "FT01"] if smoke_enabled() else ["P04", "FT01", "FT02"]
    checkpoint_evals = _existing_evaluations(
        run_dir,
        [("privileged", "privileged_D_hard_final")]
        + [(f"mean-only {phase}", f"{mean_only_prefix}_{phase}") for phase in phases]
        + [(f"mean+std {phase}", f"{mean_std_prefix}_{phase}") for phase in phases],
    )
    write_named_comparison_diagnostics(
        run_dir,
        model_evaluations=checkpoint_evals,
        output_prefix="checkpoint",
        title_prefix=f"Friction stopping {variant.version} checkpoint",
        handcoded_controllers=(),
    )

    best_evals: dict[str, Path] = {"privileged": evaluations / "privileged_D_hard_final"}
    best_payload: dict[str, Any] = {}
    for method_label, prefix in [("mean-only", mean_only_prefix), ("mean+std", mean_std_prefix)]:
        candidates = [(phase, evaluations / f"{prefix}_{phase}") for phase in phases]
        candidates = [(phase, path) for phase, path in candidates if (path / "scorecard.csv").exists()]
        if not candidates:
            continue
        best_phase, best_path = max(candidates, key=lambda item: _score_return(item[1]))
        best_evals[f"{method_label} best {best_phase}"] = best_path
        best_payload[method_label] = {
            "phase": best_phase,
            "eval_dir": str(best_path),
            "return_mean": _score_return(best_path),
        }
    save_json(evaluations / "best_checkpoint_selection.json", best_payload)
    write_named_comparison_diagnostics(
        run_dir,
        model_evaluations=best_evals,
        output_prefix="best_checkpoint",
        title_prefix=f"Friction stopping {variant.version} best checkpoint",
        handcoded_controllers=(),
    )


def _write_phase_probe_contexts(run_dir: Path, variant: Variant) -> None:
    evaluations = run_dir / "evaluations"
    mean_only_prefix = f"gradual_nll_mean_only_{variant.version}"
    mean_std_prefix = f"gradual_nll_mean_std_{variant.version}"
    for phase_name, *_rest in _adaptive_phases(variant):
        mean_only_eval = evaluations / f"{mean_only_prefix}_{phase_name}"
        mean_std_eval = evaluations / f"{mean_std_prefix}_{phase_name}"
        if (mean_only_eval / "trajectory_trace.csv").exists() and (mean_std_eval / "trajectory_trace.csv").exists():
            write_probe_fraction_by_context(mean_only_eval, mean_std_eval, mean_std_eval)


def _write_manifest(run_dir: Path, variant: Variant, stamp: str) -> None:
    save_json(
        run_dir / "run_manifest.json",
        {
            "timestamp": stamp,
            "run_dir": str(run_dir.resolve()),
            "smoke": smoke_enabled(),
            "version": variant.version,
            "mu_values": [0.25, 0.5, 0.7],
            "env_overrides": _env_overrides(variant),
            "phase_distance_ranges": PHASE_DISTANCE_RANGES,
            "model_overrides": BASE_MODEL_OVERRIDES,
            "adaptive_phases": [
                {
                    "name": phase_name,
                    "kind": phase_kind,
                    "env_phase": env_phase,
                    "encoder_probability": enc_prob,
                    "entropy": ent,
                    "learning_rate": lr,
                    "steps": raw_steps,
                    "freeze_encoder": freeze_encoder,
                }
                for phase_name, phase_kind, env_phase, enc_prob, ent, lr, raw_steps, freeze_encoder in _adaptive_phases(variant)
            ],
            "calibration_noise_std": variant.calibration_noise_std,
            "calibration_noise_dist": CALIBRATION_NOISE_DIST,
            "final_visible_distance": [0.9, 1.2],
            "focused_methods": [
                f"gradual_nll_mean_only_{variant.version}",
                f"gradual_nll_mean_std_{variant.version}",
            ],
            "privileged_gate": {
                "max_crash_rate": PRIVILEGED_GATE_MAX_CRASH_RATE,
                "min_success_rate": PRIVILEGED_GATE_MIN_SUCCESS_RATE,
                "max_pre_bound_frac": PRIVILEGED_GATE_MAX_PRE_BOUND_FRAC,
                "ignore_env_vars": [
                    "FRICTION_STOPPING_SWEEP_IGNORE_GATE",
                    f"FRICTION_STOPPING_{variant.version.upper()}_IGNORE_GATE",
                ],
            },
        },
    )


def _ignore_gate(variant: Variant) -> bool:
    keys = ["FRICTION_STOPPING_SWEEP_IGNORE_GATE", f"FRICTION_STOPPING_{variant.version.upper()}_IGNORE_GATE"]
    return any(os.environ.get(key, "0").lower() in {"1", "true", "yes"} for key in keys)


def _run_variant(base_cfg: dict, original_config: str, variant: Variant) -> Path:
    stamp = timestamp()
    run_dir = OUT_ROOT / f"friction_stopping_{variant.version}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"latest_friction_stopping_{variant.version}.txt").write_text(str(run_dir.resolve()) + "\n")
    _write_manifest(run_dir, variant, stamp)

    try:
        print(f"Writing friction-stopping {variant.version} outputs to {run_dir}", flush=True)
        run_sanity_checks(run_dir / "sanity", env_overrides=_env_overrides(variant))
        evaluate_handcoded(run_dir / "handcoded", seeds=2 if smoke_enabled() else 20, phase="final", env_overrides=_env_overrides(variant))

        privileged_dir = _train_privileged(base_cfg, run_dir, variant)
        gate_payload = _privileged_gate(run_dir)
        save_json(run_dir / "privileged_gate.json", gate_payload)
        print(f"Friction-stopping {variant.version} privileged gate: {gate_payload}", flush=True)

        if not gate_payload.get("passed", False) and not smoke_enabled() and not _ignore_gate(variant):
            print(
                f"Stopping {variant.version} before adaptive training because privileged PPO did not pass the gate. "
                f"Set FRICTION_STOPPING_{variant.version.upper()}_IGNORE_GATE=1 or "
                "FRICTION_STOPPING_SWEEP_IGNORE_GATE=1 to override.",
                flush=True,
            )
            return run_dir

        mean_only_dir = _train_adaptive_nll(
            base_cfg,
            run_dir,
            privileged_dir,
            variant,
            name=f"gradual_nll_mean_only_{variant.version}",
            condition_on_uncertainty=False,
        )
        mean_std_dir = _train_adaptive_nll(
            base_cfg,
            run_dir,
            privileged_dir,
            variant,
            name=f"gradual_nll_mean_std_{variant.version}",
            condition_on_uncertainty=True,
        )

        _write_phase_probe_contexts(run_dir, variant)
        _write_checkpoint_comparisons(run_dir, variant)
        print(
            f"Friction-stopping {variant.version} complete: {run_dir}\n"
            f"Trained directories: privileged={privileged_dir}, mean_only={mean_only_dir}, mean_std={mean_std_dir}",
            flush=True,
        )
        return run_dir
    finally:
        CONFIG_PATH.write_text(original_config)


def main() -> None:
    original_config = CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_config)
    completed: list[str] = []
    try:
        for variant in VARIANTS:
            run_dir = _run_variant(base_cfg, original_config, variant)
            completed.append(str(run_dir.resolve()))
    finally:
        CONFIG_PATH.write_text(original_config)
    print("Friction-stopping v7/v8/v9 sweep finished.", flush=True)
    for run_dir in completed:
        print(f"  {run_dir}", flush=True)


if __name__ == "__main__":
    main()
