from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

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
)


ENV_OVERRIDES = {
    "max_episode_steps": 300,
    "event_time_low": 80,
    "event_time_high": 260,
    "initial_velocity_low": 2.0,
    "initial_velocity_high": 3.0,
    "brake_accel_max": 8.0,
    "throttle_accel_max": 8.0,
    "saturate_throttle_by_friction": True,
    "action_cost_weight": 0.3,
    "safety_cost_weight": 10.0,
    "crash_penalty": 150.0,
    "crash_remaining_penalty": 150.0,
}

BASE_MODEL_OVERRIDES = {
    "window_length": 30,
    "nominal_warmup_steps": 0,
    "id_update_interval": 1,
}

PRIVILEGED_PHASES = [
    ("A_start_visible", "start_visible", True, 1_200_000, 0.04, 3e-4),
    ("B_easy", "easy", False, 1_500_000, 0.03, 3e-4),
    ("C_medium", "medium", False, 2_000_000, 0.02, 2e-4),
    ("D_hard_final", "final", False, 4_000_000, 0.01, 1e-4),
]

GRADUAL_PHASES = [
    ("P01", "easy", 0.25, 0.04, 3e-4, 1_100_000),
    ("P02", "medium", 0.50, 0.03, 3e-4, 1_200_000),
    ("P03", "final", 0.75, 0.02, 2e-4, 1_500_000),
    ("P04", "final", 1.00, 0.01, 1e-4, 2_000_000),
    ("P05", "final", 1.00, 0.002, 1e-4, 1_300_000),
]

PRIVILEGED_GATE_MAX_CRASH_RATE = 0.05
PRIVILEGED_GATE_MIN_SUCCESS_RATE = 0.95


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


def _privileged_phases() -> list[tuple[str, str, bool, int, float, float]]:
    if smoke_enabled():
        return [PRIVILEGED_PHASES[0], PRIVILEGED_PHASES[-1]]
    return PRIVILEGED_PHASES


def _gradual_phases() -> list[tuple[str, str, float, float, float, int]]:
    if smoke_enabled():
        return [GRADUAL_PHASES[0], GRADUAL_PHASES[-1]]
    return GRADUAL_PHASES


def _stage_cfg(
    base_cfg: dict,
    run_dir: Path,
    exp_name: str,
    *,
    total_timesteps: int,
    phase: str,
    start_visible: bool = False,
) -> dict:
    return make_base_cfg(
        base_cfg,
        exp_root=run_dir,
        exp_name=exp_name,
        total_timesteps=steps(total_timesteps),
        phase=phase,
        start_visible=start_visible,
        env_overrides=ENV_OVERRIDES,
        model_overrides=BASE_MODEL_OVERRIDES,
    )


def _train_privileged(base_cfg: dict, run_dir: Path) -> Path:
    exp_name = "privileged_v4"
    exp_dir = _exp_dir(run_dir, exp_name)
    for idx, (phase_name, env_phase, start_visible, raw_steps, ent, lr) in enumerate(_privileged_phases()):
        cfg = _stage_cfg(
            base_cfg,
            run_dir,
            exp_name,
            total_timesteps=raw_steps,
            phase=env_phase,
            start_visible=start_visible,
        )
        set_privileged(cfg, ent=ent, lr=lr)
        set_load(cfg, exp_dir if idx > 0 else None)
        run_training_stage(f"v4 privileged/{phase_name}", cfg)
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
    passed = crash_rate <= PRIVILEGED_GATE_MAX_CRASH_RATE and success_rate >= PRIVILEGED_GATE_MIN_SUCCESS_RATE
    payload.update(
        {
            "crash_rate": crash_rate,
            "success_rate": success_rate,
            "passed": bool(passed),
            "reason": "passed" if passed else "privileged_not_reliable_enough",
        }
    )
    return payload


def _train_gradual_nll(
    base_cfg: dict,
    run_dir: Path,
    privileged_dir: Path,
    *,
    name: str,
    condition_on_uncertainty: bool,
) -> Path:
    exp_dir = _exp_dir(run_dir, name)
    for idx, (phase_name, env_phase, enc_prob, ent, lr, raw_steps) in enumerate(_gradual_phases()):
        cfg = _stage_cfg(base_cfg, run_dir, name, total_timesteps=raw_steps, phase=env_phase)
        set_gradual_policy(
            cfg,
            method="nll",
            condition_on_uncertainty=condition_on_uncertainty,
            encoder_probability=enc_prob,
            ent=ent,
            lr=lr,
            freeze_encoder=(phase_name == "P05"),
        )
        params = cfg["model"]["params"]
        params["initial_context"] = [0.5, 0.2] if condition_on_uncertainty else [0.5]
        set_load(cfg, privileged_dir if idx == 0 else exp_dir)
        run_training_stage(f"v4 {name}/{phase_name}", cfg)
        _snapshot(exp_dir, phase_name)
        evaluate_model(
            exp_dir,
            run_dir / "evaluations" / f"{name}_{phase_name}",
            label=f"{name}_{phase_name}",
            seeds=3 if smoke_enabled() else 12,
        )
    return exp_dir


def main() -> None:
    original_config = CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_config)
    stamp = timestamp()
    run_dir = OUT_ROOT / f"friction_stopping_v4_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "latest_friction_stopping_v4.txt").write_text(str(run_dir.resolve()) + "\n")

    save_json(
        run_dir / "run_manifest.json",
        {
            "timestamp": stamp,
            "run_dir": str(run_dir.resolve()),
            "smoke": smoke_enabled(),
            "version": "v4",
            "mu_values": [0.25, 0.5, 0.7],
            "env_overrides": ENV_OVERRIDES,
            "model_overrides": BASE_MODEL_OVERRIDES,
            "final_visible_distance": [0.9, 1.2],
            "pre_probe_cost_weight": 0.0,
            "focused_methods": ["gradual_nll_mean_only_v4", "gradual_nll_mean_std_v4"],
            "privileged_gate": {
                "max_crash_rate": PRIVILEGED_GATE_MAX_CRASH_RATE,
                "min_success_rate": PRIVILEGED_GATE_MIN_SUCCESS_RATE,
                "ignore_env_var": "FRICTION_STOPPING_V4_IGNORE_GATE",
            },
        },
    )

    try:
        print(f"Writing friction-stopping v4 outputs to {run_dir}", flush=True)
        run_sanity_checks(run_dir / "sanity", env_overrides=ENV_OVERRIDES)
        evaluate_handcoded(run_dir / "handcoded", seeds=2 if smoke_enabled() else 20, phase="final", env_overrides=ENV_OVERRIDES)

        privileged_dir = _train_privileged(base_cfg, run_dir)
        gate_payload = _privileged_gate(run_dir)
        save_json(run_dir / "privileged_gate.json", gate_payload)
        print(f"Friction-stopping v4 privileged gate: {gate_payload}", flush=True)

        ignore_gate = os.environ.get("FRICTION_STOPPING_V4_IGNORE_GATE", "0").lower() in {"1", "true", "yes"}
        if not gate_payload.get("passed", False) and not smoke_enabled() and not ignore_gate:
            print(
                "Stopping v4 before adaptive training because privileged PPO did not pass the reliability gate. "
                "Set FRICTION_STOPPING_V4_IGNORE_GATE=1 to override.",
                flush=True,
            )
            return

        mean_only_dir = _train_gradual_nll(
            base_cfg,
            run_dir,
            privileged_dir,
            name="gradual_nll_mean_only_v4",
            condition_on_uncertainty=False,
        )
        mean_std_dir = _train_gradual_nll(
            base_cfg,
            run_dir,
            privileged_dir,
            name="gradual_nll_mean_std_v4",
            condition_on_uncertainty=True,
        )

        evaluations = run_dir / "evaluations"
        write_final_comparison_diagnostics(
            run_dir,
            model_evaluations={
                "privileged": evaluations / "privileged_D_hard_final",
                "mean-only": evaluations / "gradual_nll_mean_only_v4_P05",
                "mean+std": evaluations / "gradual_nll_mean_std_v4_P05",
            },
            mean_only_eval_dir=evaluations / "gradual_nll_mean_only_v4_P05",
            mean_std_eval_dir=evaluations / "gradual_nll_mean_std_v4_P05",
        )

        print(f"Friction-stopping v4 run complete: {run_dir}", flush=True)
        print(f"Trained directories: privileged={privileged_dir}, mean_only={mean_only_dir}, mean_std={mean_std_dir}", flush=True)
    finally:
        CONFIG_PATH.write_text(original_config)


if __name__ == "__main__":
    main()
