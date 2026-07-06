from __future__ import annotations

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
    set_encoder_pretrain,
    set_gradual_policy,
    set_load,
    set_privileged,
    set_staged_policy,
    smoke_enabled,
    steps,
    timestamp,
)


PRIVILEGED_PHASES = [
    ("A_start_visible", "start_visible", True, 700_000, 0.04, 3e-4),
    ("B_easy", "easy", False, 1_000_000, 0.03, 3e-4),
    ("C_final", "final", False, 1_500_000, 0.015, 1e-4),
]

STAGED_ENCODER_STEPS = 1_200_000
STAGED_POLICY_PHASES = [
    ("S01", 0.01, 1e-4, 1_000_000),
    ("S02", 0.002, 1e-4, 1_000_000),
]

GRADUAL_PHASES = [
    ("P01", "easy", 0.25, 0.04, 3e-4, 700_000),
    ("P02", "easy", 0.50, 0.03, 3e-4, 800_000),
    ("P03", "final", 0.75, 0.02, 2e-4, 900_000),
    ("P04", "final", 1.00, 0.01, 1e-4, 1_200_000),
    ("P05", "final", 1.00, 0.002, 1e-4, 800_000),
]


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


def _staged_policy_phases() -> list[tuple[str, float, float, int]]:
    if smoke_enabled():
        return [STAGED_POLICY_PHASES[-1]]
    return STAGED_POLICY_PHASES


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
    )


def _train_privileged(base_cfg: dict, run_dir: Path) -> Path:
    exp_name = "privileged"
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
        run_training_stage(f"privileged/{phase_name}", cfg)
        _snapshot(exp_dir, phase_name)
        evaluate_model(
            exp_dir,
            run_dir / "evaluations" / f"privileged_{phase_name}",
            label=f"privileged_{phase_name}",
            seeds=3 if smoke_enabled() else 10,
        )
    return exp_dir


def _train_staged_vanilla(base_cfg: dict, run_dir: Path, privileged_dir: Path) -> Path:
    encoder_name = "staged_mle_encoder"
    policy_name = "staged_vanilla_rma"
    encoder_dir = _exp_dir(run_dir, encoder_name)
    policy_dir = _exp_dir(run_dir, policy_name)

    cfg = _stage_cfg(base_cfg, run_dir, encoder_name, total_timesteps=STAGED_ENCODER_STEPS, phase="final")
    set_encoder_pretrain(cfg, method="mle")
    set_load(cfg, privileged_dir)
    run_training_stage("staged/encoder_mle", cfg)
    evaluate_model(encoder_dir, run_dir / "evaluations" / "staged_encoder_mle", label="staged_encoder_mle", seeds=3 if smoke_enabled() else 10)

    for idx, (phase_name, ent, lr, raw_steps) in enumerate(_staged_policy_phases()):
        cfg = _stage_cfg(base_cfg, run_dir, policy_name, total_timesteps=raw_steps, phase="final")
        set_staged_policy(cfg, ent=ent, lr=lr)
        set_load(cfg, encoder_dir if idx == 0 else policy_dir)
        run_training_stage(f"staged/policy/{phase_name}", cfg)
        _snapshot(policy_dir, phase_name)
        evaluate_model(
            policy_dir,
            run_dir / "evaluations" / f"staged_vanilla_{phase_name}",
            label=f"staged_vanilla_{phase_name}",
            seeds=3 if smoke_enabled() else 10,
        )
    return policy_dir


def _train_gradual(
    base_cfg: dict,
    run_dir: Path,
    privileged_dir: Path,
    *,
    name: str,
    method: str,
    condition_on_uncertainty: bool,
) -> Path:
    exp_dir = _exp_dir(run_dir, name)
    for idx, (phase_name, env_phase, enc_prob, ent, lr, raw_steps) in enumerate(_gradual_phases()):
        cfg = _stage_cfg(base_cfg, run_dir, name, total_timesteps=raw_steps, phase=env_phase)
        set_gradual_policy(
            cfg,
            method=method,
            condition_on_uncertainty=condition_on_uncertainty,
            encoder_probability=enc_prob,
            ent=ent,
            lr=lr,
            freeze_encoder=(phase_name == "P05"),
        )
        set_load(cfg, privileged_dir if idx == 0 else exp_dir)
        run_training_stage(f"{name}/{phase_name}", cfg)
        _snapshot(exp_dir, phase_name)
        evaluate_model(
            exp_dir,
            run_dir / "evaluations" / f"{name}_{phase_name}",
            label=f"{name}_{phase_name}",
            seeds=3 if smoke_enabled() else 10,
        )
    return exp_dir


def main() -> None:
    original_config = CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_config)
    stamp = timestamp()
    run_dir = OUT_ROOT / f"friction_stopping_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "latest_friction_stopping.txt").write_text(str(run_dir.resolve()) + "\n")

    payload = {
        "timestamp": stamp,
        "run_dir": str(run_dir.resolve()),
        "smoke": smoke_enabled(),
        "stages": [],
        "mu_values": [0.25, 0.5, 0.7],
    }
    save_json(run_dir / "run_manifest.json", payload)

    try:
        print(f"Writing friction-stopping outputs to {run_dir}", flush=True)
        run_sanity_checks(run_dir / "sanity")
        evaluate_handcoded(run_dir / "handcoded", seeds=2 if smoke_enabled() else 20)

        privileged_dir = _train_privileged(base_cfg, run_dir)
        _train_staged_vanilla(base_cfg, run_dir, privileged_dir)
        _train_gradual(
            base_cfg,
            run_dir,
            privileged_dir,
            name="gradual_mle_mean_only",
            method="mle",
            condition_on_uncertainty=False,
        )
        _train_gradual(
            base_cfg,
            run_dir,
            privileged_dir,
            name="gradual_nll_mean_only",
            method="nll",
            condition_on_uncertainty=False,
        )
        _train_gradual(
            base_cfg,
            run_dir,
            privileged_dir,
            name="gradual_nll_mean_std",
            method="nll",
            condition_on_uncertainty=True,
        )
        print(f"Friction-stopping run complete: {run_dir}", flush=True)
    finally:
        CONFIG_PATH.write_text(original_config)


if __name__ == "__main__":
    main()
