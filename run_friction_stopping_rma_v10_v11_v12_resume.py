from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

import run_friction_stopping_rma_v10_v11_v12 as base


def _variant(version: str) -> base.Variant:
    for variant in base.VARIANTS:
        if variant.version == version:
            return variant
    raise ValueError(f"Unknown variant: {version}")


def _run_dir_for_variant(version: str) -> Path:
    raw = os.environ.get(f"FRICTION_STOPPING_{version.upper()}_RESUME_RUN_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    pointer = base.OUT_ROOT / f"latest_friction_stopping_{version}.txt"
    if not pointer.exists():
        raise FileNotFoundError(
            f"No latest {version} pointer found. Set FRICTION_STOPPING_{version.upper()}_RESUME_RUN_DIR explicitly."
        )
    return Path(pointer.read_text().strip()).resolve()


def _phase_weight(exp_dir: Path, phase: str) -> Path:
    return exp_dir / f"weights_{phase}.zip"


def _phase_done(exp_dir: Path, phase: str) -> bool:
    return _phase_weight(exp_dir, phase).exists()


def _eval_dir(run_dir: Path, name: str, phase: str) -> Path:
    return run_dir / "evaluations" / f"{name}_{phase}"


def _evaluate_if_missing(exp_dir: Path, eval_dir: Path, label: str) -> None:
    if (eval_dir / "scorecard.csv").exists():
        return
    base.evaluate_model(
        exp_dir,
        eval_dir,
        label=label,
        seeds=3 if base.smoke_enabled() else 12,
    )


def _set_adaptive_phase_config(
    cfg: dict,
    variant: base.Variant,
    *,
    phase_kind: str,
    condition_on_uncertainty: bool,
    encoder_probability: float,
    ent: float,
    lr: float,
    freeze_encoder: bool,
) -> None:
    if phase_kind == "calibration":
        base._set_calibration_policy(
            cfg,
            variant,
            condition_on_uncertainty=condition_on_uncertainty,
            encoder_probability=encoder_probability,
            lr=lr,
        )
    else:
        base._set_policy_phase(
            cfg,
            condition_on_uncertainty=condition_on_uncertainty,
            encoder_probability=encoder_probability,
            ent=ent,
            lr=lr,
            freeze_encoder=freeze_encoder,
        )
    cfg["model"]["params"]["initial_context"] = [0.5, 0.2] if condition_on_uncertainty else [0.5]


def _train_adaptive_nll_resume(
    base_cfg: dict[str, Any],
    run_dir: Path,
    privileged_dir: Path,
    variant: base.Variant,
    *,
    name: str,
    condition_on_uncertainty: bool,
) -> Path:
    exp_dir = base._exp_dir(run_dir, name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    for idx, (phase_name, phase_kind, env_phase, enc_prob, ent, lr, raw_steps, freeze_encoder) in enumerate(
        base._adaptive_phases(variant)
    ):
        eval_dir = _eval_dir(run_dir, name, phase_name)
        if _phase_done(exp_dir, phase_name):
            print(f"SKIP {variant.version} {name}/{phase_name}: {_phase_weight(exp_dir, phase_name)} exists", flush=True)
            _evaluate_if_missing(exp_dir, eval_dir, f"{name}_{phase_name}")
            continue

        cfg = base._stage_cfg(
            base_cfg,
            run_dir,
            name,
            variant,
            total_timesteps=raw_steps,
            phase=env_phase,
        )
        _set_adaptive_phase_config(
            cfg,
            variant,
            phase_kind=phase_kind,
            condition_on_uncertainty=condition_on_uncertainty,
            encoder_probability=enc_prob,
            ent=ent,
            lr=lr,
            freeze_encoder=freeze_encoder,
        )
        load_source = privileged_dir if idx == 0 and not (exp_dir / "weights.zip").exists() else exp_dir
        base.set_load(cfg, load_source)
        base.run_training_stage(f"{variant.version} {name}/{phase_name} [resume]", cfg)
        base._snapshot(exp_dir, phase_name)
        base.evaluate_model(
            exp_dir,
            eval_dir,
            label=f"{name}_{phase_name}",
            seeds=3 if base.smoke_enabled() else 12,
        )
    return exp_dir


def _resume_variant(base_cfg: dict[str, Any], variant: base.Variant) -> Path:
    run_dir = _run_dir_for_variant(variant.version)
    if not run_dir.exists():
        raise FileNotFoundError(f"{variant.version} run directory does not exist: {run_dir}")
    base.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (base.OUT_ROOT / f"latest_friction_stopping_{variant.version}.txt").write_text(str(run_dir.resolve()) + "\n")

    privileged_dir = base._exp_dir(run_dir, f"privileged_{variant.version}")
    if not (privileged_dir / "weights_D_hard_final.zip").exists():
        raise FileNotFoundError(
            f"Cannot resume adaptive {variant.version} because privileged final checkpoint is missing: {privileged_dir}"
        )

    gate_payload = base._privileged_gate(run_dir)
    base.save_json(run_dir / "privileged_gate.json", gate_payload)
    print(f"Resuming {variant.version} from {run_dir}", flush=True)
    print(f"{variant.version} privileged gate: {gate_payload}", flush=True)
    if not gate_payload.get("passed", False) and not base.smoke_enabled() and not base._ignore_gate(variant):
        raise RuntimeError(
            f"{variant.version} privileged gate does not pass. Set FRICTION_STOPPING_{variant.version.upper()}_IGNORE_GATE=1 "
            "or FRICTION_STOPPING_SWEEP_IGNORE_GATE=1 to override."
        )

    _train_adaptive_nll_resume(
        base_cfg,
        run_dir,
        privileged_dir,
        variant,
        name=f"gradual_nll_mean_only_{variant.version}",
        condition_on_uncertainty=False,
    )
    _train_adaptive_nll_resume(
        base_cfg,
        run_dir,
        privileged_dir,
        variant,
        name=f"gradual_nll_mean_std_{variant.version}",
        condition_on_uncertainty=True,
    )
    base._write_phase_probe_contexts(run_dir, variant)
    base._write_checkpoint_comparisons(run_dir, variant)
    print(f"Completed resumed {variant.version}: {run_dir}", flush=True)
    return run_dir


def main() -> None:
    original_config = base.CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_config)
    completed: list[str] = []
    try:
        for variant in base.VARIANTS:
            run_dir = _resume_variant(base_cfg, variant)
            completed.append(str(run_dir.resolve()))
    finally:
        base.CONFIG_PATH.write_text(original_config)

    print("Friction-stopping v10/v11/v12 resume sweep finished.", flush=True)
    for run_dir in completed:
        print(f"  {run_dir}", flush=True)


if __name__ == "__main__":
    main()
