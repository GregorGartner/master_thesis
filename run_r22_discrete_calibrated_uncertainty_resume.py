from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

import run_r22_discrete_calibrated_uncertainty as base


ROOT = Path(__file__).resolve().parent


def _run_root_from_env_or_latest() -> Path:
    raw = os.environ.get("R22_CALIBRATED_RESUME_RUN_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()

    candidates = sorted(
        base.RUN_FAMILY_ROOT.glob("calibrated_uncertainty_*_r22_1p5"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No calibrated uncertainty run found. Set R22_CALIBRATED_RESUME_RUN_ROOT explicitly."
        )
    return candidates[0].resolve()


def _load_manifest(run_root: Path) -> dict[str, Any]:
    path = run_root / "calibrated_uncertainty_run.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open() as f:
        return json.load(f)


def _write_manifest(run_root: Path, payload: dict[str, Any]) -> None:
    base._write_manifest(run_root, payload)


def _dir_map(payload: dict[str, Any]) -> dict[str, dict[int, Path]]:
    return {
        method: {int(seed): Path(path).resolve() for seed, path in by_seed.items()}
        for method, by_seed in payload["dirs"].items()
    }


def _name_map(payload: dict[str, Any]) -> dict[str, dict[int, str]]:
    return {
        method: {int(seed): str(name) for seed, name in by_seed.items()}
        for method, by_seed in payload["experiments"].items()
    }


def _template_config(payload: dict[str, Any], dirs: dict[str, dict[int, Path]]) -> dict[str, Any]:
    for method in payload["methods"]:
        for seed in payload["seeds"]:
            cfg_path = dirs[method][int(seed)] / "config.yaml"
            if cfg_path.exists():
                with cfg_path.open() as f:
                    return yaml.safe_load(f)
    with base.base.CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def _gradual_row(phase: str, encoder_prob: float, ent: float, lr: float, steps: int) -> dict[str, Any]:
    return {
        "kind": "gradual",
        "phase": phase,
        "encoder_probability": encoder_prob,
        "ent_coef": ent,
        "learning_rate": lr,
        "timesteps": base._steps(steps),
    }


def _calibration_row(after_phase: str, encoder_prob: float) -> dict[str, Any]:
    cal_phase, cal_steps = base.CALIBRATION_AFTER[after_phase]
    return {
        "kind": "calibration_refresh",
        "phase": cal_phase,
        "after_phase": after_phase,
        "encoder_probability": encoder_prob,
        "timesteps": base._steps(cal_steps),
        "naive_action_noise_std": list(base.CALIBRATION_NOISE_STD),
        "naive_action_noise_dist": list(base.CALIBRATION_NOISE_DIST),
    }


def _finetune_row(phase: str, encoder_prob: float, ent: float, lr: float, steps: int) -> dict[str, Any]:
    return {
        "kind": "policy_finetune",
        "phase": phase,
        "encoder_probability": encoder_prob,
        "ent_coef": ent,
        "learning_rate": lr,
        "timesteps": base._steps(steps),
    }


def _completed_calibrated_ft_rows(include_ft02: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase, encoder_prob, ent, lr, steps in base.GRADUAL_PHASES:
        rows.append(_gradual_row(phase, encoder_prob, ent, lr, steps))
        if phase in base.CALIBRATION_AFTER:
            rows.append(_calibration_row(phase, encoder_prob))

    for phase, encoder_prob, ent, lr, steps in base.FINE_TUNE_PHASES:
        if phase == "FT02" and not include_ft02:
            break
        rows.append(_finetune_row(phase, encoder_prob, ent, lr, steps))
    return rows


def _require_checkpoint(exp_dir: Path, weights_name: str) -> None:
    path = exp_dir / f"{weights_name}.zip"
    if not path.exists():
        raise FileNotFoundError(f"Missing required checkpoint: {path}")


def _run_seed2_ft02(
    *,
    run_root: Path,
    payload: dict[str, Any],
    base_cfg: dict[str, Any],
    exp_root: Path,
    dirs: dict[str, dict[int, Path]],
    names: dict[str, dict[int, str]],
) -> None:
    method = "gradual_nll_calibrated_ft"
    seed = 2
    exp_dir = dirs[method][seed]
    _require_checkpoint(exp_dir, "weights_FT01")
    if (exp_dir / "weights_FT02.zip").exists() and not os.environ.get("R22_FORCE_RERUN_FT02"):
        print("seed2 gradual_nll_calibrated_ft FT02 already has weights_FT02.zip; skipping rerun.", flush=True)
        payload.setdefault("phase_logs", {}).setdefault(method, {})[str(seed)] = (
            _completed_calibrated_ft_rows(include_ft02=True)
        )
        _write_manifest(run_root, payload)
        return

    ft_phase, encoder_prob, ent, lr, steps = base.FINE_TUNE_PHASES[-1]
    if ft_phase != "FT02":
        raise RuntimeError(f"Expected final fine-tune phase FT02, got {ft_phase}.")

    cfg = base._base_cfg(base_cfg, exp_root, names[method][seed], steps, payload["system"])
    base._set_policy_finetune(cfg, seed=seed, encoder_probability=encoder_prob, ent=ent, lr=lr)
    base._set_load(cfg, exp_dir, weights_name="weights_FT01")
    base._run_stage(f"{method}/seed{seed}/{ft_phase} [resume from FT01]", cfg)
    base.base._snapshot_phase_weights(exp_dir, ft_phase)

    payload.setdefault("phase_logs", {}).setdefault(method, {})[str(seed)] = (
        _completed_calibrated_ft_rows(include_ft02=True)
    )
    _write_manifest(run_root, payload)


def _train_missing_seed3(
    *,
    run_root: Path,
    payload: dict[str, Any],
    base_cfg: dict[str, Any],
    exp_root: Path,
    dirs: dict[str, dict[int, Path]],
    names: dict[str, dict[int, str]],
) -> None:
    seed = 3
    for method in payload["methods"]:
        final_weights = base._final_weights_name(method)
        exp_dir = dirs[method][seed]
        if (exp_dir / f"{final_weights}.zip").exists() and not os.environ.get("R22_FORCE_RERUN_SEED3"):
            print(f"seed3 {method} already has {final_weights}.zip; skipping training.", flush=True)
            continue
        phase_rows = base._train_method(
            method=method,
            seed=seed,
            base_cfg=base_cfg,
            exp_root=exp_root,
            exp_name=names[method][seed],
            exp_dir=exp_dir,
            system=payload["system"],
        )
        payload.setdefault("phase_logs", {}).setdefault(method, {})[str(seed)] = phase_rows
        _write_manifest(run_root, payload)


def _run_final_evaluation(
    *,
    payload: dict[str, Any],
    run_root: Path,
    dirs: dict[str, dict[int, Path]],
) -> None:
    methods = list(payload["methods"])
    seeds = [int(seed) for seed in payload["seeds"]]
    episodes_per_theta = 1 if base._smoke_enabled() else 20

    prediction_sweeps: dict[int, Path] = {}
    ablation_sweeps: dict[int, Path] = {}
    constants_by_seed: dict[int, dict[str, float]] = {}
    for seed in seeds:
        target = dirs[methods[0]][seed]
        prediction_sweeps[seed] = base._run_sweep(
            target=target,
            specs=base._prediction_specs_for_seed(seed, dirs, methods),
            output_subdir=f"r22_calibrated_uncertainty_seed{seed}/predicted_final",
            episodes_per_theta=episodes_per_theta,
        )
        constants = base._constant_uncertainties_from_prediction_sweep(
            prediction_sweeps[seed],
            dirs,
            methods,
            seed,
        )
        constants_by_seed[seed] = constants
        ablation_sweeps[seed] = base._run_sweep(
            target=target,
            specs=base._ablation_specs_for_seed(seed, dirs, methods, constants),
            output_subdir=f"r22_calibrated_uncertainty_seed{seed}/uncertainty_ablations",
            episodes_per_theta=episodes_per_theta,
        )

    payload["sweeps"] = {
        "predicted": {seed: str(path.resolve()) for seed, path in prediction_sweeps.items()},
        "uncertainty_ablations": {seed: str(path.resolve()) for seed, path in ablation_sweeps.items()},
    }
    payload["matched_constant_uncertainty_scaled"] = constants_by_seed
    rows = base._collect_scorecard_rows(run_root, ablation_sweeps)
    aggregate = base._aggregate_scorecards(run_root, rows)
    success = base._success_summary(run_root, rows, methods, seeds)
    base._plot_results(run_root, rows)
    payload["aggregate_scorecard"] = aggregate
    payload["success_summary"] = success
    _write_manifest(run_root, payload)
    if not base._smoke_enabled():
        base.POINTER.write_text(str(run_root.resolve()) + "\n")


def main() -> None:
    run_root = _run_root_from_env_or_latest()
    payload = _load_manifest(run_root)

    if payload.get("seeds") != [1, 2, 3]:
        raise RuntimeError(f"Expected seeds [1, 2, 3], got {payload.get('seeds')}.")
    if payload.get("methods") != base.METHODS:
        raise RuntimeError(f"Expected methods {base.METHODS}, got {payload.get('methods')}.")

    dirs = _dir_map(payload)
    names = _name_map(payload)
    base_cfg = _template_config(payload, dirs)
    exp_root = base.base._resolve_exp_root(base_cfg)

    original_text = base.base.CONFIG_PATH.read_text()
    try:
        _run_seed2_ft02(
            run_root=run_root,
            payload=payload,
            base_cfg=base_cfg,
            exp_root=exp_root,
            dirs=dirs,
            names=names,
        )
        _train_missing_seed3(
            run_root=run_root,
            payload=payload,
            base_cfg=base_cfg,
            exp_root=exp_root,
            dirs=dirs,
            names=names,
        )
    finally:
        base.base.CONFIG_PATH.write_text(original_text)

    _write_manifest(run_root, payload)
    _run_final_evaluation(payload=payload, run_root=run_root, dirs=dirs)
    print(f"Resumed and completed R22 calibrated uncertainty run: {run_root}", flush=True)


if __name__ == "__main__":
    main()
