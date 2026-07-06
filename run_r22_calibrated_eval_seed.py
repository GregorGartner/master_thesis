from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import run_r22_discrete_calibrated_uncertainty as r22


def _run_root() -> Path:
    explicit = os.environ.get("R22_EVAL_RUN_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    pointer = r22.POINTER
    if not pointer.exists():
        candidates = sorted(
            r22.RUN_FAMILY_ROOT.glob("calibrated_uncertainty_*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0].resolve()
        raise FileNotFoundError(
            f"Missing pointer {pointer} and no calibrated_uncertainty_* run found. "
            "Set R22_EVAL_RUN_ROOT to the calibrated run directory."
        )
    return Path(pointer.read_text().strip()).expanduser().resolve()


def _selected_seeds(payload: dict[str, Any]) -> list[int]:
    raw = os.environ.get("R22_EVAL_SEEDS", "1").strip()
    if raw:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    return [int(seed) for seed in payload.get("seeds", [])]


def _load_payload(run_root: Path) -> dict[str, Any]:
    path = run_root / "calibrated_uncertainty_run.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run manifest: {path}")
    with path.open() as f:
        return json.load(f)


def _dirs_from_payload(payload: dict[str, Any]) -> dict[str, dict[int, Path]]:
    dirs: dict[str, dict[int, Path]] = {}
    for method, by_seed in payload["dirs"].items():
        dirs[method] = {int(seed): Path(path).expanduser().resolve() for seed, path in by_seed.items()}
    return dirs


def _ensure_ready(seed: int, methods: list[str], dirs: dict[str, dict[int, Path]]) -> None:
    missing: list[str] = []
    for method in methods:
        exp_dir = dirs[method][seed]
        weights = exp_dir / f"{r22._final_weights_name(method)}.zip"
        config = exp_dir / "config.yaml"
        if not config.exists():
            missing.append(str(config))
        if not weights.exists():
            missing.append(str(weights))
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"Seed {seed} is not ready for evaluation. Missing:\n  {joined}")


def main() -> None:
    run_root = _run_root()
    payload = _load_payload(run_root)
    methods = list(payload["methods"])
    dirs = _dirs_from_payload(payload)
    seeds = _selected_seeds(payload)
    episodes_per_theta = int(os.environ.get("R22_EVAL_EPISODES_PER_THETA", "20"))

    prediction_sweeps: dict[int, Path] = {}
    ablation_sweeps: dict[int, Path] = {}
    constants_by_seed: dict[int, dict[str, float]] = {}
    summary_root = run_root / ("early_eval_" + "_".join(f"seed{seed}" for seed in seeds))
    summary_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        _ensure_ready(seed, methods, dirs)
        target = dirs[methods[0]][seed]
        prediction_sweeps[seed] = r22._run_sweep(
            target=target,
            specs=r22._prediction_specs_for_seed(seed, dirs, methods),
            output_subdir=f"r22_calibrated_uncertainty_seed{seed}_early/predicted_final",
            episodes_per_theta=episodes_per_theta,
        )
        constants = r22._constant_uncertainties_from_prediction_sweep(
            prediction_sweeps[seed],
            dirs,
            methods,
            seed,
        )
        constants_by_seed[seed] = constants
        ablation_sweeps[seed] = r22._run_sweep(
            target=target,
            specs=r22._ablation_specs_for_seed(seed, dirs, methods, constants),
            output_subdir=f"r22_calibrated_uncertainty_seed{seed}_early/uncertainty_ablations",
            episodes_per_theta=episodes_per_theta,
        )

    rows = r22._collect_scorecard_rows(summary_root, ablation_sweeps)
    aggregate = r22._aggregate_scorecards(summary_root, rows)
    success = r22._success_summary(summary_root, rows, methods, seeds)
    r22._plot_results(summary_root, rows)

    early_payload = {
        "source_run_root": str(run_root),
        "seeds": seeds,
        "methods": methods,
        "episodes_per_theta": episodes_per_theta,
        "predicted_sweeps": {seed: str(path.resolve()) for seed, path in prediction_sweeps.items()},
        "uncertainty_ablations": {seed: str(path.resolve()) for seed, path in ablation_sweeps.items()},
        "matched_constant_uncertainty_scaled": constants_by_seed,
        "aggregate_scorecard": aggregate,
        "success_summary": success,
    }
    with (summary_root / "early_eval_manifest.json").open("w") as f:
        json.dump(early_payload, f, indent=2)
    print(f"Saved early R22 calibrated evaluation to: {summary_root}", flush=True)


if __name__ == "__main__":
    main()
