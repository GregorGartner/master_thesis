from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import run_r22_calibrated_all_checkpoints_eval as base
import run_r22_discrete_calibrated_uncertainty as r22


def _expected_checkpoint_count(
    *,
    seed: int,
    dirs: dict[str, dict[int, Path]],
    methods: list[str],
) -> int:
    return len(base._checkpoint_specs_for_seed(seed=seed, dirs=dirs, methods=methods))


def _sweep_dir_for_seed(
    *,
    seed: int,
    dirs: dict[str, dict[int, Path]],
    methods: list[str],
) -> Path:
    return dirs[methods[0]][seed] / f"r22_calibrated_uncertainty_seed{seed}" / "all_checkpoints"


def _is_complete(sweep_dir: Path, expected_rows: int) -> bool:
    scorecard = sweep_dir / "controller_scorecard.csv"
    if not scorecard.exists():
        return False
    with scorecard.open() as f:
        nonempty_lines = [line for line in f if line.strip()]
    return len(nonempty_lines) >= expected_rows + 1


def _requested_seeds(payload: dict[str, Any]) -> list[int]:
    raw = os.environ.get("R22_EVAL_SEEDS", "").strip()
    if raw:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    # This resume script is intentionally aimed at the interrupted tail of the run.
    return [2, 3]


def _write_combined_summaries(
    *,
    run_root: Path,
    payload: dict[str, Any],
    dirs: dict[str, dict[int, Path]],
    methods: list[str],
) -> dict[int, str]:
    all_sweeps: dict[int, str] = {}
    all_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for seed in [int(seed) for seed in payload["seeds"]]:
        expected = _expected_checkpoint_count(seed=seed, dirs=dirs, methods=methods)
        sweep_dir = _sweep_dir_for_seed(seed=seed, dirs=dirs, methods=methods)
        if not _is_complete(sweep_dir, expected):
            continue
        annotated, best = base._collect_best_rows(seed=seed, sweep_dir=sweep_dir, methods=methods)
        all_sweeps[seed] = str(sweep_dir.resolve())
        all_rows.extend(annotated)
        best_rows.extend(best)
        base._write_csv(run_root / f"all_checkpoint_best_by_method_seed{seed}.csv", best)

    base._write_csv(run_root / "all_checkpoint_scorecard.csv", all_rows)
    base._write_csv(run_root / "all_checkpoint_best_by_method.csv", best_rows)
    return all_sweeps


def main() -> None:
    run_root = base._run_root()
    payload = base._load_payload(run_root)
    methods = list(payload["methods"])
    dirs = base._dirs_from_payload(payload)
    seeds = _requested_seeds(payload)
    episodes_per_theta = int(os.environ.get("R22_EVAL_EPISODES_PER_THETA", "20"))

    for seed in seeds:
        expected = _expected_checkpoint_count(seed=seed, dirs=dirs, methods=methods)
        sweep_dir = _sweep_dir_for_seed(seed=seed, dirs=dirs, methods=methods)
        if _is_complete(sweep_dir, expected):
            print(f"SKIP seed {seed}: complete all-checkpoint sweep already exists at {sweep_dir}", flush=True)
            continue

        print(f"RESUME seed {seed}: running all-checkpoint sweep into {sweep_dir}", flush=True)
        specs = base._checkpoint_specs_for_seed(seed=seed, dirs=dirs, methods=methods)
        r22._run_sweep(
            target=dirs[methods[0]][seed],
            specs=specs,
            output_subdir=f"r22_calibrated_uncertainty_seed{seed}/all_checkpoints",
            episodes_per_theta=episodes_per_theta,
        )

    all_sweeps = _write_combined_summaries(
        run_root=run_root,
        payload=payload,
        dirs=dirs,
        methods=methods,
    )
    manifest = {
        "source_run_root": str(run_root.resolve()),
        "requested_resume_seeds": seeds,
        "completed_sweeps": all_sweeps,
        "episodes_per_theta": episodes_per_theta,
        "summary_files": [
            str((run_root / "all_checkpoint_scorecard.csv").resolve()),
            str((run_root / "all_checkpoint_best_by_method.csv").resolve()),
        ],
    }
    with (run_root / "all_checkpoint_eval_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved/resumed all-checkpoint summaries at: {run_root}", flush=True)


if __name__ == "__main__":
    main()
