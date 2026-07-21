from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import run_r22_discrete_calibrated_uncertainty as r22


def _run_root() -> Path:
    explicit = os.environ.get("R22_EVAL_RUN_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    if r22.POINTER.exists():
        return Path(r22.POINTER.read_text().strip()).expanduser().resolve()
    candidates = sorted(
        r22.RUN_FAMILY_ROOT.glob("calibrated_uncertainty_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Missing pointer {r22.POINTER} and no calibrated_uncertainty_* run found."
        )
    return candidates[0].resolve()


def _load_payload(run_root: Path) -> dict[str, Any]:
    path = run_root / "calibrated_uncertainty_run.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run manifest: {path}")
    with path.open() as f:
        return json.load(f)


def _selected_seeds(payload: dict[str, Any]) -> list[int]:
    raw = os.environ.get("R22_EVAL_SEEDS", "").strip()
    if raw:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    return [int(seed) for seed in payload["seeds"]]


def _dirs_from_payload(payload: dict[str, Any]) -> dict[str, dict[int, Path]]:
    return {
        method: {int(seed): Path(path).expanduser().resolve() for seed, path in by_seed.items()}
        for method, by_seed in payload["dirs"].items()
    }


def _checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    stem = path.stem
    if stem.startswith("weights_P"):
        prefix = 0
        suffix = stem.removeprefix("weights_P")
    elif stem.startswith("weights_CAL"):
        prefix = 1
        suffix = stem.removeprefix("weights_CAL")
    elif stem.startswith("weights_FT"):
        prefix = 2
        suffix = stem.removeprefix("weights_FT")
    else:
        prefix = 9
        suffix = stem
    try:
        number = int(suffix)
    except ValueError:
        number = 999
    return prefix, number, stem


def _checkpoint_specs_for_seed(
    *,
    seed: int,
    dirs: dict[str, dict[int, Path]],
    methods: list[str],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for method in methods:
        exp_dir = dirs[method][seed]
        for weights in sorted(exp_dir.glob("weights_*.zip"), key=_checkpoint_sort_key):
            if weights.name.startswith("weights_best"):
                continue
            phase = weights.stem.removeprefix("weights_")
            specs.append(
                r22._ppo_spec(
                    f"{method}_{phase}_seed{seed}",
                    exp_dir,
                    weights.stem,
                )
            )
    if not specs:
        raise FileNotFoundError(f"No phase checkpoints found for seed {seed}.")
    return specs


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _method_and_phase(controller: str, methods: list[str], seed: int) -> tuple[str, str]:
    suffix = f"_seed{seed}"
    base = controller.removesuffix(suffix)
    for method in sorted(methods, key=len, reverse=True):
        prefix = f"{method}_"
        if base.startswith(prefix):
            return method, base[len(prefix) :]
    return "", ""


def _collect_best_rows(
    *,
    seed: int,
    sweep_dir: Path,
    methods: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotated: list[dict[str, Any]] = []
    best_by_method: dict[str, dict[str, Any]] = {}
    for row in _read_csv(sweep_dir / "controller_scorecard.csv"):
        method, phase = _method_and_phase(row["controller"], methods, seed)
        if not method:
            continue
        out = {
            "seed": seed,
            "method": method,
            "phase": phase,
            "controller": row["controller"],
            "mean_return": float(row["mean_return"]),
            "tail_mean_return": float(row["tail_mean_return"]),
            "center_mean_return": float(row["center_mean_return"]),
            "theta_rmse": float(row["theta_rmse"]) if row.get("theta_rmse") else "",
            "tail_theta_rmse": float(row["tail_theta_rmse"]) if row.get("tail_theta_rmse") else "",
            "center_theta_rmse": float(row["center_theta_rmse"]) if row.get("center_theta_rmse") else "",
        }
        annotated.append(out)
        if method not in best_by_method or out["mean_return"] > best_by_method[method]["mean_return"]:
            best_by_method[method] = out
    return annotated, [best_by_method[method] for method in sorted(best_by_method)]


def main() -> None:
    run_root = _run_root()
    payload = _load_payload(run_root)
    methods = list(payload["methods"])
    dirs = _dirs_from_payload(payload)
    seeds = _selected_seeds(payload)
    episodes_per_theta = int(os.environ.get("R22_EVAL_EPISODES_PER_THETA", "20"))

    all_sweeps: dict[int, str] = {}
    all_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for seed in seeds:
        target = dirs[methods[0]][seed]
        specs = _checkpoint_specs_for_seed(seed=seed, dirs=dirs, methods=methods)
        sweep_dir = r22._run_sweep(
            target=target,
            specs=specs,
            output_subdir=f"r22_calibrated_uncertainty_seed{seed}/all_checkpoints",
            episodes_per_theta=episodes_per_theta,
        )
        all_sweeps[seed] = str(sweep_dir.resolve())
        annotated, best = _collect_best_rows(seed=seed, sweep_dir=sweep_dir, methods=methods)
        all_rows.extend(annotated)
        best_rows.extend(best)
        _write_csv(
            run_root / f"all_checkpoint_best_by_method_seed{seed}.csv",
            best,
        )

    _write_csv(run_root / "all_checkpoint_scorecard.csv", all_rows)
    _write_csv(run_root / "all_checkpoint_best_by_method.csv", best_rows)
    manifest = {
        "source_run_root": str(run_root.resolve()),
        "seeds": seeds,
        "methods": methods,
        "episodes_per_theta": episodes_per_theta,
        "sweeps": all_sweeps,
        "summary_files": [
            str((run_root / "all_checkpoint_scorecard.csv").resolve()),
            str((run_root / "all_checkpoint_best_by_method.csv").resolve()),
        ],
    }
    with (run_root / "all_checkpoint_eval_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved all-checkpoint R22 calibrated evaluation to: {run_root}", flush=True)


if __name__ == "__main__":
    main()
