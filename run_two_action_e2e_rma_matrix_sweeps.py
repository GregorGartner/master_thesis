from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
SWEEP_SCRIPT = ROOT / "lqr_theta_sweep_eval.py"

DEFAULT_WEIGHTS = ("weights_P01", "weights_P02", "weights_P03", "weights_P04", "weights_P05")
RUN_RE = re.compile(r"s_(?P<stamp>\d\d-\d\d__\d\d-\d\d)_two_action_e2e_rma_z(?P<z>\d+)_w(?P<w>\d+)$")


def _parse_int_set(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    values = {int(part.strip()) for part in raw.split(",") if part.strip()}
    if not values:
        raise ValueError("Parsed an empty integer set.")
    return values


def _parse_weights(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_WEIGHTS
    weights = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not weights:
        raise ValueError("Parsed an empty weights list.")
    return weights


def _latest_matrix_stamp() -> str:
    stamps = []
    for path in EXPERIMENTS.glob("s_*_two_action_e2e_rma_z*_w*"):
        match = RUN_RE.match(path.name)
        if match:
            stamps.append(match.group("stamp"))
    if not stamps:
        raise RuntimeError("No e2e RMA matrix experiment folders found.")
    return sorted(set(stamps))[-1]


def _matrix_runs() -> list[tuple[int, int, Path]]:
    stamp = os.environ.get("TWO_ACTION_E2E_MATRIX_SWEEP_STAMP") or _latest_matrix_stamp()
    latents = _parse_int_set(os.environ.get("TWO_ACTION_E2E_MATRIX_SWEEP_LATENTS"))
    windows = _parse_int_set(os.environ.get("TWO_ACTION_E2E_MATRIX_SWEEP_WINDOWS"))

    runs = []
    for path in EXPERIMENTS.glob(f"s_{stamp}_two_action_e2e_rma_z*_w*"):
        match = RUN_RE.match(path.name)
        if not match:
            continue
        latent_dim = int(match.group("z"))
        window_length = int(match.group("w"))
        if latents is not None and latent_dim not in latents:
            continue
        if windows is not None and window_length not in windows:
            continue
        runs.append((latent_dim, window_length, path))

    if not runs:
        raise RuntimeError(f"No matrix runs matched stamp={stamp!r}.")
    return sorted(runs, key=lambda item: (item[0], item[1]))


def _require_checkpoint(exp_dir: Path, weights_name: str) -> None:
    path = exp_dir / f"{weights_name}.zip"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")


def _run_sweep(latent_dim: int, window_length: int, exp_dir: Path, weights: tuple[str, ...]) -> None:
    specs = []
    for weights_name in weights:
        _require_checkpoint(exp_dir, weights_name)
        phase = weights_name.removeprefix("weights_")
        specs.append(
            {
                "label": f"e2e_z{latent_dim}_w{window_length}_{phase}",
                "kind": "ppo",
                "experiment": str(exp_dir.resolve()),
                "weights_name": weights_name,
            }
        )
    specs.append(
        {
            "label": "oracle_lqr",
            "kind": "lqr",
            "experiment": str(exp_dir.resolve()),
        }
    )

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")
    env["THETA_SWEEP_TARGET_EXPERIMENT"] = str(exp_dir.resolve())
    env["THETA_SWEEP_OUTPUT_SUBDIR"] = os.environ.get(
        "TWO_ACTION_E2E_MATRIX_SWEEP_OUTPUT_SUBDIR",
        "theta_sweep_checkpoints_only",
    )
    env["THETA_SWEEP_DETERMINISTIC"] = os.environ.get("TWO_ACTION_E2E_MATRIX_SWEEP_DETERMINISTIC", "1")
    env["THETA_SWEEP_RETURN_MODE"] = os.environ.get("TWO_ACTION_E2E_MATRIX_SWEEP_RETURN_MODE", "quadratic")
    env["THETA_SWEEP_SAVE_TRAJECTORY_TRACE"] = os.environ.get("TWO_ACTION_E2E_MATRIX_SWEEP_TRACE", "0")
    env["THETA_SWEEP_SAVE_STEP_LEVEL_CSV"] = os.environ.get("TWO_ACTION_E2E_MATRIX_SWEEP_STEP_CSV", "0")
    env["THETA_SWEEP_CONTROLLER_SPECS_JSON"] = json.dumps(specs)

    print(f"Running e2e matrix sweep: z={latent_dim}, w={window_length}")
    print(f"  experiment: {exp_dir.name}")
    print(f"  output:     {exp_dir / env['THETA_SWEEP_OUTPUT_SUBDIR']}")
    for spec in specs:
        print(f"    - {spec['label']}", flush=True)

    subprocess.run(["python3", str(SWEEP_SCRIPT)], cwd=ROOT, env=env, check=True)


def main() -> None:
    weights = _parse_weights(os.environ.get("TWO_ACTION_E2E_MATRIX_SWEEP_WEIGHTS"))
    runs = _matrix_runs()
    print(f"Found {len(runs)} matrix runs to sweep.")
    print(f"Weights: {', '.join(weights)}")
    print(
        "Trace:",
        os.environ.get("TWO_ACTION_E2E_MATRIX_SWEEP_TRACE", "0"),
        "Step CSV:",
        os.environ.get("TWO_ACTION_E2E_MATRIX_SWEEP_STEP_CSV", "0"),
    )
    for latent_dim, window_length, exp_dir in runs:
        _run_sweep(latent_dim, window_length, exp_dir, weights)


if __name__ == "__main__":
    main()
