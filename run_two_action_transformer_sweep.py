from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
SWEEP_SCRIPT = ROOT / "lqr_theta_sweep_eval.py"

TRANSFORMER_EXPERIMENT = os.environ.get(
    "TWO_ACTION_TRANSFORMER_EXPERIMENT",
    "s_06-01__22-19_two_action_transformer_policy",
)
OUTPUT_SUBDIR = os.environ.get(
    "TWO_ACTION_TRANSFORMER_SWEEP_OUTPUT_SUBDIR",
    "theta_sweep_best_checkpoints_only",
)
WEIGHTS = ["weights_best_P01", "weights_best_P02", "weights_best_P03", "weights_best_P04"]


def exp(name: str) -> Path:
    return EXPERIMENTS / name


def require_checkpoint(experiment: str, weights_name: str) -> None:
    path = exp(experiment) / f"{weights_name}.zip"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")


def main() -> None:
    exp_dir = exp(TRANSFORMER_EXPERIMENT)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Missing experiment: {exp_dir}")

    specs = []
    for weights_name in WEIGHTS:
        require_checkpoint(TRANSFORMER_EXPERIMENT, weights_name)
        specs.append(
            {
                "label": f"transformer_{weights_name.removeprefix('weights_best_')}",
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
    env["THETA_SWEEP_OUTPUT_SUBDIR"] = OUTPUT_SUBDIR
    env["THETA_SWEEP_DETERMINISTIC"] = "1"
    env["THETA_SWEEP_SAVE_TRAJECTORY_TRACE"] = "0"
    env["THETA_SWEEP_RETURN_MODE"] = "quadratic"
    env["THETA_SWEEP_CONTROLLER_SPECS_JSON"] = json.dumps(specs)

    print(f"Running Transformer checkpoint sweep: {TRANSFORMER_EXPERIMENT}")
    print(f"  output: {exp_dir / OUTPUT_SUBDIR}")
    for spec in specs:
        print(f"    - {spec['label']}")
    subprocess.run(["python3", str(SWEEP_SCRIPT)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
