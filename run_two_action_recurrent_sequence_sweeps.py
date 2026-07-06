from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
SWEEP_SCRIPT = ROOT / "lqr_theta_sweep_eval.py"

WEIGHTS = ["weights_best_P01", "weights_best_P02", "weights_best_P03", "weights_best_P04"]
RUNS = [
    {
        "experiment": "s_06-02__08-37_two_action_recurrent64_short_bptt",
        "output_subdir": "theta_sweep_best_checkpoints_only",
        "label_prefix": "short_bptt",
    },
    {
        "experiment": "s_06-02__09-59_two_action_recurrent64_long_bptt",
        "output_subdir": "theta_sweep_best_checkpoints_only",
        "label_prefix": "long_bptt",
    },
]


def exp(name: str) -> Path:
    return EXPERIMENTS / name


def require_checkpoint(experiment: str, weights_name: str) -> None:
    path = exp(experiment) / f"{weights_name}.zip"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")


def run_sweep(run: dict) -> None:
    experiment = str(run["experiment"])
    exp_dir = exp(experiment)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Missing experiment: {exp_dir}")

    specs = []
    for weights_name in WEIGHTS:
        require_checkpoint(experiment, weights_name)
        specs.append(
            {
                "label": f"{run['label_prefix']}_{weights_name.removeprefix('weights_best_')}",
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
    env["THETA_SWEEP_OUTPUT_SUBDIR"] = str(run["output_subdir"])
    env["THETA_SWEEP_DETERMINISTIC"] = "1"
    env["THETA_SWEEP_SAVE_TRAJECTORY_TRACE"] = "0"
    env["THETA_SWEEP_RETURN_MODE"] = "quadratic"
    env["THETA_SWEEP_CONTROLLER_SPECS_JSON"] = json.dumps(specs)

    print(f"Running recurrent sequence checkpoint sweep: {experiment}", flush=True)
    print(f"  output: {exp_dir / str(run['output_subdir'])}", flush=True)
    for spec in specs:
        print(f"    - {spec['label']}", flush=True)
    subprocess.run(["python3", str(SWEEP_SCRIPT)], cwd=ROOT, env=env, check=True)


def main() -> None:
    for run in RUNS:
        run_sweep(run)


if __name__ == "__main__":
    main()
