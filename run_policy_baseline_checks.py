from __future__ import annotations

import os
from pathlib import Path

import lqr_theta_sweep_eval as sweep


ROOT = Path(__file__).resolve().parent

# Baselines for the neural/non-GLS setup:
# - privileged: policy gets ground-truth theta
# - nominal RMA: encoder-based policy trained without naive exploration
# - lqr: analytic Riccati controller with ground-truth dynamics
PRIVILEGED_EXPERIMENT = Path(
    os.environ.get(
        "BASELINE_PRIVILEGED_EXPERIMENT",
        ROOT / "experiments" / "s_04-10__11-41_privileged_masked_uncertainty",
    )
)
NOMINAL_RMA_EXPERIMENT = Path(
    os.environ.get(
        "BASELINE_NOMINAL_RMA_EXPERIMENT",
        ROOT / "experiments" / "s_04-10__16-05_nll_20_no_exp",
    )
)

# The target experiment supplies the LQR environment config and theta sweep range.
TARGET_EXPERIMENT = Path(
    os.environ.get("BASELINE_TARGET_EXPERIMENT", str(PRIVILEGED_EXPERIMENT))
)
OUTPUT_SUBDIR = os.environ.get("BASELINE_OUTPUT_SUBDIR", "baseline_checks/three_baselines")

N_THETA_POINTS = int(os.environ.get("BASELINE_N_THETA_POINTS", "41"))
EPISODES_PER_THETA = int(os.environ.get("BASELINE_EPISODES_PER_THETA", "20"))
EVAL_BASE_SEED = int(os.environ.get("BASELINE_EVAL_BASE_SEED", "12345"))
DETERMINISTIC = os.environ.get("BASELINE_DETERMINISTIC", "1").lower() not in {
    "0",
    "false",
    "no",
}


def _validate_experiment(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} experiment does not exist: {path}")
    if not (path / "config.yaml").exists():
        raise FileNotFoundError(f"{label} experiment has no config.yaml: {path}")
    if not ((path / "weights_best.zip").exists() or (path / "weights.zip").exists()):
        raise FileNotFoundError(
            f"{label} experiment has neither weights_best.zip nor weights.zip: {path}"
        )


def main() -> None:
    _validate_experiment(PRIVILEGED_EXPERIMENT, "privileged")
    _validate_experiment(NOMINAL_RMA_EXPERIMENT, "nominal RMA")
    if not (TARGET_EXPERIMENT / "config.yaml").exists():
        raise FileNotFoundError(f"target experiment has no config.yaml: {TARGET_EXPERIMENT}")

    sweep.TARGET_EXPERIMENT = str(TARGET_EXPERIMENT)
    sweep.OUTPUT_SUBDIR = OUTPUT_SUBDIR
    sweep.SWEEP_DETERMINISTIC = DETERMINISTIC
    sweep.THETA_GRID = None
    sweep.N_THETA_POINTS = N_THETA_POINTS
    sweep.PROCESS_NOISE_SCALES = [1.0]
    sweep.EPISODES_PER_THETA = EPISODES_PER_THETA
    sweep.EVAL_BASE_SEED = EVAL_BASE_SEED
    sweep.SAVE_STEP_LEVEL_CSV = True

    sweep.CONTROLLER_SPECS = [
        {
            "label": "privileged",
            "kind": "ppo",
            "experiment": str(PRIVILEGED_EXPERIMENT),
        },
        {
            "label": "nominal_rma_no_exploration",
            "kind": "ppo",
            "experiment": str(NOMINAL_RMA_EXPERIMENT),
        },
        {
            "label": "lqr",
            "kind": "lqr",
            "experiment": str(TARGET_EXPERIMENT),
        },
    ]

    print("Running baseline theta sweep")
    print(f"  target config: {TARGET_EXPERIMENT}")
    print(f"  privileged:    {PRIVILEGED_EXPERIMENT}")
    print(f"  nominal RMA:   {NOMINAL_RMA_EXPERIMENT}")
    print(f"  output:        {TARGET_EXPERIMENT / OUTPUT_SUBDIR}")
    print(f"  theta points:  {N_THETA_POINTS}")
    print(f"  episodes/theta:{EPISODES_PER_THETA}")
    print(f"  deterministic: {DETERMINISTIC}")

    sweep.main()


if __name__ == "__main__":
    main()
