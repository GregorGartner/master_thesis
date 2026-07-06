from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
SWEEP_SCRIPT = ROOT / "lqr_theta_sweep_eval.py"

RECURRENT_EXPERIMENT = os.environ.get("TWO_ACTION_RECURRENT_EXPERIMENT") or None
RECURRENT_WEIGHTS = os.environ.get("TWO_ACTION_RECURRENT_WEIGHTS", "weights")

PRIVILEGED_EXPERIMENT = "s_05-16__14-44_two_action_large_privileged"
NO_UNC_EXPERIMENT = "s_05-16__14-44_two_action_large_no_uncertainty_baseline"
UNC_EXPERIMENT = "s_05-16__14-44_two_action_unc_cont_from_P05_pen0p15_ent0p002"

OUTPUT_SUBDIR = "theta_sweep_recurrent_comparison"
DETERMINISTIC_POLICY = "1"
ENABLE_TRACE = os.environ.get("TWO_ACTION_RECURRENT_SWEEP_TRACE", "1")
RETURN_MODE = os.environ.get("TWO_ACTION_RECURRENT_SWEEP_RETURN_MODE", "quadratic")


def exp(name: str) -> Path:
    return EXPERIMENTS / name


def latest_recurrent_experiment() -> str:
    candidates = sorted(
        path
        for path in EXPERIMENTS.glob("s_*_two_action_recurrent*_policy")
        if "smoke" not in path.name and "pipeline" not in path.name
    )
    if not candidates:
        raise RuntimeError("No two-action recurrent policy experiment found.")
    return candidates[-1].name


def require_checkpoint(experiment: str, weights_name: str) -> None:
    path = exp(experiment) / f"{weights_name}.zip"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")


def main() -> None:
    recurrent = RECURRENT_EXPERIMENT or latest_recurrent_experiment()
    require_checkpoint(recurrent, RECURRENT_WEIGHTS)
    require_checkpoint(PRIVILEGED_EXPERIMENT, "weights")
    require_checkpoint(NO_UNC_EXPERIMENT, "weights")
    require_checkpoint(UNC_EXPERIMENT, "weights")

    specs = [
        {
            "label": "privileged_ppo",
            "kind": "ppo",
            "experiment": str(exp(PRIVILEGED_EXPERIMENT).resolve()),
            "weights_name": "weights",
        },
        {
            "label": "no_unc_rma",
            "kind": "ppo",
            "experiment": str(exp(NO_UNC_EXPERIMENT).resolve()),
            "weights_name": "weights",
        },
        {
            "label": "unc_rma_best",
            "kind": "ppo",
            "experiment": str(exp(UNC_EXPERIMENT).resolve()),
            "weights_name": "weights",
        },
        {
            "label": "recurrent_ppo",
            "kind": "ppo",
            "experiment": str(exp(recurrent).resolve()),
            "weights_name": RECURRENT_WEIGHTS,
        },
        {
            "label": "oracle_lqr",
            "kind": "lqr",
            "experiment": str(exp(recurrent).resolve()),
        },
    ]

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")
    env["THETA_SWEEP_TARGET_EXPERIMENT"] = str(exp(recurrent).resolve())
    env["THETA_SWEEP_OUTPUT_SUBDIR"] = OUTPUT_SUBDIR
    env["THETA_SWEEP_DETERMINISTIC"] = DETERMINISTIC_POLICY
    env["THETA_SWEEP_SAVE_TRAJECTORY_TRACE"] = ENABLE_TRACE
    env["THETA_SWEEP_RETURN_MODE"] = RETURN_MODE
    env["THETA_SWEEP_CONTROLLER_SPECS_JSON"] = json.dumps(specs)

    print("Running two-action recurrent comparison sweep")
    print(f"  recurrent: {recurrent}/{RECURRENT_WEIGHTS}")
    print(f"  output:    {exp(recurrent) / OUTPUT_SUBDIR}")
    for spec in specs:
        print(f"    - {spec['label']}")

    subprocess.run(["python3", str(SWEEP_SCRIPT)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
