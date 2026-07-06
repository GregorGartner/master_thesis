from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
SWEEP_SCRIPT = ROOT / "lqr_theta_sweep_eval.py"

VANILLA_RMA_EXPERIMENT = "s_05-16__14-44_two_action_large_no_uncertainty_baseline"
WEIGHTS_NAME = "weights"
THETA_QUANTIZATION_STEPS = (0.001, 0.01, 0.05)


def main() -> None:
    experiment = EXPERIMENTS / VANILLA_RMA_EXPERIMENT
    checkpoint = experiment / f"{WEIGHTS_NAME}.zip"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    specs = [
        {
            "label": "vanilla_rma_original",
            "kind": "ppo",
            "experiment": str(experiment.resolve()),
            "weights_name": WEIGHTS_NAME,
        }
    ]
    for step in THETA_QUANTIZATION_STEPS:
        label_step = str(step).replace(".", "p")
        specs.append(
            {
                "label": f"vanilla_rma_quantized_theta_{label_step}",
                "kind": "ppo",
                "experiment": str(experiment.resolve()),
                "weights_name": WEIGHTS_NAME,
                "mean_context_override": "quantize",
                "mean_context_quantization_step": step,
            }
        )
    specs.extend(
        [
            {
                "label": "vanilla_rma_zero_context",
                "kind": "ppo",
                "experiment": str(experiment.resolve()),
                "weights_name": WEIGHTS_NAME,
                "mean_context_override": "zeros",
            },
            {
                "label": "nominal_lqr",
                "kind": "nominal_lqr",
                "experiment": str(experiment.resolve()),
            },
            {
                "label": "oracle_lqr",
                "kind": "lqr",
                "experiment": str(experiment.resolve()),
            },
        ]
    )

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")
    env["THETA_SWEEP_TARGET_EXPERIMENT"] = str(experiment.resolve())
    env["THETA_SWEEP_OUTPUT_SUBDIR"] = os.environ.get(
        "TWO_ACTION_VANILLA_RMA_CONTEXT_ABLATION_OUTPUT_SUBDIR",
        "theta_sweep_context_quantization_ablation",
    )
    env["THETA_SWEEP_DETERMINISTIC"] = "1"
    env["THETA_SWEEP_RETURN_MODE"] = "quadratic"
    env["THETA_SWEEP_SAVE_TRAJECTORY_TRACE"] = "0"
    env["THETA_SWEEP_SAVE_STEP_LEVEL_CSV"] = "0"
    env["THETA_SWEEP_CONTROLLER_SPECS_JSON"] = json.dumps(specs)

    print("Running vanilla-RMA context quantization ablation", flush=True)
    print(f"  experiment: {experiment}", flush=True)
    print(f"  checkpoint: {WEIGHTS_NAME}", flush=True)
    print("  quantization steps are in raw theta units", flush=True)
    for spec in specs:
        print(f"    - {spec['label']}", flush=True)

    subprocess.run(["python3", str(SWEEP_SCRIPT)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
