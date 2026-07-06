from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
SWEEP_SCRIPT = ROOT / "lqr_theta_sweep_eval.py"

NLL_EXPERIMENT = "s_06-10__12-55_two_action_gradual_encoder_curriculum"
MLE_EXPERIMENT = "s_06-10__22-34_two_action_gradual_mle_encoder_curriculum"

# P05 predicted uncertainty after the 49-step warmup, in the scaled units fed
# to the policy: mean=0.4350, std=0.1605. These bounds match that mean/variance.
MATCHED_UNCERTAINTY_MEAN = 0.4350
MATCHED_UNCERTAINTY_LOW = 0.1570
MATCHED_UNCERTAINTY_HIGH = 0.7130


def _require_checkpoint(experiment: Path, weights_name: str) -> None:
    checkpoint = experiment / f"{weights_name}.zip"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")


def main() -> None:
    nll_exp = EXPERIMENTS / NLL_EXPERIMENT
    mle_exp = EXPERIMENTS / MLE_EXPERIMENT
    _require_checkpoint(nll_exp, "weights_P05")
    _require_checkpoint(mle_exp, "weights_P05")

    specs = [
        {
            "label": "nll_P05_predicted_uncertainty",
            "kind": "ppo",
            "experiment": str(nll_exp.resolve()),
            "weights_name": "weights_P05",
            "uncertainty_override": "predicted",
        },
        {
            "label": "nll_P05_zero_uncertainty",
            "kind": "ppo",
            "experiment": str(nll_exp.resolve()),
            "weights_name": "weights_P05",
            "uncertainty_override": "zeros",
        },
        {
            "label": "nll_P05_constant_uncertainty",
            "kind": "ppo",
            "experiment": str(nll_exp.resolve()),
            "weights_name": "weights_P05",
            "uncertainty_override": "constant",
            "uncertainty_value": MATCHED_UNCERTAINTY_MEAN,
        },
        {
            "label": "nll_P05_random_uncertainty",
            "kind": "ppo",
            "experiment": str(nll_exp.resolve()),
            "weights_name": "weights_P05",
            "uncertainty_override": "random_uniform",
            "uncertainty_low": MATCHED_UNCERTAINTY_LOW,
            "uncertainty_high": MATCHED_UNCERTAINTY_HIGH,
        },
        {
            "label": "mle_P05",
            "kind": "ppo",
            "experiment": str(mle_exp.resolve()),
            "weights_name": "weights_P05",
        },
        {
            "label": "nominal_lqr",
            "kind": "nominal_lqr",
            "experiment": str(nll_exp.resolve()),
        },
        {
            "label": "oracle_lqr",
            "kind": "lqr",
            "experiment": str(nll_exp.resolve()),
        },
    ]

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")
    env["THETA_SWEEP_TARGET_EXPERIMENT"] = str(nll_exp.resolve())
    env["THETA_SWEEP_OUTPUT_SUBDIR"] = os.environ.get(
        "TWO_ACTION_UNCERTAINTY_ABLATION_OUTPUT_SUBDIR",
        "theta_sweep_uncertainty_input_ablation",
    )
    env["THETA_SWEEP_DETERMINISTIC"] = "1"
    env["THETA_SWEEP_RETURN_MODE"] = "quadratic"
    env["THETA_SWEEP_SAVE_TRAJECTORY_TRACE"] = "0"
    env["THETA_SWEEP_SAVE_STEP_LEVEL_CSV"] = "0"
    env["THETA_SWEEP_CONTROLLER_SPECS_JSON"] = json.dumps(specs)

    print("Running uncertainty-input and exact-nominal-LQR ablation", flush=True)
    print(f"  matched uncertainty mean:  {MATCHED_UNCERTAINTY_MEAN:.4f}", flush=True)
    print(
        f"  matched uncertainty range: [{MATCHED_UNCERTAINTY_LOW:.4f}, "
        f"{MATCHED_UNCERTAINTY_HIGH:.4f}]",
        flush=True,
    )
    for spec in specs:
        print(f"    - {spec['label']}", flush=True)

    subprocess.run(["python3", str(SWEEP_SCRIPT)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
