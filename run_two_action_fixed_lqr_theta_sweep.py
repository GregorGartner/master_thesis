from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
SWEEP_SCRIPT = ROOT / "lqr_theta_sweep_eval.py"

DEFAULT_TARGET = "s_05-28__10-52_two_action_recurrent64_policy"
DEFAULT_ASSUMED_THETAS = [-0.25, -0.125, 0.0, 0.125, 0.25]


def _parse_assumed_thetas() -> list[float]:
    raw = os.environ.get("FIXED_LQR_ASSUMED_THETAS")
    if not raw:
        return DEFAULT_ASSUMED_THETAS
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _theta_label(theta: float) -> str:
    sign = "p" if theta >= 0.0 else "m"
    return f"{sign}{abs(theta):.3f}".replace(".", "p")


def main() -> None:
    target_name = os.environ.get("FIXED_LQR_TARGET_EXPERIMENT", DEFAULT_TARGET)
    target_exp = EXPERIMENTS / target_name
    if not target_exp.exists():
        raise FileNotFoundError(f"Missing target experiment: {target_exp}")

    assumed_thetas = _parse_assumed_thetas()
    specs = [
        {
            "label": f"fixed_lqr_theta_{_theta_label(theta)}",
            "kind": "fixed_lqr",
            "experiment": str(target_exp.resolve()),
            "assumed_theta": theta,
        }
        for theta in assumed_thetas
    ]
    specs.append(
        {
            "label": "oracle_lqr",
            "kind": "lqr",
            "experiment": str(target_exp.resolve()),
        }
    )

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")
    env["THETA_SWEEP_TARGET_EXPERIMENT"] = str(target_exp.resolve())
    env["THETA_SWEEP_OUTPUT_SUBDIR"] = os.environ.get(
        "FIXED_LQR_OUTPUT_SUBDIR",
        "../theta_sweep_fixed_assumed_lqr",
    )
    env["THETA_SWEEP_DETERMINISTIC"] = "1"
    env["THETA_SWEEP_SAVE_TRAJECTORY_TRACE"] = "0"
    env["THETA_SWEEP_RETURN_MODE"] = "quadratic"
    env["THETA_SWEEP_CONTROLLER_SPECS_JSON"] = json.dumps(specs)

    print("Running fixed-assumed-theta LQR sweep", flush=True)
    print(f"  target: {target_exp}", flush=True)
    print(f"  output: {(target_exp / env['THETA_SWEEP_OUTPUT_SUBDIR']).resolve()}", flush=True)
    for spec in specs:
        assumed = spec.get("assumed_theta")
        suffix = "" if assumed is None else f" assumed_theta={assumed:g}"
        print(f"    - {spec['label']}{suffix}", flush=True)

    subprocess.run(["python3", str(SWEEP_SCRIPT)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
