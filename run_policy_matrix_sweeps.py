from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
SWEEP_SCRIPT = ROOT / "lqr_theta_sweep_eval.py"
OUTPUT_SUBDIR = "theta_sweep_policy_matrix_trace" # "theta_sweep_policy_matrix_deterministic"
MATRIX_PREFIX: str | None = "s_05-11__10-58"  # latest two-action next-matrix run
WEIGHT_NAMES = ["weights", "weights_best"] + [f"weights_P{i:02d}" for i in range(1, 10)]
ENABLE_TRACE = "0" # "0" or "1" to save trajectory traces (needed for cumulative_inforamtion_vs_cost and trajectory_diagnostics)
DETERMINISTIC_POLICY = "1" # "0" or "1" to use deterministic policy during sweeps (no action noise, no stochasticity in the policy)


def _detect_latest_prefix() -> str:
    prefixes = sorted({p.name.split("_random_curr_", 1)[0] for p in EXPERIMENTS_DIR.glob("s_*_random_curr_*")})
    if not prefixes:
        raise RuntimeError("No matrix runs found (pattern: s_*_random_curr_*).")
    return prefixes[-1]


def _collect_experiments(prefix: str) -> list[Path]:
    paths = sorted(EXPERIMENTS_DIR.glob(f"{prefix}_*"))
    if not paths:
        raise RuntimeError(f"No experiments found for prefix: {prefix}")
    return [p for p in paths if (p / "weights_best.zip").exists() or (p / "weights.zip").exists()]


def main() -> None:
    prefix = MATRIX_PREFIX or _detect_latest_prefix()
    exps = _collect_experiments(prefix)
    jobs = [(exp, weights_name) for exp in exps for weights_name in WEIGHT_NAMES if (exp / f"{weights_name}.zip").exists()]
    total = len(jobs)
    print(f"Running theta sweeps for prefix={prefix} ({total} phase checkpoints)")

    for i, (exp, weights_name) in enumerate(jobs, start=1):
        phase = weights_name.removeprefix("weights_")
        print(f"[{i}/{total}] START sweep for {exp.name} {phase}", flush=True)
        env = os.environ.copy()
        env["THETA_SWEEP_TARGET_EXPERIMENT"] = str(exp)
        env["THETA_SWEEP_SINGLE_PPO_EXPERIMENT"] = str(exp)
        env["THETA_SWEEP_SINGLE_LABEL"] = f"{exp.name}_{phase}"
        env["THETA_SWEEP_OUTPUT_SUBDIR"] = f"{OUTPUT_SUBDIR}_{phase}"
        env["THETA_SWEEP_DETERMINISTIC"] = DETERMINISTIC_POLICY
        env["THETA_SWEEP_SAVE_TRAJECTORY_TRACE"] = ENABLE_TRACE
        env["THETA_SWEEP_WEIGHTS_NAME"] = weights_name
        subprocess.run(["python3", str(SWEEP_SCRIPT)], cwd=ROOT, env=env, check=True)
        print(f"[{i}/{total}] END   sweep for {exp.name} {phase}", flush=True)


if __name__ == "__main__":
    main()
