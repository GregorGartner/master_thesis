from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")

    # Conservative paper-deadline default: same stable-wide-b family, but longer
    # optimization and a slightly stronger disturbance than the first scalar run.
    env.setdefault("SCALAR_AB_RANGE_PRESET", "stable_wide_b")
    env.setdefault("SCALAR_AB_PROCESS_NOISE_STD", "0.05")
    env.setdefault("SCALAR_AB_EPISODE_STEPS", "192")
    env.setdefault("SCALAR_AB_WINDOW_LENGTH", "48")
    env.setdefault("SCALAR_AB_WARMUP_STEPS", "47")
    env.setdefault("SCALAR_AB_UPDATE_INTERVAL", "8")
    env.setdefault("SCALAR_AB_LONG", "1")
    env.setdefault("SCALAR_AB_STEP_MULT", "2.0")
    return env


def _run(label: str, script: str, env: dict[str, str]) -> None:
    print(f"\n=== START {label}: {script} ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, env=env, check=True)
    print(f"=== END   {label}: {script} ===\n", flush=True)


def _read_pointer(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing pointer after run: {path}")
    return Path(path.read_text().strip()).resolve()


def _run_args(label: str, args: list[str], env: dict[str, str]) -> None:
    print(f"\n=== START {label}: {' '.join(args)} ===", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True)
    print(f"=== END   {label}: {' '.join(args)} ===\n", flush=True)


def main() -> None:
    env = _base_env()
    _run("scalar (a,b) analytic diagnostics", "run_scalar_ab_lqr_diagnostics.py", env)
    _run("scalar (a,b) BLR PPO long", "run_scalar_ab_blr_ppo.py", env)
    _run("scalar (a,b) neural RMA long", "run_scalar_ab_neural_rma.py", env)
    _run("scalar (a,b) uncertainty diagnostics", "run_scalar_ab_uncertainty_diagnostics.py", env)

    selection_root = ROOT / "experiments" / "scalar_ab_lqr"
    diagnostics = _read_pointer(selection_root / "latest_scalar_ab_diagnostics.txt")
    blr = _read_pointer(selection_root / "latest_scalar_ab_blr_ppo.txt")
    neural = _read_pointer(selection_root / "latest_scalar_ab_neural_rma.txt")
    _run_args(
        "scalar BLR/PPO comparison plots",
        ["plot_scalar_ab_blr_ppo_comparisons.py", str(blr), "--diagnostics-dir", str(diagnostics), "--grid", "both"],
        env,
    )
    _run_args(
        "scalar neural RMA comparison plots",
        ["plot_scalar_ab_blr_ppo_comparisons.py", str(neural), "--diagnostics-dir", str(diagnostics), "--grid", "both"],
        env,
    )
    _run_args(
        "scalar policy posterior geometry plots",
        [
            "plot_scalar_ab_policy_posterior_geometry.py",
            "--diagnostics-dir",
            str(diagnostics),
            "--blr-root",
            str(blr),
            "--neural-root",
            str(neural),
        ],
        env,
    )
    _run_args(
        "scalar CE-BLR non-ambiguous comparison plots",
        ["plot_scalar_ab_ce_blr_nonambiguous_comparison.py", "--diagnostics-dir", str(diagnostics)],
        env,
    )


if __name__ == "__main__":
    main()
