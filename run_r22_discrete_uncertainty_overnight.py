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
    return env


def _run(label: str, script: str, env: dict[str, str]) -> None:
    print(f"\n=== START {label}: {script} ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, env=env, check=True)
    print(f"=== END   {label}: {script} ===\n", flush=True)


def main() -> None:
    env = _base_env()
    env["TWO_ACTION_DISCRETE_FINAL_FULL_BUDGET"] = "1"
    _run("discrete neural final methods full budget", "run_two_action_selected_system_final_methods_discrete_theta.py", env)
    _run("discrete BLR mean vs mean+std full budget", "run_two_action_selected_system_blr_ppo_discrete_theta.py", env)

    # The diagnostic is cheap compared with training and gives the uncertainty ablation plots for the neural run.
    _run("discrete neural uncertainty diagnostic", "run_two_action_discrete_uncertainty_diagnostic.py", env)


if __name__ == "__main__":
    main()
