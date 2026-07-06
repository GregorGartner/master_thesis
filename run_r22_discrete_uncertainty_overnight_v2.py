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

    # R22 discrete-theta stress variant:
    # more process noise makes correct adaptation matter more;
    # longer episodes create more finite-window refresh pressure;
    # a longer window lets uncertainty-aware policies exploit a successful probe for longer;
    # faster ID updates let the policy react sooner after informative actions.
    env["TWO_ACTION_DISCRETE_FINAL_FULL_BUDGET"] = "1"
    env["TWO_ACTION_R22_PROCESS_NOISE_STD"] = "0.075"
    env["TWO_ACTION_R22_MAX_EPISODE_STEPS"] = "768"
    env["TWO_ACTION_R22_WINDOW_LENGTH"] = "75"
    env["TWO_ACTION_R22_ID_UPDATE_INTERVAL"] = "5"
    return env


def _run(label: str, script: str, env: dict[str, str]) -> None:
    print(f"\n=== START {label}: {script} ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, env=env, check=True)
    print(f"=== END   {label}: {script} ===\n", flush=True)


def main() -> None:
    env = _base_env()
    print(
        "R22 discrete uncertainty v2 overrides: "
        f"noise={env['TWO_ACTION_R22_PROCESS_NOISE_STD']}, "
        f"horizon={env['TWO_ACTION_R22_MAX_EPISODE_STEPS']}, "
        f"window={env['TWO_ACTION_R22_WINDOW_LENGTH']}, "
        f"id_update={env['TWO_ACTION_R22_ID_UPDATE_INTERVAL']}",
        flush=True,
    )
    _run(
        "discrete neural final methods full budget, longer/noisier variant",
        "run_two_action_selected_system_final_methods_discrete_theta.py",
        env,
    )
    _run(
        "discrete BLR mean vs mean+std full budget, longer/noisier variant",
        "run_two_action_selected_system_blr_ppo_discrete_theta.py",
        env,
    )
    _run(
        "discrete neural uncertainty diagnostic, longer/noisier variant",
        "run_two_action_discrete_uncertainty_diagnostic.py",
        env,
    )


if __name__ == "__main__":
    main()
