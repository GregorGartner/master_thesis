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
    print(f"\n##### START {label}: {script} #####", flush=True)
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, env=env, check=True)
    print(f"##### END   {label}: {script} #####\n", flush=True)


def main() -> None:
    env = _base_env()
    _run("R22 discrete theta full neural+BLR suite", "run_r22_discrete_uncertainty_overnight.py", env)
    _run("scalar (a,b) diagnostics+BLR+neural suite", "run_scalar_ab_uncertainty_overnight.py", env)
    print("All overnight uncertainty experiments finished.", flush=True)


if __name__ == "__main__":
    main()
