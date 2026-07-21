from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import run_r22_discrete_calibrated_uncertainty as r22


ROOT = Path(__file__).resolve().parent


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes"}


def _overnight_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if _truthy_env("OVERNIGHT_SMOKE"):
        env.setdefault("FRICTION_STOPPING_V15_SMOKE", "1")
        env.setdefault("FRICTION_STOPPING_SWEEP_SMOKE", "1")
        env.setdefault("R22_CALIBRATED_UNCERTAINTY_SMOKE", "1")
        env.setdefault("R22_CALIBRATED_UNCERTAINTY_METHODS", "gradual_nll_calibrated_ft")
        env.setdefault("R22_EVAL_EPISODES_PER_THETA", "1")
    return env


def _run_step(label: str, script: str, env: dict[str, str]) -> bool:
    cmd = ["python3", str(ROOT / script)]
    print(f"\n=== START {label}: {' '.join(cmd)} ===", flush=True)
    try:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            f"\n=== FAILED {label}: {' '.join(cmd)} "
            f"(exit code {exc.returncode}) ===",
            flush=True,
        )
        return False
    print(f"=== END   {label} ===\n", flush=True)
    return True


def _latest_existing_r22_run_root() -> Path | None:
    pointer = r22.POINTER
    if pointer.exists():
        raw = pointer.read_text().strip()
        if raw:
            path = Path(raw).expanduser().resolve()
            if path.exists():
                return path

    candidates = sorted(
        r22.RUN_FAMILY_ROOT.glob("calibrated_uncertainty_*_r22_1p5"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].resolve() if candidates else None


def _existing_r22_needs_training_resume() -> bool:
    run_root = _latest_existing_r22_run_root()
    if run_root is None:
        print("No existing R22 calibrated run found; skipping existing-run resume check.", flush=True)
        return False

    manifest_path = run_root / "calibrated_uncertainty_run.json"
    if not manifest_path.exists():
        print(f"Missing R22 manifest at {manifest_path}; running resume defensively.", flush=True)
        return True

    with manifest_path.open() as f:
        payload = json.load(f)

    dirs = payload.get("dirs", {})
    methods = list(payload.get("methods", []))
    seeds = [int(seed) for seed in payload.get("seeds", [])]
    missing: list[Path] = []
    for method in methods:
        by_seed = dirs.get(method, {})
        final_weights = r22._final_weights_name(method)
        for seed in seeds:
            raw_dir = by_seed.get(str(seed), by_seed.get(seed))
            if raw_dir is None:
                missing.append(run_root / f"missing_dir_{method}_seed{seed}")
                continue
            path = Path(raw_dir).resolve() / f"{final_weights}.zip"
            if not path.exists():
                missing.append(path)

    if missing:
        print("Existing R22 calibrated run is missing final checkpoints:", flush=True)
        for path in missing:
            print(f"  {path}", flush=True)
        return True

    print(f"Existing R22 calibrated run appears complete: {run_root}", flush=True)
    return False


def main() -> None:
    env = _overnight_env()
    smoke = _truthy_env("OVERNIGHT_SMOKE")
    failures: list[str] = []

    if smoke:
        print("OVERNIGHT_SMOKE=1: skipping existing long-run R22 resume/eval mutation.", flush=True)
    else:
        if _existing_r22_needs_training_resume():
            if not _run_step(
                "existing R22 calibrated training resume",
                "run_r22_discrete_calibrated_uncertainty_resume.py",
                env,
            ):
                failures.append("existing R22 calibrated training resume")
        if not _run_step(
            "existing R22 all-checkpoint eval resume",
            "run_r22_calibrated_all_checkpoints_eval_resume.py",
            env,
        ):
            failures.append("existing R22 all-checkpoint eval resume")

    if not _run_step("friction stopping v15", "run_friction_stopping_rma_v15.py", env):
        failures.append("friction stopping v15")
    if not _run_step("R22 calibrated fast-ID", "run_r22_discrete_calibrated_uncertainty_fastid.py", env):
        failures.append("R22 calibrated fast-ID")

    if failures:
        print("Overnight sequence finished with failed steps:", flush=True)
        for label in failures:
            print(f"  - {label}", flush=True)
        raise SystemExit(1)

    print("Overnight friction/R22 fast-ID sequence complete.", flush=True)


if __name__ == "__main__":
    main()
