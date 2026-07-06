from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
SWEEP_SCRIPT = ROOT / "lqr_theta_sweep_eval.py"

ENABLE_TRACE = "0"
DETERMINISTIC_POLICY = "1"


def exp(name: str) -> str:
    return str((EXPERIMENTS / name).resolve())


ONE_ACTION_PRIV = "s_04-10__11-41_privileged_masked_uncertainty"
ONE_ACTION_NO_UNC = "s_05-11__10-58_random_curr_no_uncertainty_baseline"
ONE_ACTION_UNC = "s_05-10__16-05_random_curr_dense_strong_id"

TWO_ACTION_PRIV = "s_05-16__14-44_two_action_large_privileged"
TWO_ACTION_NO_UNC = "s_05-16__14-44_two_action_large_no_uncertainty_baseline"
TWO_ACTION_UNC_BEST = "s_05-16__14-44_two_action_unc_cont_from_P05_pen0p15_ent0p002"
TWO_ACTION_POLISH = "s_05-18__22-20_two_action_polish_pen0p15_ent0p002_lr2e4"


INDIVIDUAL_JOBS = []

COMBINED_JOBS = [
    (
        "one_action_combined_final",
        ONE_ACTION_NO_UNC,
        [
            {"label": "privileged_ppo", "kind": "ppo", "experiment": exp(ONE_ACTION_PRIV), "weights_name": "weights"},
            {"label": "no_unc_rma_P03", "kind": "ppo", "experiment": exp(ONE_ACTION_NO_UNC), "weights_name": "weights_P03"},
            {"label": "unc_rma_P08", "kind": "ppo", "experiment": exp(ONE_ACTION_UNC), "weights_name": "weights_P08"},
            {"label": "unc_rma_P10", "kind": "ppo", "experiment": exp(ONE_ACTION_UNC), "weights_name": "weights_P10"},
            {"label": "lqr", "kind": "lqr", "experiment": exp(ONE_ACTION_NO_UNC)},
        ],
    ),
    (
        "two_action_combined_final",
        TWO_ACTION_NO_UNC,
        [
            {"label": "privileged_ppo", "kind": "ppo", "experiment": exp(TWO_ACTION_PRIV), "weights_name": "weights"},
            {"label": "no_unc_rma", "kind": "ppo", "experiment": exp(TWO_ACTION_NO_UNC), "weights_name": "weights"},
            {"label": "unc_rma_best_found", "kind": "ppo", "experiment": exp(TWO_ACTION_UNC_BEST), "weights_name": "weights"},
            {"label": "unc_rma_polish_best", "kind": "ppo", "experiment": exp(TWO_ACTION_POLISH), "weights_name": "weights_best"},
            {"label": "lqr", "kind": "lqr", "experiment": exp(TWO_ACTION_NO_UNC)},
        ],
    ),
]


def _check_checkpoint(experiment: str, weights_name: str) -> None:
    path = EXPERIMENTS / experiment / f"{weights_name}.zip"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")


def _run_sweep(target_experiment: str, output_subdir: str, specs: list[dict] | None = None, weights_name: str | None = None) -> None:
    env = os.environ.copy()
    env["THETA_SWEEP_TARGET_EXPERIMENT"] = exp(target_experiment)
    env["THETA_SWEEP_OUTPUT_SUBDIR"] = output_subdir
    env["THETA_SWEEP_DETERMINISTIC"] = DETERMINISTIC_POLICY
    env["THETA_SWEEP_SAVE_TRAJECTORY_TRACE"] = ENABLE_TRACE

    if specs is None:
        env["THETA_SWEEP_SINGLE_PPO_EXPERIMENT"] = exp(target_experiment)
        env["THETA_SWEEP_SINGLE_LABEL"] = f"{target_experiment}_{weights_name}"
        env["THETA_SWEEP_WEIGHTS_NAME"] = str(weights_name)
    else:
        env["THETA_SWEEP_CONTROLLER_SPECS_JSON"] = json.dumps(specs)

    subprocess.run(["python3", str(SWEEP_SCRIPT)], cwd=ROOT, env=env, check=True)


def main() -> None:
    stamp = datetime.now().strftime("%m-%d__%H-%M")
    schedule_dir = EXPERIMENTS / f"s_{stamp}_final_meeting_sweeps"
    schedule_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": stamp,
        "enable_trace": ENABLE_TRACE,
        "deterministic_policy": DETERMINISTIC_POLICY,
        "individual_jobs": [],
        "combined_jobs": [],
    }

    for label, experiment, weights_name in INDIVIDUAL_JOBS:
        _check_checkpoint(experiment, weights_name)
        output_subdir = f"final_meeting_no_trace_{label}"
        payload["individual_jobs"].append({
            "label": label,
            "experiment": experiment,
            "weights_name": weights_name,
            "output_folder": f"experiments/{experiment}/{output_subdir}",
        })
        with open(schedule_dir / "final_meeting_sweeps.yaml", "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        print(f"START individual {label}", flush=True)
        _run_sweep(experiment, output_subdir, weights_name=weights_name)
        print(f"END   individual {label}", flush=True)

    for label, target_experiment, specs in COMBINED_JOBS:
        for spec in specs:
            if spec["kind"] == "ppo":
                _check_checkpoint(Path(spec["experiment"]).name, spec["weights_name"])
        output_subdir = f"final_meeting_no_trace_{label}"
        payload["combined_jobs"].append({
            "label": label,
            "target_experiment": target_experiment,
            "controllers": specs,
            "output_folder": f"experiments/{target_experiment}/{output_subdir}",
        })
        with open(schedule_dir / "final_meeting_sweeps.yaml", "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        print(f"START combined {label}", flush=True)
        _run_sweep(target_experiment, output_subdir, specs=specs)
        print(f"END   combined {label}", flush=True)

    with open(schedule_dir / "final_meeting_sweeps.yaml", "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"Wrote schedule to {schedule_dir / 'final_meeting_sweeps.yaml'}")


if __name__ == "__main__":
    main()
