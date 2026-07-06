from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import yaml

from run_two_action_pipeline import CONFIG_PATH, _resolve_exp_root
from run_two_action_system_neural_screening import (
    SELECTION_ROOT,
    _run_sweep,
    _score_sweep,
    _train_system,
    _write_csv,
)


def _latest_selected_system() -> tuple[Path, dict, dict]:
    override = os.environ.get("TWO_ACTION_NEURAL_SELECTION_DIR")
    if override:
        neural_dir = Path(override).resolve()
    else:
        pointer = SELECTION_ROOT / "latest_neural_screening.txt"
        if not pointer.exists():
            raise FileNotFoundError("No latest neural-screening pointer found.")
        neural_dir = Path(pointer.read_text().strip())
    with open(neural_dir / "selected_system.json", "r") as f:
        selected = json.load(f)
    if not selected["selection_criteria_met"] and not os.environ.get("TWO_ACTION_CONFIRM_UNQUALIFIED"):
        raise RuntimeError(
            "The first-seed neural screening did not produce a qualifying system. "
            "Inspect the screening results instead of freezing its fallback selection."
        )
    return neural_dir, selected["run"]["system"], selected["score"]


def main() -> None:
    neural_dir, system, first_seed_score = _latest_selected_system()
    original_text = CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_text)
    exp_root = _resolve_exp_root(base_cfg)
    stamp = datetime.now().strftime("%m-%d__%H-%M")
    run_root = SELECTION_ROOT / f"confirmation_{stamp}_{system['label']}_seed1"
    run_root.mkdir(parents=True, exist_ok=True)

    try:
        run = _train_system(
            base_cfg=base_cfg,
            exp_root=exp_root,
            system=system,
            stamp=stamp,
            seed=1,
            run_root=run_root,
        )
    finally:
        CONFIG_PATH.write_text(original_text)

    with open(run_root / "confirmation_run.json", "w") as f:
        json.dump(
            {
                "source_neural_screening": str(neural_dir),
                "seed": 1,
                "run": run,
            },
            f,
            indent=2,
        )
    sweep_dir = _run_sweep(
        run,
        theta_points=41,
        episodes_per_theta=20,
        output_subdir="theta_sweep_confirmation_seed1",
    )
    score = _score_sweep(run, sweep_dir)
    _write_csv(run_root / "confirmation_scorecard.csv", [score])
    reproduced = bool(first_seed_score["qualifies"]) and bool(score["qualifies"])
    with open(run_root / "freeze_decision.json", "w") as f:
        json.dump(
            {
                "system": system,
                "first_seed_qualifies": bool(first_seed_score["qualifies"]),
                "second_seed_qualifies": bool(score["qualifies"]),
                "freeze_system": reproduced,
            },
            f,
            indent=2,
        )
    (SELECTION_ROOT / "latest_confirmation.txt").write_text(str(run_root.resolve()) + "\n")
    print(f"Saved second-seed confirmation to: {run_root}", flush=True)
    print(f"Freeze system: {reproduced}", flush=True)


if __name__ == "__main__":
    main()
