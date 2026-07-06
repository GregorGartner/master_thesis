from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from run_two_action_gradual_mle_encoder_curriculum import _set_gradual_mle_curriculum
from run_two_action_pipeline import (
    CONFIG_PATH,
    ROOT,
    _base_stage_cfg,
    _resolve_exp_root,
    _run_with_config,
    _set_common_model_params,
    _snapshot_phase_weights,
)


SELECTION_ROOT = ROOT / "experiments" / "two_action_system_selection"
SWEEP_SCRIPT = ROOT / "lqr_theta_sweep_eval.py"
ARCH = [128, 128]

PRIVILEGED_STEPS = 8_000_000
ENCODER_STEPS = 2_000_000
VANILLA_STEPS = 5_000_000
GRADUAL_PHASES = [
    ("P01", 0.25, 0.10, 3e-4, 1_000_000),
    ("P02", 0.50, 0.08, 3e-4, 1_000_000),
    ("P03", 0.75, 0.05, 2e-4, 1_000_000),
    ("P04", 1.00, 0.02, 1e-4, 1_000_000),
    ("P05", 1.00, 0.005, 1e-4, 1_000_000),
]


def _smoke_enabled() -> bool:
    return os.environ.get("TWO_ACTION_NEURAL_SCREEN_SMOKE", "0").lower() in {"1", "true", "yes"}


def _steps(steps: int) -> int:
    return 16_384 if _smoke_enabled() else steps


def _latest_screening_dir() -> Path:
    override = os.environ.get("TWO_ACTION_SYSTEM_SCREENING_DIR")
    if override:
        return Path(override).resolve()
    latest = SELECTION_ROOT / "latest_screening.txt"
    if not latest.exists():
        raise FileNotFoundError("No latest screening pointer found. Run run_two_action_system_screening.py first.")
    return Path(latest.read_text().strip())


def _load_selected_systems() -> tuple[Path, list[dict]]:
    screening_dir = _latest_screening_dir()
    with open(screening_dir / "selected_systems.json", "r") as f:
        payload = json.load(f)
    systems = list(payload["selected_for_neural_screening"])
    if len(systems) != 2:
        raise ValueError(f"Expected exactly two neural-screening systems, got {len(systems)}.")
    return screening_dir, systems


def _set_system(cfg: dict, system: dict) -> None:
    lqr = cfg.setdefault("lqr_env", {})
    for key in ["A", "B", "delta_B", "Q", "R"]:
        lqr[key] = system[key]
    lqr["process_noise_std"] = 0.05
    lqr["initial_state_low"] = -0.3
    lqr["initial_state_high"] = 0.3
    lqr["max_episode_steps"] = 512
    for wrapper in cfg.get("wrappers", []):
        if wrapper.get("name") in {"DomainRandomizationWrapper", "ChangingCartPoleDynamics"}:
            wrapper.setdefault("params", {})["theta_mult_range"] = [-0.25, 0.25]


def _set_common(cfg: dict, seed: int) -> None:
    _set_common_model_params(cfg)
    params = cfg["model"]["params"]
    params["seed"] = int(seed)
    params["n_steps"] = 8_192
    params["batch_size"] = 1_024
    params["n_epochs"] = 8
    params["gamma"] = 0.995
    params["encoder_net_arch"] = list(ARCH)
    params["actor_net_arch"] = list(ARCH)
    params["critic_net_arch"] = list(ARCH)
    for callback in cfg.get("callbacks", []):
        if callback.get("name") == "LivePlotCallback":
            callback["enabled"] = True
        if callback.get("name") == "SaveModelCallback":
            callback.setdefault("params", {})
            callback["params"]["save_on_training_end"] = True


def _set_privileged(cfg: dict, seed: int) -> None:
    _set_common(cfg, seed)
    params = cfg["model"]["params"]
    params["context_mode"] = "privileged"
    params["privileged_uncertainty_mode"] = "zeros"
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["condition_on_uncertainty"] = False
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["learning_rate"] = 3e-4
    params["ent_coef"] = 0.01


def _set_mle_encoder(cfg: dict, seed: int) -> None:
    _set_common(cfg, seed)
    params = cfg["model"]["params"]
    params["context_mode"] = "encoder_mle"
    params["freeze_ppo"] = True
    params["regression_coef"] = 1.0
    params["condition_on_uncertainty"] = False
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["naive_action_noise_std"] = [0.0, 0.5]
    params["naive_action_noise_dist"] = ["gaussian", "uniform"]
    params["learning_rate"] = 1e-5
    params["ent_coef"] = 0.0


def _set_vanilla_policy(cfg: dict, seed: int) -> None:
    _set_common(cfg, seed)
    params = cfg["model"]["params"]
    params["context_mode"] = "encoder_mle"
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["condition_on_uncertainty"] = False
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["learning_rate"] = 2e-4
    params["ent_coef"] = 0.01


def _base_cfg(
    base_cfg: dict,
    exp_root: Path,
    exp_name: str,
    steps: int,
    system: dict,
) -> dict:
    cfg = _base_stage_cfg(base_cfg, exp_root, exp_name, _steps(steps))
    _set_system(cfg, system)
    return cfg


def _write_manifest(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "system_selection_neural_screening.yaml", "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _train_system(
    *,
    base_cfg: dict,
    exp_root: Path,
    system: dict,
    stamp: str,
    seed: int,
    run_root: Path,
) -> dict:
    label = system["label"]
    names = {
        "privileged": f"s_{stamp}_{label}_screen_privileged",
        "encoder": f"s_{stamp}_{label}_screen_mle_encoder",
        "vanilla": f"s_{stamp}_{label}_screen_vanilla_rma",
        "gradual": f"s_{stamp}_{label}_screen_gradual_mle",
    }
    dirs = {key: exp_root / value for key, value in names.items()}
    payload = {
        "system": system,
        "seed": seed,
        "smoke": _smoke_enabled(),
        "experiments": names,
        "budgets": {
            "privileged": _steps(PRIVILEGED_STEPS),
            "encoder": _steps(ENCODER_STEPS),
            "vanilla": _steps(VANILLA_STEPS),
            "gradual_phases": [
                {
                    "phase": phase,
                    "encoder_probability": encoder_prob,
                    "ent_coef": ent,
                    "learning_rate": lr,
                    "timesteps": _steps(steps),
                }
                for phase, encoder_prob, ent, lr, steps in GRADUAL_PHASES
            ],
        },
    }
    _write_manifest(run_root, payload)

    cfg = _base_cfg(base_cfg, exp_root, names["privileged"], PRIVILEGED_STEPS, system)
    _set_privileged(cfg, seed)
    cfg["training"]["load_weights"] = False
    cfg["training"]["load_weights_from"] = None
    cfg["training"]["load_encoder_only"] = False
    print(f"START {label}/privileged", flush=True)
    _run_with_config(cfg)

    cfg = _base_cfg(base_cfg, exp_root, names["encoder"], ENCODER_STEPS, system)
    _set_mle_encoder(cfg, seed)
    cfg["training"]["load_weights"] = True
    cfg["training"]["load_weights_from"] = str(dirs["privileged"].resolve())
    cfg["training"]["load_weights_name"] = "weights"
    cfg["training"]["load_encoder_only"] = False
    print(f"START {label}/mle_encoder", flush=True)
    _run_with_config(cfg)

    cfg = _base_cfg(base_cfg, exp_root, names["vanilla"], VANILLA_STEPS, system)
    _set_vanilla_policy(cfg, seed)
    cfg["training"]["load_weights"] = True
    cfg["training"]["load_weights_from"] = str(dirs["encoder"].resolve())
    cfg["training"]["load_weights_name"] = "weights"
    # The encoder experiment still contains the frozen privileged policy.
    # Loading both preserves the intended staged-RMA initialization.
    cfg["training"]["load_encoder_only"] = False
    print(f"START {label}/vanilla_rma", flush=True)
    _run_with_config(cfg)

    gradual_dir = dirs["gradual"]
    for phase_idx, (phase, encoder_prob, ent, lr, steps) in enumerate(GRADUAL_PHASES):
        cfg = _base_cfg(base_cfg, exp_root, names["gradual"], steps, system)
        _set_gradual_mle_curriculum(cfg, encoder_probability=encoder_prob, ent=ent, lr=lr)
        _set_common(cfg, seed)
        cfg["training"]["load_weights"] = phase_idx > 0
        cfg["training"]["load_weights_from"] = str(gradual_dir.resolve()) if phase_idx > 0 else None
        cfg["training"]["load_weights_name"] = "weights"
        cfg["training"]["load_encoder_only"] = False
        print(f"START {label}/gradual_mle/{phase}", flush=True)
        _run_with_config(cfg)
        _snapshot_phase_weights(gradual_dir, phase)

    return {"system": system, "names": names, "dirs": {key: str(value.resolve()) for key, value in dirs.items()}}


def _run_sweep(
    run: dict,
    *,
    theta_points: int | None = None,
    episodes_per_theta: int | None = None,
    output_subdir: str | None = None,
) -> Path:
    dirs = {key: Path(value) for key, value in run["dirs"].items()}
    system_label = run["system"]["label"]
    specs = [
        {
            "label": "privileged",
            "kind": "ppo",
            "experiment": str(dirs["privileged"]),
            "weights_name": "weights",
        },
        {
            "label": "vanilla_rma",
            "kind": "ppo",
            "experiment": str(dirs["vanilla"]),
            "weights_name": "weights",
        },
    ]
    for phase, *_ in GRADUAL_PHASES:
        specs.append(
            {
                "label": f"gradual_mle_{phase}",
                "kind": "ppo",
                "experiment": str(dirs["gradual"]),
                "weights_name": f"weights_{phase}",
            }
        )
    specs.extend(
        [
            {"label": "nominal_lqr", "kind": "nominal_lqr", "experiment": str(dirs["vanilla"])},
            {"label": "oracle_lqr", "kind": "lqr", "experiment": str(dirs["vanilla"])},
        ]
    )

    if output_subdir is None:
        output_subdir = "theta_sweep_neural_screening_smoke" if _smoke_enabled() else "theta_sweep_neural_screening"
    if theta_points is None:
        theta_points = 3 if _smoke_enabled() else 21
    if episodes_per_theta is None:
        episodes_per_theta = 1 if _smoke_enabled() else 10
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")
    env["THETA_SWEEP_TARGET_EXPERIMENT"] = str(dirs["vanilla"])
    env["THETA_SWEEP_OUTPUT_SUBDIR"] = output_subdir
    env["THETA_SWEEP_N_THETA_POINTS"] = str(theta_points)
    env["THETA_SWEEP_EPISODES_PER_THETA"] = str(episodes_per_theta)
    env["THETA_SWEEP_DETERMINISTIC"] = "1"
    env["THETA_SWEEP_RETURN_MODE"] = "quadratic"
    env["THETA_SWEEP_SAVE_TRAJECTORY_TRACE"] = "0"
    env["THETA_SWEEP_SAVE_STEP_LEVEL_CSV"] = "0"
    env["THETA_SWEEP_CONTROLLER_SPECS_JSON"] = json.dumps(specs)
    print(f"START {system_label}/theta_sweep", flush=True)
    subprocess.run(["python3", str(SWEEP_SCRIPT)], cwd=ROOT, env=env, check=True)
    return dirs["vanilla"] / output_subdir


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _score_sweep(run: dict, sweep_dir: Path) -> dict:
    scorecard = {row["controller"]: row for row in _read_csv(sweep_dir / "controller_scorecard.csv")}
    nominal = scorecard["nominal_lqr"]
    oracle = scorecard["oracle_lqr"]
    privileged = scorecard["privileged"]
    vanilla = scorecard["vanilla_rma"]
    gradual_rows = [scorecard[f"gradual_mle_{phase}"] for phase, *_ in GRADUAL_PHASES]

    def value(row: dict, key: str) -> float:
        return float(row[key])

    best_gradual = max(
        gradual_rows,
        key=lambda row: (
            value(row, "mean_return") - value(vanilla, "mean_return")
            + 0.25 * (value(row, "tail_mean_return") - value(vanilla, "tail_mean_return"))
        ),
    )
    gap = max(value(oracle, "mean_return") - value(nominal, "mean_return"), 1e-8)
    privileged_closure = (value(privileged, "mean_return") - value(nominal, "mean_return")) / gap
    vanilla_closure = (value(vanilla, "mean_return") - value(nominal, "mean_return")) / gap
    gradual_mean_gain = value(best_gradual, "mean_return") - value(vanilla, "mean_return")
    gradual_tail_gain = value(best_gradual, "tail_mean_return") - value(vanilla, "tail_mean_return")

    episode_rows = _read_csv(sweep_dir / "episode_summary.csv")
    best_label = best_gradual["controller"]
    best_episode_rows = [row for row in episode_rows if row["controller"] == best_label]
    best_returns = np.asarray([float(row["episode_return"]) for row in best_episode_rows], dtype=np.float64)
    failure_rate = float(np.mean(best_returns < -1e5)) if best_returns.size else float("nan")

    def split_tail_return(controller: str, positive: bool) -> float:
        values = [
            float(row["episode_return"])
            for row in episode_rows
            if row["controller"] == controller
            and ((float(row["theta"]) >= 0.15) if positive else (float(row["theta"]) <= -0.15))
        ]
        return float(np.mean(values))

    gradual_negative_tail = split_tail_return(best_label, positive=False)
    gradual_positive_tail = split_tail_return(best_label, positive=True)
    oracle_negative_tail = split_tail_return("oracle_lqr", positive=False)
    oracle_positive_tail = split_tail_return("oracle_lqr", positive=True)
    tail_asymmetry_deviation = abs(
        (gradual_negative_tail - gradual_positive_tail)
        - (oracle_negative_tail - oracle_positive_tail)
    )
    qualifies = (
        privileged_closure >= 0.75
        and vanilla_closure <= 0.35
        and value(vanilla, "theta_rmse") >= 0.1
        and gradual_mean_gain > 0.0
        and gradual_tail_gain > 0.0
        and failure_rate == 0.0
        and tail_asymmetry_deviation <= 20.0
    )
    return {
        "system": run["system"]["label"],
        "r22": run["system"]["r22"],
        "nominal_mean_return": value(nominal, "mean_return"),
        "oracle_mean_return": value(oracle, "mean_return"),
        "privileged_mean_return": value(privileged, "mean_return"),
        "vanilla_mean_return": value(vanilla, "mean_return"),
        "vanilla_tail_return": value(vanilla, "tail_mean_return"),
        "vanilla_theta_rmse": value(vanilla, "theta_rmse"),
        "best_gradual_checkpoint": best_label,
        "best_gradual_mean_return": value(best_gradual, "mean_return"),
        "best_gradual_tail_return": value(best_gradual, "tail_mean_return"),
        "best_gradual_theta_rmse": value(best_gradual, "theta_rmse"),
        "privileged_gap_closure": privileged_closure,
        "vanilla_gap_closure": vanilla_closure,
        "gradual_mean_gain_over_vanilla": gradual_mean_gain,
        "gradual_tail_gain_over_vanilla": gradual_tail_gain,
        "best_gradual_catastrophic_failure_rate": failure_rate,
        "best_gradual_negative_tail_return": gradual_negative_tail,
        "best_gradual_positive_tail_return": gradual_positive_tail,
        "best_gradual_tail_asymmetry_deviation_from_oracle": tail_asymmetry_deviation,
        "qualifies": qualifies,
        "selection_score": gradual_mean_gain + 0.25 * gradual_tail_gain,
        "sweep_dir": str(sweep_dir.resolve()),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    screening_dir, systems = _load_selected_systems()
    original_text = CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_text)
    exp_root = _resolve_exp_root(base_cfg)
    stamp = datetime.now().strftime("%m-%d__%H-%M")
    suffix = "smoke" if _smoke_enabled() else "full"
    run_root = SELECTION_ROOT / f"neural_{stamp}_{suffix}"
    run_root.mkdir(parents=True, exist_ok=True)
    runs = []

    try:
        for system in systems:
            runs.append(
                _train_system(
                    base_cfg=base_cfg,
                    exp_root=exp_root,
                    system=system,
                    stamp=stamp,
                    seed=0,
                    run_root=run_root,
                )
            )
    finally:
        CONFIG_PATH.write_text(original_text)

    with open(run_root / "neural_runs.json", "w") as f:
        json.dump({"screening_dir": str(screening_dir), "runs": runs}, f, indent=2)

    score_rows = []
    for run in runs:
        score_rows.append(_score_sweep(run, _run_sweep(run)))
    _write_csv(run_root / "neural_selection_scorecard.csv", score_rows)

    qualifying = [row for row in score_rows if row["qualifies"]]
    pool = qualifying if qualifying else score_rows
    selected_row = max(pool, key=lambda row: row["selection_score"])
    selected_run = next(run for run in runs if run["system"]["label"] == selected_row["system"])
    with open(run_root / "selected_system.json", "w") as f:
        json.dump(
            {
                "selection_criteria_met": bool(qualifying),
                "score": selected_row,
                "run": selected_run,
            },
            f,
            indent=2,
        )
    (SELECTION_ROOT / "latest_neural_screening.txt").write_text(str(run_root.resolve()) + "\n")
    print(f"Saved neural screening to: {run_root}", flush=True)
    print(f"Selected system: {selected_row['system']}", flush=True)
    if not qualifying:
        print("WARNING: no system met every selection criterion; selected highest score.", flush=True)


if __name__ == "__main__":
    main()
