from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

import run_two_action_selected_system_final_methods as staged_base
import run_two_action_selected_system_final_methods_discrete_theta as discrete
from run_two_action_pipeline import CONFIG_PATH, ROOT, TRAIN_CMD, _base_stage_cfg, _snapshot_phase_weights
from run_two_action_system_neural_screening import _set_common, _set_system


SELECTION_ROOT = ROOT / "experiments" / "two_action_system_selection"
RUN_FAMILY_ROOT = ROOT / "experiments" / "r22_discrete_blr_gradual"
POINTER = RUN_FAMILY_ROOT / "latest_r22_discrete_blr_gradual.txt"
SWEEP_SCRIPT = ROOT / "lqr_theta_sweep_eval.py"

SELECTED_SYSTEM_LABEL = "r22_1p5"
DISCRETE_THETA_VALUES = [-0.25, 0.0, 0.25]
DISCRETE_PRIOR_MEAN = 0.0
DISCRETE_PRIOR_VAR = (0.25**2 + 0.0 + 0.25**2) / 3.0
DEFAULT_SEEDS = [1, 2, 3]

TRAIN_NUM_ENVS = 8
TRAIN_VEC_ENV_TYPE = "dummy"
TRAIN_N_STEPS = 512
TRAIN_BATCH_SIZE = 1024

GRADUAL_PHASES = [
    ("P01", 0.25, 0.10, 3e-4, 2_000_000),
    ("P02", 0.40, 0.08, 3e-4, 2_000_000),
    ("P03", 0.55, 0.06, 2e-4, 2_000_000),
    ("P04", 0.70, 0.04, 2e-4, 2_000_000),
    ("P05", 0.85, 0.02, 1e-4, 2_000_000),
    ("P06", 0.95, 0.01, 1e-4, 2_000_000),
    ("P07", 1.00, 0.005, 1e-4, 2_000_000),
    ("P08", 1.00, 0.002, 5e-5, 2_000_000),
]


def _smoke_enabled() -> bool:
    return os.environ.get("R22_BLR_GRADUAL_SMOKE", "0").lower() in {"1", "true", "yes"}


def _steps(steps: int) -> int:
    return 16_384 if _smoke_enabled() else int(steps)


def _episodes_per_theta() -> int:
    return 1 if _smoke_enabled() else 20


def _selected_seeds() -> list[int]:
    raw = os.environ.get("R22_BLR_GRADUAL_SEEDS", "")
    if raw.strip():
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    return [1] if _smoke_enabled() else list(DEFAULT_SEEDS)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint(exp_dir: Path, weights_name: str = "weights") -> Path:
    path = exp_dir / weights_name
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def _require_checkpoint(exp_dir: Path, weights_name: str = "weights") -> None:
    path = _checkpoint(exp_dir, weights_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_selected_system() -> tuple[Path, dict]:
    pointer = SELECTION_ROOT / "latest_confirmation.txt"
    if not pointer.exists():
        raise FileNotFoundError("No latest confirmation pointer found. Run system confirmation first.")
    confirmation_dir = Path(pointer.read_text().strip()).resolve()
    freeze_path = confirmation_dir / "freeze_decision.json"
    if not freeze_path.exists():
        raise RuntimeError(f"Confirmation is incomplete: {confirmation_dir}")
    with freeze_path.open() as f:
        freeze = json.load(f)
    if not freeze.get("freeze_system", False):
        raise RuntimeError("Confirmation did not reproduce. Refusing to start BLR gradual run.")
    system = freeze["system"]
    if system["label"] != SELECTED_SYSTEM_LABEL:
        raise RuntimeError(f"Expected frozen system {SELECTED_SYSTEM_LABEL}, got {system['label']}.")
    return confirmation_dir, system


def _apply_standard_discrete_r22_settings(cfg: dict, system: dict) -> None:
    _set_system(cfg, system)
    discrete._set_discrete_theta_wrappers(cfg)
    cfg.setdefault("training", {})["num_envs"] = TRAIN_NUM_ENVS
    cfg.setdefault("training", {})["vec_env_type"] = TRAIN_VEC_ENV_TYPE
    lqr = cfg.setdefault("lqr_env", {})
    lqr["process_noise_std"] = 0.05
    lqr["max_episode_steps"] = 512
    params = cfg["model"]["params"]
    params["n_steps"] = TRAIN_N_STEPS
    params["batch_size"] = TRAIN_BATCH_SIZE
    params["window_length"] = 50
    params["nominal_warmup_steps"] = 49
    params["id_update_interval"] = 10
    params["use_transition_features"] = True
    params["transition_type"] = "delta"


def _base_cfg(base_cfg: dict, exp_root: Path, exp_name: str, steps: int, system: dict) -> dict:
    cfg = _base_stage_cfg(base_cfg, exp_root, exp_name, _steps(steps))
    _apply_standard_discrete_r22_settings(cfg, system)
    return cfg


def _set_load(cfg: dict, source: Path | None, weights_name: str = "weights") -> None:
    training = cfg["training"]
    training["load_weights"] = source is not None
    training["load_weights_from"] = str(source.resolve()) if source is not None else None
    training["load_weights_name"] = weights_name
    training["load_encoder_only"] = False


def _set_blr_stage(
    cfg: dict,
    *,
    seed: int,
    condition_on_uncertainty: bool,
    blr_probability: float,
    ent: float,
    lr: float,
) -> None:
    _set_common(cfg, seed)
    params = cfg["model"]["params"]
    params["context_mode"] = "closed_form"
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["policy_loss_coef"] = 1.0
    params["condition_on_uncertainty"] = bool(condition_on_uncertainty)
    params["privileged_uncertainty_mode"] = "zeros"
    params["privileged_context_probability"] = 1.0 - float(blr_probability)
    params["closed_form_prior_mean"] = DISCRETE_PRIOR_MEAN
    params["closed_form_prior_var"] = DISCRETE_PRIOR_VAR
    params["uncertainty_regularization_coef"] = 0.0
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["uncertainty_penalty_metric"] = "std"
    params["detach_context_for_rl"] = True
    params["naive_action_noise_std"] = 0.0
    params["naive_action_noise_dist"] = "gaussian"
    params["learning_rate"] = float(lr)
    params["ent_coef"] = float(ent)
    params["n_steps"] = TRAIN_N_STEPS
    params["batch_size"] = TRAIN_BATCH_SIZE
    params["window_length"] = 50
    params["nominal_warmup_steps"] = 49
    params["id_update_interval"] = 10
    params["use_transition_features"] = True
    params["transition_type"] = "delta"


def _theta_wrapper_params(cfg: dict) -> dict[str, Any]:
    for wrapper in cfg.get("wrappers", []):
        if wrapper.get("name") in {"DomainRandomizationWrapper", "ChangingCartPoleDynamics"}:
            return wrapper.get("params", {})
    return {}


def _assert_equal(actual: Any, expected: Any, key: str) -> None:
    if actual != expected:
        raise AssertionError(f"Config verification failed for {key}: expected {expected!r}, got {actual!r}")


def _assert_close(actual: Any, expected: float, key: str, *, tol: float = 1e-12) -> None:
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(f"Config verification failed for {key}: expected {expected!r}, got {actual!r}")


def _verify_written_config(expected: dict, *, phase: str) -> None:
    written = yaml.safe_load(CONFIG_PATH.read_text())
    params = written["model"]["params"]
    exp = expected["model"]["params"]
    training = written["training"]
    exp_training = expected["training"]
    wrapper = _theta_wrapper_params(written)

    for key in [
        "experiment_name",
        "load_weights",
        "load_weights_from",
        "load_weights_name",
        "load_encoder_only",
        "num_envs",
        "vec_env_type",
    ]:
        _assert_equal(training.get(key), exp_training.get(key), f"{phase}.training.{key}")
    _assert_equal(written.get("total_timesteps"), expected.get("total_timesteps"), f"{phase}.total_timesteps")
    _assert_equal(written.get("environment"), "lqr", f"{phase}.environment")
    _assert_equal(wrapper.get("randomize_theta"), True, f"{phase}.randomize_theta")
    _assert_equal(wrapper.get("randomize_mu"), False, f"{phase}.randomize_mu")
    _assert_equal(wrapper.get("randomize_a"), False, f"{phase}.randomize_a")
    _assert_equal(wrapper.get("randomize_b"), False, f"{phase}.randomize_b")
    _assert_equal(wrapper.get("randomize_process_noise_scale"), False, f"{phase}.randomize_process_noise_scale")
    _assert_equal(wrapper.get("theta_mult_range"), DISCRETE_THETA_VALUES, f"{phase}.theta_mult_range")
    _assert_equal(wrapper.get("categorical"), True, f"{phase}.categorical")
    _assert_close(written["lqr_env"].get("process_noise_std"), 0.05, f"{phase}.process_noise_std")
    _assert_equal(written["lqr_env"].get("max_episode_steps"), 512, f"{phase}.max_episode_steps")

    exact_param_keys = [
        "seed",
        "context_mode",
        "freeze_ppo",
        "regression_coef",
        "policy_loss_coef",
        "condition_on_uncertainty",
        "privileged_uncertainty_mode",
        "privileged_context_probability",
        "closed_form_prior_mean",
        "closed_form_prior_var",
        "uncertainty_regularization_coef",
        "uncertainty_reward_penalty_coef",
        "uncertainty_penalty_metric",
        "detach_context_for_rl",
        "naive_action_noise_std",
        "naive_action_noise_dist",
        "learning_rate",
        "ent_coef",
        "n_steps",
        "batch_size",
        "window_length",
        "nominal_warmup_steps",
        "id_update_interval",
        "use_transition_features",
        "transition_type",
    ]
    for key in exact_param_keys:
        _assert_equal(params.get(key), exp.get(key), f"{phase}.model.params.{key}")


def _run_stage(label: str, cfg: dict) -> None:
    print(f"START {label}: {cfg['training']['experiment_name']} steps={cfg['total_timesteps']}", flush=True)
    with CONFIG_PATH.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    _verify_written_config(cfg, phase=label)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")
    subprocess.run(TRAIN_CMD, cwd=ROOT, env=env, check=True)
    print(f"END   {label}: {cfg['training']['experiment_name']}", flush=True)


def _phase_cfg(
    *,
    base_cfg: dict,
    exp_root: Path,
    exp_name: str,
    steps: int,
    system: dict,
    seed: int,
    condition_on_uncertainty: bool,
    blr_probability: float,
    ent: float,
    lr: float,
    source: Path | None = None,
    weights_name: str = "weights",
) -> dict:
    cfg = _base_cfg(base_cfg, exp_root, exp_name, steps, system)
    _set_blr_stage(
        cfg,
        seed=seed,
        condition_on_uncertainty=condition_on_uncertainty,
        blr_probability=blr_probability,
        ent=ent,
        lr=lr,
    )
    _set_load(cfg, source, weights_name)
    return cfg


def _train_method(
    *,
    seed: int,
    method: str,
    condition_on_uncertainty: bool,
    base_cfg: dict,
    exp_root: Path,
    exp_name: str,
    exp_dir: Path,
    system: dict,
) -> list[dict[str, Any]]:
    phase_rows: list[dict[str, Any]] = []
    source: Path | None = None
    weights = "weights"
    for phase, probability, ent, lr, steps in GRADUAL_PHASES:
        cfg = _phase_cfg(
            base_cfg=base_cfg,
            exp_root=exp_root,
            exp_name=exp_name,
            steps=steps,
            system=system,
            seed=seed,
            condition_on_uncertainty=condition_on_uncertainty,
            blr_probability=probability,
            ent=ent,
            lr=lr,
            source=source,
            weights_name=weights,
        )
        _run_stage(f"seed{seed}/{method}/{phase}", cfg)
        _snapshot_phase_weights(exp_dir, phase)
        _require_checkpoint(exp_dir, f"weights_{phase}")
        source = exp_dir
        weights = f"weights_{phase}"
        phase_rows.append(
            {
                "method": method,
                "phase": phase,
                "blr_probability": probability,
                "privileged_probability": 1.0 - probability,
                "ent_coef": ent,
                "learning_rate": lr,
                "timesteps": _steps(steps),
            }
        )
    return phase_rows


def _validate_mean_pair(mean_dir: Path, mean_std_dir: Path) -> None:
    with (mean_dir / "config.yaml").open() as f:
        mean = yaml.safe_load(f)["model"]["params"]
    with (mean_std_dir / "config.yaml").open() as f:
        mean_std = yaml.safe_load(f)["model"]["params"]
    matched_keys = [
        "seed",
        "context_mode",
        "freeze_ppo",
        "regression_coef",
        "policy_loss_coef",
        "privileged_context_probability",
        "closed_form_prior_mean",
        "closed_form_prior_var",
        "uncertainty_regularization_coef",
        "uncertainty_reward_penalty_coef",
        "detach_context_for_rl",
        "n_steps",
        "batch_size",
        "window_length",
        "nominal_warmup_steps",
        "id_update_interval",
        "use_transition_features",
        "transition_type",
    ]
    mismatched = [key for key in matched_keys if mean.get(key) != mean_std.get(key)]
    if mismatched:
        raise RuntimeError(f"BLR mean and mean+std configs differ unexpectedly: {mismatched}")
    if mean.get("condition_on_uncertainty") or not mean_std.get("condition_on_uncertainty"):
        raise RuntimeError("BLR policies do not differ exactly by uncertainty conditioning.")
    if mean.get("context_mode") != "closed_form":
        raise RuntimeError("BLR experiment is not using closed_form context mode.")
    for exp_dir in [mean_dir, mean_std_dir]:
        for phase, *_ in GRADUAL_PHASES:
            _require_checkpoint(exp_dir, f"weights_{phase}")


def _train_seed(
    *,
    seed: int,
    base_cfg: dict,
    exp_root: Path,
    system: dict,
    stamp: str,
    smoke_suffix: str,
) -> dict[str, Any]:
    names = {
        "blr_mean": f"s_{stamp}_{system['label']}_discrete_gradual_blr_mean_seed{seed}{smoke_suffix}",
        "blr_mean_std": f"s_{stamp}_{system['label']}_discrete_gradual_blr_mean_std_seed{seed}{smoke_suffix}",
    }
    dirs = {key: exp_root / name for key, name in names.items()}
    phases: list[dict[str, Any]] = []
    phases.extend(
        _train_method(
            seed=seed,
            method="blr_mean",
            condition_on_uncertainty=False,
            base_cfg=base_cfg,
            exp_root=exp_root,
            exp_name=names["blr_mean"],
            exp_dir=dirs["blr_mean"],
            system=system,
        )
    )
    phases.extend(
        _train_method(
            seed=seed,
            method="blr_mean_std",
            condition_on_uncertainty=True,
            base_cfg=base_cfg,
            exp_root=exp_root,
            exp_name=names["blr_mean_std"],
            exp_dir=dirs["blr_mean_std"],
            system=system,
        )
    )
    _validate_mean_pair(dirs["blr_mean"], dirs["blr_mean_std"])
    return {
        "seed": seed,
        "names": names,
        "dirs": {key: str(path.resolve()) for key, path in dirs.items()},
        "phases": phases,
        "final_weights": {"blr_mean": "weights_P08", "blr_mean_std": "weights_P08"},
    }


def _ppo_spec(label: str, experiment: Path, weights_name: str, **extra: Any) -> dict[str, Any]:
    spec = {
        "label": label,
        "kind": "ppo",
        "experiment": str(experiment.resolve()),
        "weights_name": weights_name,
    }
    spec.update(extra)
    return spec


def _run_sweep(*, target: Path, specs: list[dict[str, Any]], output_subdir: str) -> Path:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")
    env["THETA_SWEEP_TARGET_EXPERIMENT"] = str(target.resolve())
    env["THETA_SWEEP_OUTPUT_SUBDIR"] = output_subdir
    env["THETA_SWEEP_THETA_VALUES"] = ",".join(str(value) for value in DISCRETE_THETA_VALUES)
    env["THETA_SWEEP_N_THETA_POINTS"] = str(len(DISCRETE_THETA_VALUES))
    env["THETA_SWEEP_EPISODES_PER_THETA"] = str(_episodes_per_theta())
    env["THETA_SWEEP_DETERMINISTIC"] = "1"
    env["THETA_SWEEP_RETURN_MODE"] = "quadratic"
    env["THETA_SWEEP_SAVE_TRAJECTORY_TRACE"] = "0"
    env["THETA_SWEEP_SAVE_STEP_LEVEL_CSV"] = "0"
    env["THETA_SWEEP_COLLECT_STEP_LEVEL_PREDICTIONS"] = "1"
    env["THETA_SWEEP_CONTROLLER_SPECS_JSON"] = json.dumps(specs)
    print(f"START BLR gradual theta sweep: {output_subdir}", flush=True)
    subprocess.run(["python3", str(SWEEP_SCRIPT)], cwd=ROOT, env=env, check=True)
    sweep_dir = target / output_subdir
    if not (sweep_dir / "controller_scorecard.csv").exists():
        raise RuntimeError(f"Sweep did not produce controller_scorecard.csv: {sweep_dir}")
    return sweep_dir


def _final_specs(seed_result: dict[str, Any]) -> list[dict[str, Any]]:
    seed = seed_result["seed"]
    dirs = {key: Path(value) for key, value in seed_result["dirs"].items()}
    return [
        _ppo_spec(f"blr_mean_P08_seed{seed}", dirs["blr_mean"], "weights_P08", method="blr_mean", seed=seed, checkpoint="P08"),
        _ppo_spec(
            f"blr_mean_std_P08_seed{seed}",
            dirs["blr_mean_std"],
            "weights_P08",
            method="blr_mean_std",
            seed=seed,
            checkpoint="P08",
        ),
    ]


def _all_checkpoint_specs(seed_result: dict[str, Any]) -> list[dict[str, Any]]:
    seed = seed_result["seed"]
    dirs = {key: Path(value) for key, value in seed_result["dirs"].items()}
    specs: list[dict[str, Any]] = []
    for method in ["blr_mean", "blr_mean_std"]:
        for phase, *_ in GRADUAL_PHASES:
            for prefix, kind in [("weights", "phase_end"), ("weights_best", "best_within_phase")]:
                weights_name = f"{prefix}_{phase}"
                if _checkpoint(dirs[method], weights_name).exists():
                    specs.append(
                        _ppo_spec(
                            f"{method}_{kind}_{phase}_seed{seed}",
                            dirs[method],
                            weights_name,
                            method=method,
                            seed=seed,
                            checkpoint=phase,
                            checkpoint_kind=kind,
                        )
                    )
    return specs


def _collect_scorecard_rows(sweep_dir: Path, *, seed: int, source: str) -> list[dict[str, Any]]:
    def as_float(value: Any) -> float:
        if value in {"", None}:
            return float("nan")
        return float(value)

    rows: list[dict[str, Any]] = []
    for row in _read_csv(sweep_dir / "controller_scorecard.csv"):
        controller = row["controller"]
        method = "unknown"
        for candidate in ["blr_mean_std", "blr_mean"]:
            if controller.startswith(candidate):
                method = candidate
                break
        rows.append(
            {
                "seed": seed,
                "source": source,
                "method": method,
                "controller": controller,
                "mean_return": as_float(row.get("mean_return", "nan")),
                "tail_mean_return": as_float(row.get("tail_mean_return", "nan")),
                "center_mean_return": as_float(row.get("center_mean_return", "nan")),
                "theta_rmse": as_float(row.get("theta_rmse", "nan")),
                "tail_theta_rmse": as_float(row.get("tail_theta_rmse", "nan")),
                "center_theta_rmse": as_float(row.get("center_theta_rmse", "nan")),
            }
        )
    return rows


def _write_best_by_method(run_root: Path, all_rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[(int(row["seed"]), str(row["method"]))].append(row)
    best_rows: list[dict[str, Any]] = []
    for (seed, method), rows in sorted(grouped.items()):
        best = max(rows, key=lambda row: float(row["mean_return"]))
        out = dict(best)
        out["selection_metric"] = "mean_return"
        best_rows.append(out)
    _write_csv(run_root / "all_checkpoint_best_by_method.csv", best_rows)


def _evaluate(run_root: Path, seed_results: list[dict[str, Any]]) -> None:
    final_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for result in seed_results:
        seed = int(result["seed"])
        target = Path(result["dirs"]["blr_mean"])
        final_dir = _run_sweep(
            target=target,
            specs=_final_specs(result),
            output_subdir=f"r22_discrete_blr_gradual_seed{seed}/final",
        )
        final_rows.extend(_collect_scorecard_rows(final_dir, seed=seed, source="final"))
        checkpoint_dir = _run_sweep(
            target=target,
            specs=_all_checkpoint_specs(result),
            output_subdir=f"r22_discrete_blr_gradual_seed{seed}/all_checkpoints",
        )
        checkpoint_rows.extend(_collect_scorecard_rows(checkpoint_dir, seed=seed, source="all_checkpoints"))
    _write_csv(run_root / "final_method_scorecard.csv", final_rows)
    _write_csv(run_root / "all_checkpoint_scorecard.csv", checkpoint_rows)
    _write_best_by_method(run_root, checkpoint_rows)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    confirmation_dir, system = _load_selected_system()
    original_text = CONFIG_PATH.read_text()
    original_hash = _file_sha256(CONFIG_PATH)
    base_cfg = yaml.safe_load(original_text)
    stamp = datetime.now().strftime("%m-%d__%H-%M")
    smoke_suffix = "_smoke" if _smoke_enabled() else ""
    seeds = _selected_seeds()

    if _smoke_enabled():
        run_root = Path("/private/tmp") / f"r22_discrete_blr_gradual_smoke_{stamp}"
        exp_root = run_root / "experiments"
    else:
        RUN_FAMILY_ROOT.mkdir(parents=True, exist_ok=True)
        run_root = RUN_FAMILY_ROOT / f"blr_gradual_{stamp}_{system['label']}"
        exp_root = ROOT / "experiments"

    payload: dict[str, Any] = {
        "confirmation_dir": str(confirmation_dir),
        "system": system,
        "smoke": _smoke_enabled(),
        "seeds": seeds,
        "theta_distribution": {"type": "categorical", "values": list(DISCRETE_THETA_VALUES)},
        "prior": {"mean": DISCRETE_PRIOR_MEAN, "variance": DISCRETE_PRIOR_VAR},
        "setup": {
            "process_noise_std": 0.05,
            "max_episode_steps": 512,
            "window_length": 50,
            "nominal_warmup_steps": 49,
            "id_update_interval": 10,
            "use_transition_features": True,
            "transition_type": "delta",
            "num_envs": TRAIN_NUM_ENVS,
            "vec_env_type": TRAIN_VEC_ENV_TYPE,
            "n_steps": TRAIN_N_STEPS,
            "batch_size": TRAIN_BATCH_SIZE,
        },
        "phase_budgets": [
            {
                "phase": phase,
                "timesteps": _steps(steps),
                "blr_probability": probability,
                "privileged_probability": 1.0 - probability,
                "learning_rate": lr,
                "ent_coef": ent,
            }
            for phase, probability, ent, lr, steps in GRADUAL_PHASES
        ],
        "methods": {
            "blr_mean": {"context_mode": "closed_form", "condition_on_uncertainty": False},
            "blr_mean_std": {"context_mode": "closed_form", "condition_on_uncertainty": True},
        },
        "seed_results": [],
    }
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = run_root / "blr_gradual_run.json"
    _write_manifest(manifest, payload)

    try:
        for seed in seeds:
            result = _train_seed(
                seed=seed,
                base_cfg=base_cfg,
                exp_root=exp_root,
                system=system,
                stamp=stamp,
                smoke_suffix=smoke_suffix,
            )
            payload["seed_results"].append(result)
            _write_manifest(manifest, payload)
    finally:
        CONFIG_PATH.write_text(original_text)
        if _file_sha256(CONFIG_PATH) != original_hash:
            raise RuntimeError("Root config.yaml was not restored byte-for-byte.")

    _evaluate(run_root, payload["seed_results"])
    payload["evaluation"] = {
        "final_method_scorecard": str((run_root / "final_method_scorecard.csv").resolve()),
        "all_checkpoint_scorecard": str((run_root / "all_checkpoint_scorecard.csv").resolve()),
        "all_checkpoint_best_by_method": str((run_root / "all_checkpoint_best_by_method.csv").resolve()),
    }
    _write_manifest(manifest, payload)

    if _smoke_enabled():
        required = [
            run_root / "final_method_scorecard.csv",
            run_root / "all_checkpoint_scorecard.csv",
            run_root / "all_checkpoint_best_by_method.csv",
        ]
        for path in required:
            if not path.exists():
                raise RuntimeError(f"Smoke verification failed, missing {path}")
        print(f"Smoke verification passed; removing {run_root}", flush=True)
        shutil.rmtree(run_root)
    else:
        POINTER.write_text(str(run_root.resolve()))
        print(f"Saved BLR gradual R22 discrete run to: {run_root}", flush=True)


if __name__ == "__main__":
    main()
