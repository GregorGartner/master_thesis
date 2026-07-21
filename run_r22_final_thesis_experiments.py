from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import run_two_action_selected_system_final_methods as checkpoint_utils
from run_two_action_gradual_encoder_curriculum import _set_gradual_encoder_curriculum
from run_two_action_gradual_mle_encoder_curriculum import _set_gradual_mle_curriculum
from run_two_action_pipeline import CONFIG_PATH, ROOT, TRAIN_CMD, _base_stage_cfg, _snapshot_phase_weights
from run_two_action_system_neural_screening import SELECTION_ROOT, _set_common, _set_system


RUN_FAMILY_ROOT = ROOT / "experiments" / "r22_final_thesis"
POINTER = RUN_FAMILY_ROOT / "latest_r22_final_thesis.txt"
SWEEP_SCRIPT = ROOT / "lqr_theta_sweep_eval.py"
SELECTED_SYSTEM_LABEL = "r22_1p5"
THETA_LOW = -0.25
THETA_HIGH = 0.25
DISCRETE_THETA_VALUES = [-0.25, 0.0, 0.25]

DEFAULT_SEEDS = [1, 2, 3]
TRAIN_NUM_ENVS = 8
TRAIN_VEC_ENV_TYPE = "dummy"
TRAIN_N_STEPS = 512
TRAIN_BATCH_SIZE = 1024
WINDOW_LENGTH = 50
WARMUP_STEPS = 49
ID_UPDATE_INTERVAL = 10
VALIDATION_BASE_SEED = 12_345
HELDOUT_BASE_SEED = 98_765

CATEGORICAL_PHASES = [
    ("P01", 0.25, 0.10, 3e-4, 2_000_000),
    ("P02", 0.40, 0.08, 3e-4, 2_000_000),
    ("P03", 0.55, 0.06, 2e-4, 2_000_000),
    ("P04", 0.70, 0.04, 2e-4, 2_000_000),
    ("P05", 0.85, 0.02, 1e-4, 2_000_000),
    ("P06", 0.95, 0.01, 1e-4, 2_000_000),
    ("P07", 1.00, 0.005, 1e-4, 2_000_000),
    ("P08", 1.00, 0.002, 5e-5, 2_000_000),
]
CATEGORICAL_METHODS = {
    "gradual_mle_mean_only": ("mle", False),
    "gradual_nll_mean_only": ("nll", False),
    "gradual_nll_mean_std": ("nll", True),
}

PRIVILEGED_PHASES = [
    ("PRIV01", 3_000_000, 3e-4, 0.02),
    ("PRIV02", 4_000_000, 2e-4, 0.01),
    ("PRIV03", 4_000_000, 1e-4, 0.005),
]
SHARED_ENCODER_PHASES = [
    ("E01", 4_000_000, 3e-4),
    ("E02", 3_000_000, 1e-4),
]
TWO_STAGE_EXTENSION = [("E03", 5_000_000, 1e-4)]
THREE_STAGE_POLICY_PHASES = [
    ("S01", 2_500_000, 2e-4, 0.01),
    ("S02", 2_500_000, 1e-4, 0.002),
]
CONTINUOUS_GRADUAL_PHASES = [
    ("P01", 0.25, 0.10, 3e-4, 3_000_000),
    ("P02", 0.50, 0.08, 3e-4, 3_000_000),
    ("P03", 0.75, 0.05, 2e-4, 3_000_000),
    ("P04", 1.00, 0.02, 1e-4, 4_000_000),
    ("P05", 1.00, 0.005, 1e-4, 4_000_000),
    ("P06", 1.00, 0.002, 1e-4, 4_000_000),
]


def _truthy(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes"}


def _smoke_enabled() -> bool:
    return _truthy("R22_FINAL_THESIS_SMOKE")


def _steps(value: int) -> int:
    # One complete PPO rollout with 8 envs x 512 steps.
    return 4_096 if _smoke_enabled() else int(value)


def _episodes(validation: bool, *, continuous: bool) -> int:
    if _smoke_enabled():
        return 1
    if validation:
        return 10 if continuous else 20
    return 20 if continuous else 100


def _selected_seeds() -> list[int]:
    raw = os.environ.get("R22_FINAL_THESIS_SEEDS", "")
    if raw.strip():
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    return [1] if _smoke_enabled() else list(DEFAULT_SEEDS)


def _selected_studies() -> list[str]:
    raw = os.environ.get("R22_FINAL_THESIS_STUDIES", "categorical,continuous")
    studies = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(studies) - {"categorical", "continuous"})
    if unknown:
        raise ValueError(f"Unknown R22_FINAL_THESIS_STUDIES values: {unknown}")
    return studies


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(exp_dir: Path, weights_name: str = "weights") -> Path:
    path = exp_dir / weights_name
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def _require_checkpoint(exp_dir: Path, weights_name: str) -> None:
    path = _checkpoint(exp_dir, weights_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")


def _load_selected_system() -> tuple[Path, dict[str, Any]]:
    pointer = SELECTION_ROOT / "latest_confirmation.txt"
    if not pointer.exists():
        raise FileNotFoundError("No frozen two-action system confirmation pointer exists.")
    confirmation_dir = Path(pointer.read_text().strip()).resolve()
    freeze_path = confirmation_dir / "freeze_decision.json"
    with freeze_path.open() as handle:
        freeze = json.load(handle)
    if not freeze.get("freeze_system", False):
        raise RuntimeError(f"System confirmation is not frozen: {confirmation_dir}")
    system = freeze["system"]
    if system.get("label") != SELECTED_SYSTEM_LABEL:
        raise RuntimeError(f"Expected {SELECTED_SYSTEM_LABEL}, got {system.get('label')}")
    return confirmation_dir, system


def _domain_wrapper(*, categorical: bool) -> dict[str, Any]:
    theta_range: list[float] = list(DISCRETE_THETA_VALUES) if categorical else [THETA_LOW, THETA_HIGH]
    return {
        "name": "DomainRandomizationWrapper",
        "enabled": True,
        "params": {
            "change_prob": 1.0,
            "only_at_episode_end": True,
            "randomize_on_reset": True,
            "randomize_theta": True,
            "theta_mult_range": theta_range,
            # The adapter validates every range even when its randomizer is
            # disabled, so keep inactive families explicit and continuous-safe.
            "mu_range": [0.25, 0.75],
            "a_range": [0.75, 1.05],
            "b_range": [0.4, 1.8],
            "process_noise_scale_mult_range": [1.0, 1.0],
            "categorical": bool(categorical),
            "randomize_mu": False,
            "randomize_a": False,
            "randomize_b": False,
            "randomize_process_noise_scale": False,
        },
    }


def _canonical_base_cfg(
    base_cfg: dict[str, Any],
    *,
    exp_root: Path,
    exp_name: str,
    steps: int,
    system: dict[str, Any],
    seed: int,
    categorical: bool,
) -> dict[str, Any]:
    cfg = _base_stage_cfg(base_cfg, exp_root, exp_name, _steps(steps))
    _set_system(cfg, system)
    _set_common(cfg, seed)
    cfg["wrappers"] = [_domain_wrapper(categorical=categorical)]
    cfg["training"]["experiment_name_suffix"] = None
    cfg["training"]["num_envs"] = TRAIN_NUM_ENVS
    cfg["training"]["vec_env_type"] = TRAIN_VEC_ENV_TYPE
    lqr = cfg["lqr_env"]
    lqr["process_noise_std"] = 0.05
    lqr["initial_state_low"] = -0.3
    lqr["initial_state_high"] = 0.3
    lqr["max_episode_steps"] = 512
    params = cfg["model"]["params"]
    params.pop("initial_context", None)
    params.pop("action_log_std_init", None)
    params["n_steps"] = TRAIN_N_STEPS
    params["batch_size"] = TRAIN_BATCH_SIZE
    params["n_epochs"] = 8
    params["gamma"] = 0.995
    params["window_length"] = WINDOW_LENGTH
    params["nominal_warmup_steps"] = WARMUP_STEPS
    params["id_update_interval"] = ID_UPDATE_INTERVAL
    params["use_transition_features"] = True
    params["transition_type"] = "delta"
    params["detach_context_for_rl"] = True
    params["regression_param_names"] = ["theta"]
    params["latent_dim"] = 1
    params["z_scale"] = 10.0
    params["naive_action_noise_std"] = 0.0
    params["naive_action_noise_dist"] = "gaussian"
    params["uncertainty_regularization_coef"] = 0.0
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["uncertainty_penalty_metric"] = "std"
    params["privileged_uncertainty_value"] = 0.0
    params["policy_loss_coef"] = 1.0
    return cfg


def _set_load(cfg: dict[str, Any], source: Path | None, weights_name: str = "weights") -> None:
    training = cfg["training"]
    training["load_weights"] = source is not None
    training["load_weights_from"] = str(source.resolve()) if source is not None else None
    training["load_weights_name"] = weights_name
    training["load_encoder_only"] = False


def _set_privileged(cfg: dict[str, Any], *, lr: float, ent: float) -> None:
    params = cfg["model"]["params"]
    params["context_mode"] = "privileged"
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["policy_loss_coef"] = 1.0
    params["condition_on_uncertainty"] = False
    params["privileged_uncertainty_mode"] = "zeros"
    params["privileged_context_probability"] = 0.0
    params["learning_rate"] = float(lr)
    params["ent_coef"] = float(ent)


def _set_encoder_only(cfg: dict[str, Any], *, lr: float) -> None:
    params = cfg["model"]["params"]
    params["context_mode"] = "encoder_mle"
    params["freeze_ppo"] = True
    params["regression_coef"] = 1.0
    params["policy_loss_coef"] = 0.0
    params["condition_on_uncertainty"] = False
    params["privileged_uncertainty_mode"] = "zeros"
    params["privileged_context_probability"] = 0.0
    params["learning_rate"] = float(lr)
    params["ent_coef"] = 0.0
    params["naive_action_noise_std"] = 0.0


def _set_policy_only(cfg: dict[str, Any], *, lr: float, ent: float) -> None:
    params = cfg["model"]["params"]
    params["context_mode"] = "encoder_mle"
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["policy_loss_coef"] = 1.0
    params["condition_on_uncertainty"] = False
    params["privileged_uncertainty_mode"] = "zeros"
    params["privileged_context_probability"] = 0.0
    params["learning_rate"] = float(lr)
    params["ent_coef"] = float(ent)
    params["naive_action_noise_std"] = 0.0


def _set_gradual(
    cfg: dict[str, Any],
    *,
    kind: str,
    condition_on_uncertainty: bool,
    encoder_probability: float,
    lr: float,
    ent: float,
) -> None:
    if kind == "mle":
        _set_gradual_mle_curriculum(
            cfg,
            encoder_probability=encoder_probability,
            ent=ent,
            lr=lr,
        )
    elif kind == "nll":
        _set_gradual_encoder_curriculum(
            cfg,
            encoder_probability=encoder_probability,
            ent=ent,
            lr=lr,
        )
    else:
        raise ValueError(f"Unknown gradual kind: {kind}")
    params = cfg["model"]["params"]
    params["condition_on_uncertainty"] = bool(condition_on_uncertainty)
    params["privileged_uncertainty_mode"] = "predicted" if kind == "nll" else "zeros"
    params["privileged_context_probability"] = 1.0 - float(encoder_probability)
    params["freeze_ppo"] = False
    params["regression_coef"] = 1.0
    params["policy_loss_coef"] = 1.0
    params["naive_action_noise_std"] = 0.0
    params["naive_action_noise_dist"] = "gaussian"
    params["n_steps"] = TRAIN_N_STEPS
    params["batch_size"] = TRAIN_BATCH_SIZE


VERIFY_PARAM_KEYS = [
    "seed",
    "context_mode",
    "freeze_ppo",
    "regression_coef",
    "policy_loss_coef",
    "condition_on_uncertainty",
    "privileged_uncertainty_mode",
    "privileged_context_probability",
    "learning_rate",
    "ent_coef",
    "n_steps",
    "batch_size",
    "n_epochs",
    "window_length",
    "nominal_warmup_steps",
    "id_update_interval",
    "use_transition_features",
    "transition_type",
    "detach_context_for_rl",
    "regression_param_names",
    "latent_dim",
    "z_scale",
    "encoder_net_arch",
    "actor_net_arch",
    "critic_net_arch",
    "naive_action_noise_std",
    "naive_action_noise_dist",
    "uncertainty_reward_penalty_coef",
]


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def _verify_written_config(expected: dict[str, Any], *, phase: str) -> None:
    written = yaml.safe_load(CONFIG_PATH.read_text())
    for key in ["environment", "total_timesteps"]:
        _assert_equal(written.get(key), expected.get(key), f"{phase}.{key}")
    for key in [
        "experiment_root",
        "experiment_name",
        "load_weights",
        "load_weights_from",
        "load_weights_name",
        "load_encoder_only",
        "num_envs",
        "vec_env_type",
    ]:
        _assert_equal(written["training"].get(key), expected["training"].get(key), f"{phase}.training.{key}")
    for key in ["A", "B", "delta_B", "Q", "R", "process_noise_std", "initial_state_low", "initial_state_high", "max_episode_steps"]:
        _assert_equal(written["lqr_env"].get(key), expected["lqr_env"].get(key), f"{phase}.lqr_env.{key}")
    _assert_equal(len(written.get("wrappers", [])), 1, f"{phase}.wrapper_count")
    _assert_equal(written["wrappers"][0], expected["wrappers"][0], f"{phase}.domain_wrapper")
    params = written["model"]["params"]
    expected_params = expected["model"]["params"]
    for key in VERIFY_PARAM_KEYS:
        _assert_equal(params.get(key), expected_params.get(key), f"{phase}.model.params.{key}")
    for forbidden in ["initial_context", "action_log_std_init"]:
        if forbidden in params:
            raise AssertionError(f"{phase}: stale parameter {forbidden!r} survived canonicalization")


def _run_stage(label: str, cfg: dict[str, Any]) -> None:
    print(f"START {label}: {cfg['training']['experiment_name']} steps={cfg['total_timesteps']}", flush=True)
    CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))
    _verify_written_config(cfg, phase=label)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    subprocess.run(TRAIN_CMD, cwd=ROOT, env=env, check=True)
    print(f"END   {label}", flush=True)


def _train_gradual_method(
    *,
    base_cfg: dict[str, Any],
    system: dict[str, Any],
    exp_root: Path,
    exp_name: str,
    exp_dir: Path,
    seed: int,
    categorical: bool,
    kind: str,
    condition_on_uncertainty: bool,
    phases: list[tuple[str, float, float, float, int]],
) -> list[str]:
    completed: list[str] = []
    for index, (phase, probability, ent, lr, steps) in enumerate(phases):
        cfg = _canonical_base_cfg(
            base_cfg,
            exp_root=exp_root,
            exp_name=exp_name,
            steps=steps,
            system=system,
            seed=seed,
            categorical=categorical,
        )
        _set_gradual(
            cfg,
            kind=kind,
            condition_on_uncertainty=condition_on_uncertainty,
            encoder_probability=probability,
            lr=lr,
            ent=ent,
        )
        _set_load(cfg, exp_dir if index else None)
        _run_stage(f"seed{seed}/{exp_name}/{phase}", cfg)
        _snapshot_phase_weights(exp_dir, phase)
        _require_checkpoint(exp_dir, f"weights_{phase}")
        completed.append(phase)
    return completed


def _train_continuous_staged_seed(
    *,
    base_cfg: dict[str, Any],
    system: dict[str, Any],
    exp_root: Path,
    seed: int,
) -> dict[str, Any]:
    names = {
        "privileged": f"continuous_shared_privileged_seed{seed}",
        "shared_encoder": f"continuous_shared_mle_encoder_seed{seed}",
        "two_stage": f"continuous_two_stage_rma_seed{seed}",
        "three_stage": f"continuous_three_stage_arma_seed{seed}",
    }
    dirs = {key: exp_root / value for key, value in names.items()}

    source: Path | None = None
    weights_name = "weights"
    for phase, steps, lr, ent in PRIVILEGED_PHASES:
        cfg = _canonical_base_cfg(
            base_cfg,
            exp_root=exp_root,
            exp_name=names["privileged"],
            steps=steps,
            system=system,
            seed=seed,
            categorical=False,
        )
        _set_privileged(cfg, lr=lr, ent=ent)
        _set_load(cfg, source, weights_name)
        _run_stage(f"continuous/seed{seed}/privileged/{phase}", cfg)
        _snapshot_phase_weights(dirs["privileged"], phase)
        _require_checkpoint(dirs["privileged"], f"weights_{phase}")
        source = dirs["privileged"]
        weights_name = f"weights_{phase}"

    privileged_reference = _checkpoint(dirs["privileged"], "weights_PRIV03")
    for phase, steps, lr in SHARED_ENCODER_PHASES:
        cfg = _canonical_base_cfg(
            base_cfg,
            exp_root=exp_root,
            exp_name=names["shared_encoder"],
            steps=steps,
            system=system,
            seed=seed,
            categorical=False,
        )
        _set_encoder_only(cfg, lr=lr)
        _set_load(cfg, source, weights_name)
        _run_stage(f"continuous/seed{seed}/shared_encoder/{phase}", cfg)
        _snapshot_phase_weights(dirs["shared_encoder"], phase)
        current = _checkpoint(dirs["shared_encoder"], f"weights_{phase}")
        checkpoint_utils._assert_same_policy_group(
            privileged_reference,
            current,
            encoder=False,
            description=f"continuous seed {seed} encoder phase {phase} changed the frozen policy",
        )
        source = dirs["shared_encoder"]
        weights_name = f"weights_{phase}"

    shared_encoder_reference = _checkpoint(dirs["shared_encoder"], "weights_E02")

    source = dirs["shared_encoder"]
    weights_name = "weights_E02"
    for phase, steps, lr in TWO_STAGE_EXTENSION:
        cfg = _canonical_base_cfg(
            base_cfg,
            exp_root=exp_root,
            exp_name=names["two_stage"],
            steps=steps,
            system=system,
            seed=seed,
            categorical=False,
        )
        _set_encoder_only(cfg, lr=lr)
        _set_load(cfg, source, weights_name)
        _run_stage(f"continuous/seed{seed}/two_stage/{phase}", cfg)
        _snapshot_phase_weights(dirs["two_stage"], phase)
        current = _checkpoint(dirs["two_stage"], f"weights_{phase}")
        checkpoint_utils._assert_same_policy_group(
            privileged_reference,
            current,
            encoder=False,
            description=f"continuous seed {seed} two-stage extension changed the frozen policy",
        )
        source = dirs["two_stage"]
        weights_name = f"weights_{phase}"

    source = dirs["shared_encoder"]
    weights_name = "weights_E02"
    for phase, steps, lr, ent in THREE_STAGE_POLICY_PHASES:
        cfg = _canonical_base_cfg(
            base_cfg,
            exp_root=exp_root,
            exp_name=names["three_stage"],
            steps=steps,
            system=system,
            seed=seed,
            categorical=False,
        )
        _set_policy_only(cfg, lr=lr, ent=ent)
        _set_load(cfg, source, weights_name)
        _run_stage(f"continuous/seed{seed}/three_stage/{phase}", cfg)
        _snapshot_phase_weights(dirs["three_stage"], phase)
        current = _checkpoint(dirs["three_stage"], f"weights_{phase}")
        checkpoint_utils._assert_same_policy_group(
            shared_encoder_reference,
            current,
            encoder=True,
            description=f"continuous seed {seed} policy phase {phase} changed the frozen encoder",
        )
        source = dirs["three_stage"]
        weights_name = f"weights_{phase}"

    return {
        "seed": seed,
        "names": names,
        "dirs": {key: str(value.resolve()) for key, value in dirs.items()},
    }


def _ppo_spec(
    label: str,
    experiment: Path,
    weights_name: str,
    **extra: Any,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "label": label,
        "kind": "ppo",
        "experiment": str(experiment.resolve()),
        "weights_name": weights_name,
    }
    spec.update(extra)
    return spec


def _run_sweep(
    *,
    target: Path,
    specs: list[dict[str, Any]],
    output_subdir: str,
    categorical: bool,
    validation: bool,
    collect_predictions: bool = True,
) -> Path:
    specs = list(specs)
    labels = {str(spec["label"]) for spec in specs}
    if "oracle_lqr" not in labels:
        specs.append({"label": "oracle_lqr", "kind": "lqr", "experiment": str(target.resolve())})
    if "nominal_lqr" not in labels:
        specs.append({"label": "nominal_lqr", "kind": "nominal_lqr", "experiment": str(target.resolve())})
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env["THETA_SWEEP_TARGET_EXPERIMENT"] = str(target.resolve())
    env["THETA_SWEEP_OUTPUT_SUBDIR"] = output_subdir
    env["THETA_SWEEP_EPISODES_PER_THETA"] = str(
        _episodes(validation, continuous=not categorical)
    )
    env["THETA_SWEEP_EVAL_BASE_SEED"] = str(
        VALIDATION_BASE_SEED if validation else HELDOUT_BASE_SEED
    )
    if categorical:
        env["THETA_SWEEP_THETA_VALUES"] = ",".join(str(value) for value in DISCRETE_THETA_VALUES)
        env["THETA_SWEEP_N_THETA_POINTS"] = str(len(DISCRETE_THETA_VALUES))
    else:
        env.pop("THETA_SWEEP_THETA_VALUES", None)
        env["THETA_SWEEP_N_THETA_POINTS"] = "3" if _smoke_enabled() else ("21" if validation else "41")
    env["THETA_SWEEP_DETERMINISTIC"] = "1"
    env["THETA_SWEEP_RETURN_MODE"] = "quadratic"
    env["THETA_SWEEP_SAVE_TRAJECTORY_TRACE"] = "0"
    env["THETA_SWEEP_SAVE_STEP_LEVEL_CSV"] = "0"
    env["THETA_SWEEP_COLLECT_STEP_LEVEL_PREDICTIONS"] = "1" if collect_predictions else "0"
    env["THETA_SWEEP_CONTROLLER_SPECS_JSON"] = json.dumps(specs)
    print(f"START theta sweep: {output_subdir}", flush=True)
    subprocess.run(["python3", str(SWEEP_SCRIPT)], cwd=ROOT, env=env, check=True)
    output = target / output_subdir
    if not (output / "controller_scorecard.csv").exists():
        raise RuntimeError(f"Sweep did not create a scorecard: {output}")
    return output


def _scorecard_rows(
    sweep_dir: Path,
    metadata: dict[str, dict[str, Any]],
    *,
    source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(sweep_dir / "controller_scorecard.csv"):
        label = row["controller"]
        if label not in metadata:
            continue
        meta = metadata.get(label, {})
        out: dict[str, Any] = {
            **meta,
            "source": source,
            "controller": label,
        }
        for key in [
            "mean_return",
            "tail_mean_return",
            "center_mean_return",
            "theta_rmse",
            "tail_theta_rmse",
            "center_theta_rmse",
            "mean_info_proxy",
            "tail_mean_info_proxy",
            "center_mean_info_proxy",
        ]:
            value = row.get(key)
            out[key] = float(value) if value not in {None, ""} else float("nan")
        rows.append(out)
    return rows


def _aggregate_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    output: list[dict[str, Any]] = []
    for method, group in sorted(grouped.items()):
        result: dict[str, Any] = {"method": method, "num_seeds": len(group)}
        for key in [
            "mean_return",
            "tail_mean_return",
            "center_mean_return",
            "theta_rmse",
            "tail_theta_rmse",
            "center_theta_rmse",
            "mean_info_proxy",
            "tail_mean_info_proxy",
            "center_mean_info_proxy",
        ]:
            values = np.asarray([float(row[key]) for row in group], dtype=np.float64)
            finite = values[np.isfinite(values)]
            result[f"{key}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            result[f"{key}_std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        output.append(result)
    return output


def _checkpoint_specs(
    *,
    seed: int,
    method: str,
    exp_dir: Path,
    phases: list[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    specs: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for phase in phases:
        for prefix, kind in [("weights", "phase_end"), ("weights_best", "best_within_phase")]:
            weights_name = f"{prefix}_{phase}"
            if not _checkpoint(exp_dir, weights_name).exists():
                continue
            label = f"{method}__{kind}__{phase}__seed{seed}"
            specs.append(_ppo_spec(label, exp_dir, weights_name))
            metadata[label] = {
                "seed": seed,
                "method": method,
                "checkpoint": phase,
                "checkpoint_kind": kind,
                "experiment": str(exp_dir.resolve()),
                "weights_name": weights_name,
            }
    return specs, metadata


def _select_best(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), str(row["method"]))].append(row)
    selected: list[dict[str, Any]] = []
    for _, group in sorted(grouped.items()):
        best = max(group, key=lambda row: float(row["mean_return"]))
        result = dict(best)
        result["selection_metric"] = "validation_mean_return"
        result["validation_base_seed"] = VALIDATION_BASE_SEED
        selected.append(result)
    return selected


def _heldout_specs(
    selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    specs: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for row in selected:
        label = f"{row['method']}__locked__seed{row['seed']}"
        specs.append(
            _ppo_spec(
                label,
                Path(str(row["experiment"])),
                str(row["weights_name"]),
            )
        )
        metadata[label] = {
            "seed": int(row["seed"]),
            "method": str(row["method"]),
            "selected_checkpoint": str(row["checkpoint"]),
            "selected_checkpoint_kind": str(row["checkpoint_kind"]),
            "experiment": str(row["experiment"]),
            "weights_name": str(row["weights_name"]),
            "selection_validation_mean_return": float(row["mean_return"]),
        }
    return specs, metadata


def _checkpoint_specs_from_sources(
    *,
    seed: int,
    method: str,
    sources: list[tuple[Path, list[str]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    specs: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for exp_dir, phases in sources:
        source_specs, source_metadata = _checkpoint_specs(
            seed=seed,
            method=method,
            exp_dir=exp_dir,
            phases=phases,
        )
        specs.extend(source_specs)
        overlap = set(metadata).intersection(source_metadata)
        if overlap:
            raise RuntimeError(f"Duplicate checkpoint labels: {sorted(overlap)}")
        metadata.update(source_metadata)
    return specs, metadata


def _final_spec(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    exp_dir = Path(record["final_experiment"])
    weights_name = str(record["final_weights_name"])
    seed = int(record["seed"])
    method = str(record["method"])
    label = f"{method}__final__seed{seed}"
    spec = _ppo_spec(label, exp_dir, weights_name)
    metadata = {
        label: {
            "seed": seed,
            "method": method,
            "checkpoint": str(record["final_checkpoint"]),
            "checkpoint_kind": "phase_end",
            "experiment": str(exp_dir.resolve()),
            "weights_name": weights_name,
        }
    }
    return spec, metadata


def _matched_constant_uncertainty(
    sweep_dir: Path,
    *,
    controller: str,
    experiment: Path,
) -> float:
    values = [
        float(row["pred_theta_std_tail_mean_mean"])
        for row in _read_csv(sweep_dir / "theta_sweep_aggregate.csv")
        if row.get("controller") == controller
        and row.get("pred_theta_std_tail_mean_mean") not in {None, ""}
    ]
    if not values:
        return 0.0
    with (experiment / "config.yaml").open() as handle:
        z_scale = float(yaml.safe_load(handle)["model"]["params"].get("z_scale", 1.0))
    return float(np.mean(values) * z_scale)


def _paired_deltas(rows: list[dict[str, Any]], pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    lookup = {(int(row["seed"]), str(row["method"])): row for row in rows}
    output: list[dict[str, Any]] = []
    for numerator, denominator in pairs:
        seeds = sorted(
            seed
            for seed, method in lookup
            if method == numerator and (seed, denominator) in lookup
        )
        for seed in seeds:
            row_a = lookup[(seed, numerator)]
            row_b = lookup[(seed, denominator)]
            result: dict[str, Any] = {
                "seed": seed,
                "comparison": f"{numerator}_minus_{denominator}",
                "numerator": numerator,
                "denominator": denominator,
            }
            for key in [
                "mean_return",
                "tail_mean_return",
                "center_mean_return",
                "theta_rmse",
                "tail_theta_rmse",
                "center_theta_rmse",
                "mean_info_proxy",
                "tail_mean_info_proxy",
                "center_mean_info_proxy",
            ]:
                result[f"delta_{key}"] = float(row_a[key]) - float(row_b[key])
            output.append(result)
    return output


def _evaluate_records(
    *,
    study_root: Path,
    records: list[dict[str, Any]],
    categorical: bool,
    paired_comparisons: list[tuple[str, str]],
    run_uncertainty_ablation: bool,
) -> dict[str, Any]:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_seed[int(record["seed"])].append(record)

    validation_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    validation_dirs: dict[int, Path] = {}
    for seed, seed_records in sorted(by_seed.items()):
        all_specs: list[dict[str, Any]] = []
        all_metadata: dict[str, dict[str, Any]] = {}
        final_specs: list[dict[str, Any]] = []
        final_metadata: dict[str, dict[str, Any]] = {}
        for record in seed_records:
            sources = [
                (Path(source["experiment"]), list(source["phases"]))
                for source in record["sources"]
            ]
            specs, metadata = _checkpoint_specs_from_sources(
                seed=seed,
                method=str(record["method"]),
                sources=sources,
            )
            all_specs.extend(specs)
            all_metadata.update(metadata)
            final_spec, metadata = _final_spec(record)
            final_specs.append(final_spec)
            final_metadata.update(metadata)

        if not all_specs:
            raise RuntimeError(f"No checkpoints found for seed {seed} in {study_root}")
        target = Path(seed_records[0]["final_experiment"])
        all_dir = _run_sweep(
            target=target,
            specs=all_specs,
            output_subdir=f"final_thesis/{study_root.name}/seed{seed}/validation_all_checkpoints",
            categorical=categorical,
            validation=True,
        )
        validation_dirs[seed] = all_dir
        validation_rows.extend(
            _scorecard_rows(all_dir, all_metadata, source="validation_all_checkpoints")
        )

        final_dir = _run_sweep(
            target=target,
            specs=final_specs,
            output_subdir=f"final_thesis/{study_root.name}/seed{seed}/validation_final_checkpoints",
            categorical=categorical,
            validation=True,
        )
        final_rows.extend(_scorecard_rows(final_dir, final_metadata, source="validation_final"))

    selected = _select_best(validation_rows)
    _write_csv(study_root / "all_checkpoint_validation_scorecard.csv", validation_rows)
    _write_csv(study_root / "final_method_scorecard.csv", final_rows)
    _write_csv(study_root / "best_checkpoint_validation_scorecard.csv", selected)

    heldout_rows: list[dict[str, Any]] = []
    for seed, seed_records in sorted(by_seed.items()):
        seed_selected = [row for row in selected if int(row["seed"]) == seed]
        specs, metadata = _heldout_specs(seed_selected)
        target = Path(seed_records[0]["final_experiment"])
        heldout_dir = _run_sweep(
            target=target,
            specs=specs,
            output_subdir=f"final_thesis/{study_root.name}/seed{seed}/locked_heldout",
            categorical=categorical,
            validation=False,
        )
        heldout_rows.extend(_scorecard_rows(heldout_dir, metadata, source="locked_heldout"))

    heldout_aggregate = _aggregate_by_method(heldout_rows)
    _write_csv(study_root / "locked_heldout_scorecard.csv", heldout_rows)
    _write_csv(study_root / "locked_heldout_aggregate.csv", heldout_aggregate)
    _write_csv(
        study_root / "locked_heldout_paired_deltas.csv",
        _paired_deltas(heldout_rows, paired_comparisons),
    )

    ablation_rows: list[dict[str, Any]] = []
    if run_uncertainty_ablation:
        for seed, seed_records in sorted(by_seed.items()):
            chosen = next(
                row
                for row in selected
                if int(row["seed"]) == seed and row["method"] == "gradual_nll_mean_std"
            )
            experiment = Path(str(chosen["experiment"]))
            validation_controller = str(chosen["controller"])
            constant = _matched_constant_uncertainty(
                validation_dirs[seed],
                controller=validation_controller,
                experiment=experiment,
            )
            specs: list[dict[str, Any]] = []
            metadata: dict[str, dict[str, Any]] = {}
            for variant, extra in [
                ("predicted", {"uncertainty_override": "predicted"}),
                ("zero", {"uncertainty_override": "zeros"}),
                ("constant", {"uncertainty_override": "constant", "uncertainty_value": constant}),
                ("reflected", {"uncertainty_override": "reflected", "uncertainty_value": constant}),
            ]:
                label = f"gradual_nll_mean_std__{variant}__seed{seed}"
                specs.append(
                    _ppo_spec(label, experiment, str(chosen["weights_name"]), **extra)
                )
                metadata[label] = {
                    "seed": seed,
                    "method": "gradual_nll_mean_std",
                    "variant": variant,
                    "matched_constant_uncertainty_scaled": constant,
                    "selected_checkpoint": str(chosen["checkpoint"]),
                    "selected_checkpoint_kind": str(chosen["checkpoint_kind"]),
                    "experiment": str(experiment.resolve()),
                    "weights_name": str(chosen["weights_name"]),
                }
            target = Path(seed_records[0]["final_experiment"])
            ablation_dir = _run_sweep(
                target=target,
                specs=specs,
                output_subdir=f"final_thesis/{study_root.name}/seed{seed}/locked_heldout_uncertainty_ablation",
                categorical=categorical,
                validation=False,
            )
            ablation_rows.extend(
                _scorecard_rows(ablation_dir, metadata, source="locked_heldout_uncertainty_ablation")
            )
        _write_csv(study_root / "locked_heldout_uncertainty_ablation.csv", ablation_rows)

    return {
        "num_validation_checkpoint_rows": len(validation_rows),
        "num_selected_checkpoints": len(selected),
        "num_heldout_rows": len(heldout_rows),
        "num_uncertainty_ablation_rows": len(ablation_rows),
        "heldout_base_seed": HELDOUT_BASE_SEED,
        "validation_base_seed": VALIDATION_BASE_SEED,
    }


def _record(
    *,
    seed: int,
    method: str,
    sources: list[tuple[Path, list[str]]],
    final_experiment: Path,
    final_checkpoint: str,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "method": method,
        "sources": [
            {"experiment": str(experiment.resolve()), "phases": phases}
            for experiment, phases in sources
        ],
        "final_experiment": str(final_experiment.resolve()),
        "final_checkpoint": final_checkpoint,
        "final_weights_name": f"weights_{final_checkpoint}",
    }


def _run_categorical_study(
    *,
    run_root: Path,
    training_root: Path,
    base_cfg: dict[str, Any],
    system: dict[str, Any],
    seeds: list[int],
) -> dict[str, Any]:
    study_root = run_root / "categorical_nll_controlled"
    study_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    phases = [phase[0] for phase in CATEGORICAL_PHASES]
    for seed in seeds:
        for method, (kind, uncertainty) in CATEGORICAL_METHODS.items():
            exp_name = f"categorical_{method}_seed{seed}"
            exp_dir = training_root / exp_name
            _train_gradual_method(
                base_cfg=base_cfg,
                system=system,
                exp_root=training_root,
                exp_name=exp_name,
                exp_dir=exp_dir,
                seed=seed,
                categorical=True,
                kind=kind,
                condition_on_uncertainty=uncertainty,
                phases=CATEGORICAL_PHASES,
            )
            records.append(
                _record(
                    seed=seed,
                    method=method,
                    sources=[(exp_dir, phases)],
                    final_experiment=exp_dir,
                    final_checkpoint=phases[-1],
                )
            )

    evaluation = _evaluate_records(
        study_root=study_root,
        records=records,
        categorical=True,
        paired_comparisons=[
            ("gradual_nll_mean_std", "gradual_nll_mean_only"),
            ("gradual_nll_mean_std", "gradual_mle_mean_only"),
            ("gradual_nll_mean_only", "gradual_mle_mean_only"),
        ],
        run_uncertainty_ablation=True,
    )
    payload = {
        "purpose": "Controlled NLL mean-only versus NLL mean+std, with the natural MLE mean-only baseline.",
        "theta_distribution": {"categorical": True, "values": DISCRETE_THETA_VALUES},
        "methods": CATEGORICAL_METHODS,
        "phase_schedule": CATEGORICAL_PHASES,
        "total_steps_per_method": sum(phase[-1] for phase in CATEGORICAL_PHASES),
        "records": records,
        "evaluation": evaluation,
    }
    _write_json(study_root / "study_manifest.json", payload)
    return payload


def _run_continuous_study(
    *,
    run_root: Path,
    training_root: Path,
    base_cfg: dict[str, Any],
    system: dict[str, Any],
    seeds: list[int],
) -> dict[str, Any]:
    study_root = run_root / "continuous_staging"
    study_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    staged_runs: list[dict[str, Any]] = []
    privileged_phases = [phase[0] for phase in PRIVILEGED_PHASES]
    encoder_phases = [phase[0] for phase in SHARED_ENCODER_PHASES]
    two_extension = [phase[0] for phase in TWO_STAGE_EXTENSION]
    policy_phases = [phase[0] for phase in THREE_STAGE_POLICY_PHASES]
    gradual_phases = [phase[0] for phase in CONTINUOUS_GRADUAL_PHASES]

    for seed in seeds:
        staged = _train_continuous_staged_seed(
            base_cfg=base_cfg,
            system=system,
            exp_root=training_root,
            seed=seed,
        )
        staged_runs.append(staged)
        dirs = {key: Path(value) for key, value in staged["dirs"].items()}
        records.extend(
            [
                _record(
                    seed=seed,
                    method="privileged",
                    sources=[(dirs["privileged"], privileged_phases)],
                    final_experiment=dirs["privileged"],
                    final_checkpoint=privileged_phases[-1],
                ),
                _record(
                    seed=seed,
                    method="two_stage_rma",
                    sources=[
                        (dirs["shared_encoder"], encoder_phases),
                        (dirs["two_stage"], two_extension),
                    ],
                    final_experiment=dirs["two_stage"],
                    final_checkpoint=two_extension[-1],
                ),
                _record(
                    seed=seed,
                    method="three_stage_arma",
                    sources=[
                        (dirs["shared_encoder"], encoder_phases),
                        (dirs["three_stage"], policy_phases),
                    ],
                    final_experiment=dirs["three_stage"],
                    final_checkpoint=policy_phases[-1],
                ),
            ]
        )

        for method, kind, uncertainty in [
            ("gradual_mle_mean_only", "mle", False),
            ("gradual_nll_mean_std", "nll", True),
        ]:
            exp_name = f"continuous_{method}_seed{seed}"
            exp_dir = training_root / exp_name
            _train_gradual_method(
                base_cfg=base_cfg,
                system=system,
                exp_root=training_root,
                exp_name=exp_name,
                exp_dir=exp_dir,
                seed=seed,
                categorical=False,
                kind=kind,
                condition_on_uncertainty=uncertainty,
                phases=CONTINUOUS_GRADUAL_PHASES,
            )
            records.append(
                _record(
                    seed=seed,
                    method=method,
                    sources=[(exp_dir, gradual_phases)],
                    final_experiment=exp_dir,
                    final_checkpoint=gradual_phases[-1],
                )
            )

    evaluation = _evaluate_records(
        study_root=study_root,
        records=records,
        categorical=False,
        paired_comparisons=[
            ("three_stage_arma", "two_stage_rma"),
            ("gradual_mle_mean_only", "three_stage_arma"),
            ("gradual_nll_mean_std", "gradual_mle_mean_only"),
        ],
        run_uncertainty_ablation=True,
    )
    staged_budgets = {
        "privileged": sum(phase[1] for phase in PRIVILEGED_PHASES),
        "shared_encoder": sum(phase[1] for phase in SHARED_ENCODER_PHASES),
        "two_stage_total": sum(phase[1] for phase in PRIVILEGED_PHASES)
        + sum(phase[1] for phase in SHARED_ENCODER_PHASES)
        + sum(phase[1] for phase in TWO_STAGE_EXTENSION),
        "three_stage_total": sum(phase[1] for phase in PRIVILEGED_PHASES)
        + sum(phase[1] for phase in SHARED_ENCODER_PHASES)
        + sum(phase[1] for phase in THREE_STAGE_POLICY_PHASES),
        "gradual_total": sum(phase[-1] for phase in CONTINUOUS_GRADUAL_PHASES),
    }
    payload = {
        "purpose": "Continuous-theta two-stage RMA versus three-stage A-RMA versus gradual replacement.",
        "theta_distribution": {"categorical": False, "range": [THETA_LOW, THETA_HIGH]},
        "causal_design": (
            "Two-stage and three-stage share the exact privileged checkpoint and exact E02 encoder checkpoint; "
            "their only branch is E03 encoder-only versus S01/S02 policy-only."
        ),
        "privileged_phases": PRIVILEGED_PHASES,
        "shared_encoder_phases": SHARED_ENCODER_PHASES,
        "two_stage_extension": TWO_STAGE_EXTENSION,
        "three_stage_policy_phases": THREE_STAGE_POLICY_PHASES,
        "gradual_phases": CONTINUOUS_GRADUAL_PHASES,
        "budgets": staged_budgets,
        "staged_runs": staged_runs,
        "records": records,
        "evaluation": evaluation,
    }
    _write_json(study_root / "study_manifest.json", payload)
    return payload


def _verify_smoke_outputs(run_root: Path, studies: list[str]) -> None:
    for study in studies:
        name = "categorical_nll_controlled" if study == "categorical" else "continuous_staging"
        root = run_root / name
        required = [
            "study_manifest.json",
            "all_checkpoint_validation_scorecard.csv",
            "final_method_scorecard.csv",
            "best_checkpoint_validation_scorecard.csv",
            "locked_heldout_scorecard.csv",
            "locked_heldout_aggregate.csv",
            "locked_heldout_paired_deltas.csv",
            "locked_heldout_uncertainty_ablation.csv",
        ]
        missing = [str(root / path) for path in required if not (root / path).exists()]
        if missing:
            raise RuntimeError(f"Smoke verification failed; missing: {missing}")


def main() -> None:
    original_config = CONFIG_PATH.read_bytes()
    original_hash = hashlib.sha256(original_config).hexdigest()
    base_cfg = yaml.safe_load(original_config)
    confirmation_dir, system = _load_selected_system()
    seeds = _selected_seeds()
    studies = _selected_studies()
    stamp = datetime.now().strftime("%m-%d__%H-%M-%S")
    if _smoke_enabled():
        run_root = Path("/private/tmp") / f"r22_final_thesis_smoke_{stamp}"
    else:
        run_root = RUN_FAMILY_ROOT / f"final_thesis_{stamp}_{SELECTED_SYSTEM_LABEL}"
    training_root = run_root / "training"
    training_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "timestamp": stamp,
        "run_root": str(run_root.resolve()),
        "smoke": _smoke_enabled(),
        "studies": studies,
        "seeds": seeds,
        "selected_system_confirmation": str(confirmation_dir.resolve()),
        "selected_system": system,
        "fixed_setup": {
            "process_noise_std": 0.05,
            "max_episode_steps": 512,
            "window_length": WINDOW_LENGTH,
            "nominal_warmup_steps": WARMUP_STEPS,
            "id_update_interval": ID_UPDATE_INTERVAL,
            "num_envs": TRAIN_NUM_ENVS,
            "vec_env_type": TRAIN_VEC_ENV_TYPE,
            "n_steps": TRAIN_N_STEPS,
            "batch_size": TRAIN_BATCH_SIZE,
            "validation_base_seed": VALIDATION_BASE_SEED,
            "heldout_base_seed": HELDOUT_BASE_SEED,
        },
        "config_sha256_before": original_hash,
        "results": {},
        "errors": {},
    }
    _write_json(run_root / "final_thesis_run_manifest.json", manifest)
    if not _smoke_enabled():
        RUN_FAMILY_ROOT.mkdir(parents=True, exist_ok=True)
        POINTER.write_text(str(run_root.resolve()))
    errors: dict[str, str] = {}
    try:
        for study in studies:
            try:
                if study == "categorical":
                    result = _run_categorical_study(
                        run_root=run_root,
                        training_root=training_root,
                        base_cfg=base_cfg,
                        system=system,
                        seeds=seeds,
                    )
                else:
                    result = _run_continuous_study(
                        run_root=run_root,
                        training_root=training_root,
                        base_cfg=base_cfg,
                        system=system,
                        seeds=seeds,
                    )
                manifest["results"][study] = result
            except Exception:
                errors[study] = traceback.format_exc()
                manifest["errors"][study] = errors[study]
                print(f"FAILED study {study}:\n{errors[study]}", flush=True)
            finally:
                CONFIG_PATH.write_bytes(original_config)
        manifest["completed"] = not errors
        _write_json(run_root / "final_thesis_run_manifest.json", manifest)
        if _smoke_enabled() and not errors:
            _verify_smoke_outputs(run_root, studies)
        if not _smoke_enabled():
            RUN_FAMILY_ROOT.mkdir(parents=True, exist_ok=True)
            POINTER.write_text(str(run_root.resolve()))
    finally:
        CONFIG_PATH.write_bytes(original_config)
        restored_hash = _file_sha256(CONFIG_PATH)
        if restored_hash != original_hash:
            raise RuntimeError(
                f"Root config restoration failed: expected {original_hash}, got {restored_hash}"
            )
        if _smoke_enabled() and run_root.exists():
            shutil.rmtree(run_root)

    if errors:
        failed = ", ".join(sorted(errors))
        raise RuntimeError(f"Final thesis runner completed with failed studies: {failed}")
    print(f"Final thesis experiments complete: {run_root}", flush=True)


if __name__ == "__main__":
    main()
