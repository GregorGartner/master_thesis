from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
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
from run_two_action_pipeline import CONFIG_PATH, ROOT, TRAIN_CMD, _snapshot_phase_weights
from run_two_action_system_neural_screening import _set_system


SELECTION_ROOT = ROOT / "experiments" / "two_action_system_selection"
RUN_FAMILY_ROOT = ROOT / "experiments" / "r22_discrete_robust_ppo"
POINTER = RUN_FAMILY_ROOT / "latest_r22_discrete_robust_ppo.txt"
SWEEP_SCRIPT = ROOT / "lqr_theta_sweep_eval.py"

SELECTED_SYSTEM_LABEL = "r22_1p5"
DISCRETE_THETA_VALUES = [-0.25, 0.0, 0.25]
DEFAULT_SEEDS = [1, 2, 3]

TRAIN_NUM_ENVS = 8
TRAIN_VEC_ENV_TYPE = "dummy"
TRAIN_N_STEPS = 512
TRAIN_BATCH_SIZE = 1024

ROBUST_PHASES = [
    ("R01", 3_000_000, 3e-4, 0.02),
    ("R02", 4_000_000, 2e-4, 0.01),
    ("R03", 4_000_000, 1e-4, 0.005),
    ("R04", 4_000_000, 1e-4, 0.002),
    ("R05", 4_000_000, 5e-5, 0.001),
    ("R06", 4_000_000, 3e-5, 0.0005),
]

FORBIDDEN_PPO_PARAM_KEYS = {
    "action_log_std_init",
    "actor_net_arch",
    "condition_on_uncertainty",
    "context_mode",
    "critic_net_arch",
    "detach_context_for_rl",
    "encoder_net_arch",
    "encoder_type",
    "freeze_ppo",
    "id_update_interval",
    "initial_context",
    "latent_dim",
    "naive_action_noise_dist",
    "naive_action_noise_std",
    "nominal_warmup_steps",
    "policy_loss_coef",
    "privileged_context_probability",
    "privileged_uncertainty_mode",
    "privileged_uncertainty_value",
    "regression_coef",
    "regression_param_names",
    "transition_type",
    "uncertainty_penalty_metric",
    "uncertainty_regularization_coef",
    "uncertainty_reward_penalty_coef",
    "use_transition_features",
    "window_length",
    "z_scale",
}


def _smoke_enabled() -> bool:
    return os.environ.get("R22_ROBUST_PPO_SMOKE", "0").lower() in {"1", "true", "yes"}


def _steps(steps: int) -> int:
    return 16_384 if _smoke_enabled() else int(steps)


def _episodes_per_theta() -> int:
    return 1 if _smoke_enabled() else 20


def _selected_seeds() -> list[int]:
    raw = os.environ.get("R22_ROBUST_PPO_SEEDS", "")
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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_selected_system() -> tuple[Path, dict[str, Any]]:
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
        raise RuntimeError("Confirmation did not reproduce. Refusing to start robust PPO baseline.")
    system = freeze["system"]
    if system["label"] != SELECTED_SYSTEM_LABEL:
        raise RuntimeError(f"Expected frozen system {SELECTED_SYSTEM_LABEL}, got {system['label']}.")
    return confirmation_dir, system


def _assert_equal(actual: Any, expected: Any, key: str) -> None:
    if actual != expected:
        raise AssertionError(f"Config verification failed for {key}: expected {expected!r}, got {actual!r}")


def _theta_wrapper_params(cfg: dict[str, Any]) -> dict[str, Any]:
    for wrapper in cfg.get("wrappers", []):
        if wrapper.get("enabled", True) is False:
            continue
        if wrapper.get("name") in {"DomainRandomizationWrapper", "ChangingCartPoleDynamics"}:
            return wrapper.get("params", {}) or {}
    return {}


def _plain_ppo_params(*, seed: int, lr: float, ent: float) -> dict[str, Any]:
    return {
        "policy": "MlpPolicy",
        "learning_rate": float(lr),
        "n_steps": TRAIN_N_STEPS,
        "batch_size": TRAIN_BATCH_SIZE,
        "n_epochs": 8,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": float(ent),
        "vf_coef": 0.5,
        "max_grad_norm": 1.0,
        "normalize_advantage": True,
        "target_kl": None,
        "seed": int(seed),
        "verbose": 1,
        "device": "cpu",
        "policy_kwargs": {
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            "log_std_init": -1.2,
        },
    }


def _robust_phase_cfg(
    *,
    base_cfg: dict[str, Any],
    exp_root: Path,
    exp_name: str,
    steps: int,
    system: dict[str, Any],
    seed: int,
    lr: float,
    ent: float,
    source: Path | None,
    weights_name: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["environment"] = "lqr"
    cfg["total_timesteps"] = _steps(steps)
    _set_system(cfg, system)
    discrete._set_discrete_theta_wrappers(cfg)

    cfg["wrappers"] = [
        wrapper
        for wrapper in cfg.get("wrappers", [])
        if wrapper.get("enabled", True) is not False
        and wrapper.get("name") in {"DomainRandomizationWrapper", "ChangingCartPoleDynamics"}
    ]

    lqr = cfg.setdefault("lqr_env", {})
    lqr["process_noise_std"] = 0.05
    lqr["max_episode_steps"] = 512

    cfg["model"] = {
        "name": "PPO",
        "params": _plain_ppo_params(seed=seed, lr=lr, ent=ent),
    }
    cfg["training"] = {
        "experiment_root": str(exp_root.resolve()),
        "experiment_name": exp_name,
        "experiment_name_suffix": None,
        "load_weights": source is not None,
        "load_weights_from": str(source.resolve()) if source is not None else None,
        "load_weights_name": weights_name,
        "load_encoder_only": False,
        "num_envs": TRAIN_NUM_ENVS,
        "vec_env_type": TRAIN_VEC_ENV_TYPE,
    }

    # Keep useful training callbacks but make their behavior explicit.
    for callback in cfg.get("callbacks", []):
        if callback.get("name") == "LivePlotCallback":
            callback["enabled"] = True
        if callback.get("name") == "SaveModelCallback":
            callback["enabled"] = True
            callback.setdefault("params", {})
            callback["params"]["save_name"] = "weights"
            callback["params"]["save_best_name"] = "weights_best"
            callback["params"]["save_on_training_end"] = True

    return cfg


def _verify_written_config(expected: dict[str, Any], *, phase: str) -> None:
    written = yaml.safe_load(CONFIG_PATH.read_text())
    _assert_equal(written.get("environment"), "lqr", f"{phase}.environment")
    _assert_equal(written.get("total_timesteps"), expected.get("total_timesteps"), f"{phase}.total_timesteps")

    _assert_equal(written.get("model", {}).get("name"), "PPO", f"{phase}.model.name")
    params = written["model"]["params"]
    exp_params = expected["model"]["params"]
    forbidden_present = sorted(FORBIDDEN_PPO_PARAM_KEYS.intersection(params))
    if forbidden_present:
        raise AssertionError(f"{phase}: plain PPO params contain context/encoder keys: {forbidden_present}")

    for key in [
        "policy",
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "normalize_advantage",
        "target_kl",
        "seed",
        "verbose",
        "device",
        "policy_kwargs",
    ]:
        _assert_equal(params.get(key), exp_params.get(key), f"{phase}.model.params.{key}")

    training = written["training"]
    exp_training = expected["training"]
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
        _assert_equal(training.get(key), exp_training.get(key), f"{phase}.training.{key}")

    wrappers = [
        wrapper
        for wrapper in written.get("wrappers", [])
        if wrapper.get("enabled", True) is not False
        and wrapper.get("name") in {"DomainRandomizationWrapper", "ChangingCartPoleDynamics"}
    ]
    _assert_equal(len(wrappers), 1, f"{phase}.domain_randomization_wrapper_count")
    _assert_equal(wrappers[0].get("name"), "DomainRandomizationWrapper", f"{phase}.wrapper.name")
    wrapper = _theta_wrapper_params(written)
    _assert_equal(wrapper.get("randomize_theta"), True, f"{phase}.randomize_theta")
    _assert_equal(wrapper.get("randomize_mu"), False, f"{phase}.randomize_mu")
    _assert_equal(wrapper.get("randomize_a"), False, f"{phase}.randomize_a")
    _assert_equal(wrapper.get("randomize_b"), False, f"{phase}.randomize_b")
    _assert_equal(wrapper.get("randomize_process_noise_scale"), False, f"{phase}.randomize_process_noise_scale")
    _assert_equal(wrapper.get("theta_mult_range"), DISCRETE_THETA_VALUES, f"{phase}.theta_mult_range")
    _assert_equal(wrapper.get("categorical"), True, f"{phase}.categorical")

    lqr = written["lqr_env"]
    _assert_equal(lqr.get("process_noise_std"), 0.05, f"{phase}.process_noise_std")
    _assert_equal(lqr.get("max_episode_steps"), 512, f"{phase}.max_episode_steps")
    if len(lqr.get("A", [])) != 2 or len(lqr.get("B", [])) != 2:
        raise AssertionError(f"{phase}: expected 2D R22 LQR state/action matrices.")


def _run_stage(label: str, cfg: dict[str, Any]) -> None:
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


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _train_seed(
    *,
    seed: int,
    base_cfg: dict[str, Any],
    exp_root: Path,
    system: dict[str, Any],
    stamp: str,
    smoke_suffix: str,
) -> dict[str, Any]:
    exp_name = f"s_{stamp}_{system['label']}_discrete_robust_ppo_seed{seed}{smoke_suffix}"
    exp_dir = exp_root / exp_name
    phase_rows: list[dict[str, Any]] = []
    previous_source: Path | None = None
    previous_weights = "weights"

    for phase, steps, lr, ent in ROBUST_PHASES:
        cfg = _robust_phase_cfg(
            base_cfg=base_cfg,
            exp_root=exp_root,
            exp_name=exp_name,
            steps=steps,
            system=system,
            seed=seed,
            lr=lr,
            ent=ent,
            source=previous_source,
            weights_name=previous_weights,
        )
        _run_stage(f"seed{seed}/robust/{phase}", cfg)
        _snapshot_phase_weights(exp_dir, phase)
        previous_source = exp_dir
        previous_weights = f"weights_{phase}"
        phase_rows.append(
            {
                "phase": phase,
                "timesteps": _steps(steps),
                "learning_rate": lr,
                "ent_coef": ent,
            }
        )

    return {
        "seed": seed,
        "name": exp_name,
        "dir": str(exp_dir.resolve()),
        "phases": phase_rows,
        "final_weights": "weights_R06",
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
    print(f"START robust PPO theta sweep: {output_subdir}", flush=True)
    subprocess.run(["python3", str(SWEEP_SCRIPT)], cwd=ROOT, env=env, check=True)
    sweep_dir = target / output_subdir
    if not (sweep_dir / "controller_scorecard.csv").exists():
        raise RuntimeError(f"Sweep did not produce controller_scorecard.csv: {sweep_dir}")
    return sweep_dir


def _final_specs(seed_result: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(seed_result["seed"])
    exp_dir = Path(seed_result["dir"])
    return [
        _ppo_spec(
            f"robust_ppo_R06_seed{seed}",
            exp_dir,
            "weights_R06",
            method="robust_ppo",
            seed=seed,
            checkpoint="R06",
            checkpoint_kind="phase_end",
        )
    ]


def _all_checkpoint_specs(seed_result: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(seed_result["seed"])
    exp_dir = Path(seed_result["dir"])
    specs: list[dict[str, Any]] = []
    for phase, *_ in ROBUST_PHASES:
        for prefix, kind in [("weights", "phase_end"), ("weights_best", "best_within_phase")]:
            weights_name = f"{prefix}_{phase}"
            if _checkpoint(exp_dir, weights_name).exists():
                specs.append(
                    _ppo_spec(
                        f"robust_ppo_{kind}_{phase}_seed{seed}",
                        exp_dir,
                        weights_name,
                        method="robust_ppo",
                        seed=seed,
                        checkpoint=phase,
                        checkpoint_kind=kind,
                    )
                )
    return specs


def _as_float(value: Any) -> float:
    if value in {"", None}:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _info_proxy_by_controller(sweep_dir: Path) -> dict[str, dict[str, float]]:
    aggregate = sweep_dir / "theta_sweep_aggregate.csv"
    if not aggregate.exists():
        return {}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(aggregate):
        grouped[row["controller"]].append(row)

    out: dict[str, dict[str, float]] = {}
    for controller, rows in grouped.items():
        all_values: list[float] = []
        tail_values: list[float] = []
        center_values: list[float] = []
        for row in rows:
            value = _as_float(row.get("episode_info_proxy_mean"))
            if math.isnan(value):
                continue
            theta = _as_float(row.get("theta"))
            all_values.append(value)
            if abs(theta) >= 0.15:
                tail_values.append(value)
            if abs(theta) <= 0.05:
                center_values.append(value)

        def mean(values: list[float]) -> float:
            return float(sum(values) / len(values)) if values else float("nan")

        out[controller] = {
            "info_proxy_mean": mean(all_values),
            "tail_info_proxy_mean": mean(tail_values),
            "center_info_proxy_mean": mean(center_values),
        }
    return out


def _collect_scorecard_rows(sweep_dir: Path, *, seed: int, source: str) -> list[dict[str, Any]]:
    info_by_controller = _info_proxy_by_controller(sweep_dir)
    rows: list[dict[str, Any]] = []
    for row in _read_csv(sweep_dir / "controller_scorecard.csv"):
        controller = row["controller"]
        out = {
            "seed": seed,
            "source": source,
            "method": "robust_ppo",
            "controller": controller,
            "mean_return": _as_float(row.get("mean_return")),
            "tail_mean_return": _as_float(row.get("tail_mean_return")),
            "center_mean_return": _as_float(row.get("center_mean_return")),
            "theta_rmse": float("nan"),
            "tail_theta_rmse": float("nan"),
            "center_theta_rmse": float("nan"),
            "theta_rmse_note": "n/a_non_adaptive",
        }
        out.update(info_by_controller.get(controller, {}))
        rows.append(out)
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
        target = Path(result["dir"])
        final_dir = _run_sweep(
            target=target,
            specs=_final_specs(result),
            output_subdir=f"r22_discrete_robust_ppo_seed{seed}/final",
        )
        final_rows.extend(_collect_scorecard_rows(final_dir, seed=seed, source="final"))
        checkpoint_dir = _run_sweep(
            target=target,
            specs=_all_checkpoint_specs(result),
            output_subdir=f"r22_discrete_robust_ppo_seed{seed}/all_checkpoints",
        )
        checkpoint_rows.extend(_collect_scorecard_rows(checkpoint_dir, seed=seed, source="all_checkpoints"))
    _write_csv(run_root / "final_method_scorecard.csv", final_rows)
    _write_csv(run_root / "all_checkpoint_scorecard.csv", checkpoint_rows)
    _write_best_by_method(run_root, checkpoint_rows)


def _verify_smoke_outputs(run_root: Path) -> None:
    required = [
        run_root / "final_method_scorecard.csv",
        run_root / "all_checkpoint_scorecard.csv",
        run_root / "all_checkpoint_best_by_method.csv",
    ]
    for path in required:
        if not path.exists():
            raise RuntimeError(f"Smoke verification failed, missing {path}")
    for row in _read_csv(run_root / "final_method_scorecard.csv"):
        if row.get("theta_rmse_note") != "n/a_non_adaptive":
            raise RuntimeError("Smoke verification failed: robust baseline should mark theta RMSE as n/a.")


def main() -> None:
    confirmation_dir, system = _load_selected_system()
    original_text = CONFIG_PATH.read_text()
    original_hash = _file_sha256(CONFIG_PATH)
    base_cfg = yaml.safe_load(original_text)
    stamp = datetime.now().strftime("%m-%d__%H-%M")
    smoke_suffix = "_smoke" if _smoke_enabled() else ""
    seeds = _selected_seeds()

    if _smoke_enabled():
        run_root = Path("/private/tmp") / f"r22_discrete_robust_ppo_smoke_{stamp}"
        exp_root = run_root / "experiments"
    else:
        RUN_FAMILY_ROOT.mkdir(parents=True, exist_ok=True)
        run_root = RUN_FAMILY_ROOT / f"robust_ppo_{stamp}_{system['label']}"
        exp_root = ROOT / "experiments"

    payload: dict[str, Any] = {
        "confirmation_dir": str(confirmation_dir),
        "system": system,
        "smoke": _smoke_enabled(),
        "seeds": seeds,
        "theta_distribution": {"type": "categorical", "values": list(DISCRETE_THETA_VALUES)},
        "setup": {
            "non_adaptive": True,
            "model_type": "plain_sb3_ppo",
            "observation_dim": 2,
            "uses_previous_action_observation": False,
            "uses_context_or_encoder": False,
            "process_noise_std": 0.05,
            "max_episode_steps": 512,
            "num_envs": TRAIN_NUM_ENVS,
            "vec_env_type": TRAIN_VEC_ENV_TYPE,
            "n_steps": TRAIN_N_STEPS,
            "batch_size": TRAIN_BATCH_SIZE,
            "policy_net_arch": {"pi": [128, 128], "vf": [128, 128]},
        },
        "phase_budgets": [
            {"phase": phase, "timesteps": _steps(steps), "learning_rate": lr, "ent_coef": ent}
            for phase, steps, lr, ent in ROBUST_PHASES
        ],
        "seed_results": [],
    }

    run_root.mkdir(parents=True, exist_ok=True)
    manifest = run_root / "robust_ppo_run.json"
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
    _write_manifest(manifest, payload)

    if _smoke_enabled():
        _verify_smoke_outputs(run_root)
        print(f"Smoke verification passed; removing {run_root}", flush=True)
        shutil.rmtree(run_root)
    else:
        POINTER.write_text(str(run_root.resolve()))
        print(f"Saved robust PPO baseline to: {run_root}", flush=True)


if __name__ == "__main__":
    main()
