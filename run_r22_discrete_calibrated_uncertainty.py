from __future__ import annotations

import csv
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

import run_two_action_selected_system_final_methods as base
import run_two_action_selected_system_final_methods_discrete_theta as discrete
from run_two_action_gradual_encoder_curriculum import _set_gradual_encoder_curriculum
from run_two_action_gradual_mle_encoder_curriculum import _set_gradual_mle_curriculum
from run_two_action_system_neural_screening import _set_common, _set_system


ROOT = Path(__file__).resolve().parent
SELECTION_ROOT = ROOT / "experiments" / "two_action_system_selection"
RUN_FAMILY_ROOT = ROOT / "experiments" / "r22_discrete_calibrated_uncertainty"
POINTER = RUN_FAMILY_ROOT / "latest_r22_discrete_calibrated_uncertainty.txt"
SWEEP_SCRIPT = ROOT / "lqr_theta_sweep_eval.py"
SELECTED_SYSTEM_LABEL = "r22_1p5"
DISCRETE_THETA_VALUES = [-0.25, 0.0, 0.25]

DEFAULT_SEEDS = [1, 2, 3]
METHODS = [
    "gradual_mle_long",
    "gradual_nll_long",
    "gradual_nll_calibrated",
    "gradual_nll_calibrated_ft",
]
NLL_METHODS = [method for method in METHODS if "nll" in method]

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
CALIBRATION_AFTER = {
    "P04": ("CAL04", 1_000_000),
    "P06": ("CAL06", 1_500_000),
    "P08": ("CAL08", 2_000_000),
}
FINE_TUNE_PHASES = [
    ("FT01", 1.00, 0.002, 5e-5, 1_500_000),
    ("FT02", 1.00, 0.0, 3e-5, 1_500_000),
]
CALIBRATION_NOISE_STD = [0.0, 0.35]
CALIBRATION_NOISE_DIST = ["gaussian", "uniform"]
TRAIN_NUM_ENVS = 8
TRAIN_VEC_ENV_TYPE = "dummy"
TRAIN_N_STEPS = 512
TRAIN_BATCH_SIZE = 1024


def _smoke_enabled() -> bool:
    return os.environ.get("R22_CALIBRATED_UNCERTAINTY_SMOKE", "0").lower() in {
        "1",
        "true",
        "yes",
    }


def _steps(steps: int) -> int:
    return 16_384 if _smoke_enabled() else int(steps)


def _selected_seeds() -> list[int]:
    raw = os.environ.get("R22_CALIBRATED_UNCERTAINTY_SEEDS", "")
    if raw.strip():
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    return [1] if _smoke_enabled() else list(DEFAULT_SEEDS)


def _selected_methods() -> list[str]:
    raw = os.environ.get("R22_CALIBRATED_UNCERTAINTY_METHODS", "")
    if raw.strip():
        methods = [part.strip() for part in raw.split(",") if part.strip()]
        unknown = sorted(set(methods) - set(METHODS))
        if unknown:
            raise ValueError(f"Unknown methods in R22_CALIBRATED_UNCERTAINTY_METHODS: {unknown}")
        return methods
    return list(METHODS)


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


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


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
        raise RuntimeError("Confirmation did not reproduce. Refusing to start calibrated run.")
    system = freeze["system"]
    if system["label"] != SELECTED_SYSTEM_LABEL:
        raise RuntimeError(f"Expected frozen system {SELECTED_SYSTEM_LABEL}, got {system['label']}.")
    return confirmation_dir, system


def _apply_standard_discrete_r22_settings(cfg: dict, system: dict) -> None:
    _set_system(cfg, system)
    discrete._set_discrete_theta_wrappers(cfg)
    training = cfg.setdefault("training", {})
    training["num_envs"] = TRAIN_NUM_ENVS
    training["vec_env_type"] = TRAIN_VEC_ENV_TYPE
    lqr = cfg.setdefault("lqr_env", {})
    lqr["process_noise_std"] = 0.05
    lqr["max_episode_steps"] = 512
    params = cfg["model"]["params"]
    params["n_steps"] = TRAIN_N_STEPS
    params["batch_size"] = TRAIN_BATCH_SIZE
    params["window_length"] = 50
    params["nominal_warmup_steps"] = 49
    params["id_update_interval"] = 10


def _base_cfg(base_cfg: dict, exp_root: Path, exp_name: str, steps: int, system: dict) -> dict:
    cfg = base._base_stage_cfg(base_cfg, exp_root, exp_name, _steps(steps))
    _apply_standard_discrete_r22_settings(cfg, system)
    return cfg


def _set_load(cfg: dict, source: Path | None, weights_name: str = "weights") -> None:
    cfg["training"]["load_weights"] = source is not None
    cfg["training"]["load_weights_from"] = str(source.resolve()) if source is not None else None
    cfg["training"]["load_weights_name"] = weights_name
    cfg["training"]["load_encoder_only"] = False


def _run_stage(label: str, cfg: dict) -> None:
    print(f"START {label}: {cfg['training']['experiment_name']} steps={cfg['total_timesteps']}", flush=True)
    base._run_with_config(cfg)
    print(f"END   {label}: {cfg['training']['experiment_name']}", flush=True)


def _set_gradual(
    cfg: dict,
    *,
    method: str,
    seed: int,
    encoder_probability: float,
    ent: float,
    lr: float,
) -> None:
    if method == "mle":
        _set_gradual_mle_curriculum(
            cfg,
            encoder_probability=encoder_probability,
            ent=ent,
            lr=lr,
        )
    elif method == "nll":
        _set_gradual_encoder_curriculum(
            cfg,
            encoder_probability=encoder_probability,
            ent=ent,
            lr=lr,
        )
    else:
        raise ValueError(f"Unknown gradual method: {method}")
    _set_common(cfg, seed)
    params = cfg["model"]["params"]
    params["privileged_context_probability"] = 1.0 - float(encoder_probability)
    params["regression_coef"] = 1.0
    params["policy_loss_coef"] = 1.0
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["naive_action_noise_std"] = 0.0
    params["naive_action_noise_dist"] = "gaussian"


def _set_calibration_refresh(
    cfg: dict,
    *,
    seed: int,
    encoder_probability: float,
) -> None:
    _set_gradual_encoder_curriculum(
        cfg,
        encoder_probability=encoder_probability,
        ent=0.0,
        lr=1e-5,
    )
    _set_common(cfg, seed)
    params = cfg["model"]["params"]
    params["context_mode"] = "encoder_nll"
    params["freeze_ppo"] = True
    params["regression_coef"] = 1.0
    params["policy_loss_coef"] = 0.0
    params["condition_on_uncertainty"] = True
    params["privileged_uncertainty_mode"] = "predicted"
    params["privileged_context_probability"] = 1.0 - float(encoder_probability)
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["naive_action_noise_std"] = list(CALIBRATION_NOISE_STD)
    params["naive_action_noise_dist"] = list(CALIBRATION_NOISE_DIST)


def _set_policy_finetune(
    cfg: dict,
    *,
    seed: int,
    encoder_probability: float,
    ent: float,
    lr: float,
) -> None:
    _set_gradual_encoder_curriculum(
        cfg,
        encoder_probability=encoder_probability,
        ent=ent,
        lr=lr,
    )
    _set_common(cfg, seed)
    params = cfg["model"]["params"]
    params["context_mode"] = "encoder_nll"
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["policy_loss_coef"] = 1.0
    params["condition_on_uncertainty"] = True
    params["privileged_uncertainty_mode"] = "predicted"
    params["privileged_context_probability"] = 1.0 - float(encoder_probability)
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["naive_action_noise_std"] = 0.0
    params["naive_action_noise_dist"] = "gaussian"


def _train_method(
    *,
    method: str,
    seed: int,
    base_cfg: dict,
    exp_root: Path,
    exp_name: str,
    exp_dir: Path,
    system: dict,
) -> list[dict[str, Any]]:
    is_nll = "nll" in method
    use_calibration = method in {"gradual_nll_calibrated", "gradual_nll_calibrated_ft"}
    use_finetune = method == "gradual_nll_calibrated_ft"
    method_kind = "nll" if is_nll else "mle"
    phase_rows: list[dict[str, Any]] = []

    for phase_idx, (phase, encoder_prob, ent, lr, steps) in enumerate(GRADUAL_PHASES):
        cfg = _base_cfg(base_cfg, exp_root, exp_name, steps, system)
        _set_gradual(
            cfg,
            method=method_kind,
            seed=seed,
            encoder_probability=encoder_prob,
            ent=ent,
            lr=lr,
        )
        _set_load(cfg, exp_dir if phase_idx > 0 else None)
        _run_stage(f"{method}/seed{seed}/{phase}", cfg)
        base._snapshot_phase_weights(exp_dir, phase)
        phase_rows.append(
            {
                "kind": "gradual",
                "phase": phase,
                "encoder_probability": encoder_prob,
                "ent_coef": ent,
                "learning_rate": lr,
                "timesteps": _steps(steps),
            }
        )

        if use_calibration and phase in CALIBRATION_AFTER:
            cal_phase, cal_steps = CALIBRATION_AFTER[phase]
            cfg = _base_cfg(base_cfg, exp_root, exp_name, cal_steps, system)
            _set_calibration_refresh(cfg, seed=seed, encoder_probability=encoder_prob)
            _set_load(cfg, exp_dir)
            _run_stage(f"{method}/seed{seed}/{cal_phase}", cfg)
            base._snapshot_phase_weights(exp_dir, cal_phase)
            phase_rows.append(
                {
                    "kind": "calibration_refresh",
                    "phase": cal_phase,
                    "after_phase": phase,
                    "encoder_probability": encoder_prob,
                    "timesteps": _steps(cal_steps),
                    "naive_action_noise_std": list(CALIBRATION_NOISE_STD),
                    "naive_action_noise_dist": list(CALIBRATION_NOISE_DIST),
                }
            )

    if use_finetune:
        for ft_phase, encoder_prob, ent, lr, steps in FINE_TUNE_PHASES:
            cfg = _base_cfg(base_cfg, exp_root, exp_name, steps, system)
            _set_policy_finetune(
                cfg,
                seed=seed,
                encoder_probability=encoder_prob,
                ent=ent,
                lr=lr,
            )
            _set_load(cfg, exp_dir)
            _run_stage(f"{method}/seed{seed}/{ft_phase}", cfg)
            base._snapshot_phase_weights(exp_dir, ft_phase)
            phase_rows.append(
                {
                    "kind": "policy_finetune",
                    "phase": ft_phase,
                    "encoder_probability": encoder_prob,
                    "ent_coef": ent,
                    "learning_rate": lr,
                    "timesteps": _steps(steps),
                }
            )

    return phase_rows


def _final_weights_name(method: str) -> str:
    if method == "gradual_nll_calibrated":
        return "weights_CAL08"
    if method == "gradual_nll_calibrated_ft":
        return "weights_FT02"
    return "weights_P08"


def _ppo_spec(label: str, experiment: Path, weights_name: str, **extra: Any) -> dict[str, Any]:
    spec = {
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
    episodes_per_theta: int,
) -> Path:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")
    env["THETA_SWEEP_TARGET_EXPERIMENT"] = str(target.resolve())
    env["THETA_SWEEP_OUTPUT_SUBDIR"] = output_subdir
    env["THETA_SWEEP_THETA_VALUES"] = ",".join(str(value) for value in DISCRETE_THETA_VALUES)
    env["THETA_SWEEP_N_THETA_POINTS"] = str(len(DISCRETE_THETA_VALUES))
    env["THETA_SWEEP_EPISODES_PER_THETA"] = str(episodes_per_theta)
    env["THETA_SWEEP_DETERMINISTIC"] = "1"
    env["THETA_SWEEP_RETURN_MODE"] = "quadratic"
    env["THETA_SWEEP_SAVE_TRAJECTORY_TRACE"] = "0"
    env["THETA_SWEEP_SAVE_STEP_LEVEL_CSV"] = "0"
    env["THETA_SWEEP_COLLECT_STEP_LEVEL_PREDICTIONS"] = "1"
    env["THETA_SWEEP_CONTROLLER_SPECS_JSON"] = json.dumps(specs)
    print(f"START calibrated discrete sweep: {output_subdir}", flush=True)
    subprocess.run(["python3", str(SWEEP_SCRIPT)], cwd=ROOT, env=env, check=True)
    sweep_dir = target / output_subdir
    if not (sweep_dir / "controller_scorecard.csv").exists():
        raise RuntimeError(f"Sweep did not produce controller_scorecard.csv: {sweep_dir}")
    return sweep_dir


def _method_label(method: str, seed: int, variant: str = "predicted") -> str:
    if method == "gradual_mle_long":
        return f"{method}_seed{seed}"
    return f"{method}_{variant}_seed{seed}"


def _prediction_specs_for_seed(seed: int, dirs: dict[str, dict[int, Path]], methods: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for method in methods:
        label = _method_label(method, seed)
        specs.append(_ppo_spec(label, dirs[method][seed], _final_weights_name(method)))
    return specs


def _constant_uncertainties_from_prediction_sweep(
    sweep_dir: Path,
    dirs: dict[str, dict[int, Path]],
    methods: list[str],
    seed: int,
) -> dict[str, float]:
    rows = _read_csv(sweep_dir / "theta_sweep_aggregate.csv")
    out: dict[str, float] = {}
    for method in methods:
        if method not in NLL_METHODS:
            continue
        label = _method_label(method, seed)
        values = [
            float(row["pred_theta_std_tail_mean_mean"])
            for row in rows
            if row["controller"] == label and row.get("pred_theta_std_tail_mean_mean") not in {"", None}
        ]
        if not values:
            out[method] = 0.0
            continue
        with (dirs[method][seed] / "config.yaml").open() as f:
            params = yaml.safe_load(f)["model"]["params"]
        out[method] = float(np.mean(values) * float(params.get("z_scale", 1.0)))
    return out


def _ablation_specs_for_seed(
    seed: int,
    dirs: dict[str, dict[int, Path]],
    methods: list[str],
    constants: dict[str, float],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if "gradual_mle_long" in methods:
        specs.append(_ppo_spec(_method_label("gradual_mle_long", seed), dirs["gradual_mle_long"][seed], _final_weights_name("gradual_mle_long")))
    for method in methods:
        if method not in NLL_METHODS:
            continue
        exp = dirs[method][seed]
        weights = _final_weights_name(method)
        value = constants.get(method, 0.0)
        specs.extend(
            [
                _ppo_spec(_method_label(method, seed, "predicted"), exp, weights, uncertainty_override="predicted"),
                _ppo_spec(_method_label(method, seed, "zero"), exp, weights, uncertainty_override="zeros"),
                _ppo_spec(
                    _method_label(method, seed, "constant"),
                    exp,
                    weights,
                    uncertainty_override="constant",
                    uncertainty_value=value,
                ),
                _ppo_spec(
                    _method_label(method, seed, "reflected"),
                    exp,
                    weights,
                    uncertainty_override="reflected",
                    uncertainty_value=value,
                ),
            ]
        )
    return specs


def _collect_scorecard_rows(run_root: Path, sweep_dirs: dict[int, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed, sweep_dir in sorted(sweep_dirs.items()):
        for row in _read_csv(sweep_dir / "controller_scorecard.csv"):
            controller = row["controller"]
            method = ""
            variant = "predicted"
            for candidate in sorted(METHODS, key=len, reverse=True):
                prefix = f"{candidate}_"
                if controller.startswith(prefix):
                    method = candidate
                    rest = controller[len(prefix) :]
                    if rest.startswith("seed"):
                        variant = "mean_only"
                    else:
                        variant = rest.split("_seed", 1)[0]
                    break
            if not method:
                continue
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "variant": variant,
                    "controller": controller,
                    "mean_return": float(row.get("mean_return", "nan")),
                    "tail_mean_return": float(row.get("tail_mean_return", "nan")),
                    "center_mean_return": float(row.get("center_mean_return", "nan")),
                    "theta_rmse": float(row.get("theta_rmse", "nan")),
                    "tail_theta_rmse": float(row.get("tail_theta_rmse", "nan")),
                    "center_theta_rmse": float(row.get("center_theta_rmse", "nan")),
                }
            )
    _write_csv(run_root / "final_method_scorecard.csv", rows)
    return rows


def _aggregate_scorecards(run_root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["variant"])].append(row)
    out: list[dict[str, Any]] = []
    for (method, variant), group in sorted(grouped.items()):
        returns = [float(row["mean_return"]) for row in group]
        rmses = [float(row["theta_rmse"]) for row in group]
        out.append(
            {
                "method": method,
                "variant": variant,
                "num_seeds": len(group),
                "mean_return_mean": _mean(returns),
                "mean_return_std": _std(returns),
                "theta_rmse_mean": _mean(rmses),
                "theta_rmse_std": _std(rmses),
                "tail_theta_rmse_mean": _mean([float(row["tail_theta_rmse"]) for row in group]),
                "center_theta_rmse_mean": _mean([float(row["center_theta_rmse"]) for row in group]),
            }
        )
    _write_csv(run_root / "seed_aggregate_scorecard.csv", out)
    return out


def _success_summary(run_root: Path, rows: list[dict[str, Any]], methods: list[str], seeds: list[int]) -> dict[str, Any]:
    by_seed_method_variant = {
        (int(row["seed"]), row["method"], row["variant"]): row
        for row in rows
    }
    nll_vs_mle: dict[str, dict[str, Any]] = {}
    for method in methods:
        if method not in NLL_METHODS:
            continue
        wins = 0
        deltas: list[float] = []
        ablation_wins = {"zero": 0, "constant": 0, "reflected": 0}
        for seed in seeds:
            mle = by_seed_method_variant.get((seed, "gradual_mle_long", "mean_only"))
            pred = by_seed_method_variant.get((seed, method, "predicted"))
            if mle and pred:
                delta = float(pred["mean_return"]) - float(mle["mean_return"])
                deltas.append(delta)
                wins += int(delta > 0.0)
            for variant in ablation_wins:
                ablated = by_seed_method_variant.get((seed, method, variant))
                if pred and ablated:
                    ablation_wins[variant] += int(float(pred["mean_return"]) > float(ablated["mean_return"]))
        nll_vs_mle[method] = {
            "return_delta_vs_mle_by_seed": deltas,
            "return_delta_vs_mle_mean": _mean(deltas),
            "nll_return_wins_vs_mle": wins,
            "nll_return_win_fraction_vs_mle": wins / max(len(deltas), 1),
            "predicted_uncertainty_wins_vs_ablations": ablation_wins,
        }
    payload = {
        "primary_success_rule": "one NLL variant beats gradual_mle_long in mean_return in at least 2/3 seeds",
        "uncertainty_success_rule": "predicted uncertainty beats zero/constant/reflected ablations in at least 2/3 seeds",
        "methods": nll_vs_mle,
    }
    with (run_root / "success_summary.json").open("w") as f:
        json.dump(payload, f, indent=2)
    return payload


def _plot_results(run_root: Path, rows: list[dict[str, Any]]) -> None:
    predicted = [
        row
        for row in rows
        if row["variant"] in {"mean_only", "predicted"}
    ]
    methods = [method for method in METHODS if any(row["method"] == method for row in predicted)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(methods))
    returns_by_method = [
        [float(row["mean_return"]) for row in predicted if row["method"] == method]
        for method in methods
    ]
    rmse_by_method = [
        [float(row["theta_rmse"]) for row in predicted if row["method"] == method]
        for method in methods
    ]
    axes[0].bar(x, [_mean(vals) for vals in returns_by_method], yerr=[_std(vals) for vals in returns_by_method], alpha=0.75)
    axes[1].bar(x, [_mean(vals) for vals in rmse_by_method], yerr=[_std(vals) for vals in rmse_by_method], alpha=0.75)
    for ax, values in [(axes[0], returns_by_method), (axes[1], rmse_by_method)]:
        for idx, vals in enumerate(values):
            ax.scatter(np.full(len(vals), idx), vals, color="black", s=20, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([method.replace("gradual_", "").replace("_", "\n") for method in methods], fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_title("Mean Return Across Seeds")
    axes[0].set_ylabel("mean return")
    axes[1].set_title("Theta RMSE Across Seeds")
    axes[1].set_ylabel("theta RMSE")
    fig.tight_layout()
    fig.savefig(run_root / "return_and_rmse_by_method.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for method in methods:
        vals = [row for row in predicted if row["method"] == method]
        ax.scatter(
            [float(row["theta_rmse"]) for row in vals],
            [float(row["mean_return"]) for row in vals],
            label=method.replace("gradual_", ""),
            s=60,
        )
    ax.set_title("Return vs Identification Tradeoff")
    ax.set_xlabel("theta RMSE")
    ax.set_ylabel("mean return")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_root / "return_vs_identification_tradeoff.png", dpi=200)
    plt.close(fig)

    nll_rows = [row for row in rows if row["method"] in NLL_METHODS]
    by_key = {(row["seed"], row["method"], row["variant"]): row for row in nll_rows}
    delta_rows = []
    for seed in sorted({row["seed"] for row in nll_rows}):
        for method in NLL_METHODS:
            pred = by_key.get((seed, method, "predicted"))
            if not pred:
                continue
            for variant in ["zero", "constant", "reflected"]:
                other = by_key.get((seed, method, variant))
                if other:
                    delta_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "ablation": variant,
                            "return_delta": float(pred["mean_return"]) - float(other["mean_return"]),
                        }
                    )
    _write_csv(run_root / "uncertainty_ablation_deltas.csv", delta_rows)
    if delta_rows:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        labels = []
        means = []
        stds = []
        for method in NLL_METHODS:
            for variant in ["zero", "constant", "reflected"]:
                vals = [
                    row["return_delta"]
                    for row in delta_rows
                    if row["method"] == method and row["ablation"] == variant
                ]
                if not vals:
                    continue
                labels.append(f"{method.replace('gradual_nll_', 'nll_')}\nvs {variant}")
                means.append(_mean(vals))
                stds.append(_std(vals))
        xpos = np.arange(len(labels))
        ax.bar(xpos, means, yerr=stds, alpha=0.8)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("predicted return - ablated return")
        ax.set_title("Uncertainty Ablation Return Deltas")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_root / "uncertainty_ablation_return_deltas.png", dpi=200)
        plt.close(fig)


def _write_manifest(run_root: Path, payload: dict[str, Any]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "calibrated_uncertainty_run.json").open("w") as f:
        json.dump(payload, f, indent=2)
    with (run_root / "calibrated_uncertainty_run.yaml").open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main() -> None:
    confirmation_dir, system = _load_selected_system()
    original_text = base.CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_text)
    exp_root = base._resolve_exp_root(base_cfg)
    stamp = datetime.now().strftime("%m-%d__%H-%M")
    smoke_suffix = "_smoke" if _smoke_enabled() else ""
    seeds = _selected_seeds()
    methods = _selected_methods()
    RUN_FAMILY_ROOT.mkdir(parents=True, exist_ok=True)
    run_root = RUN_FAMILY_ROOT / f"calibrated_uncertainty_{stamp}_{system['label']}{smoke_suffix}"

    dirs: dict[str, dict[int, Path]] = {method: {} for method in methods}
    names: dict[str, dict[int, str]] = {method: {} for method in methods}
    for method in methods:
        for seed in seeds:
            exp_name = f"s_{stamp}_{system['label']}_discrete_{method}_seed{seed}{smoke_suffix}"
            names[method][seed] = exp_name
            dirs[method][seed] = exp_root / exp_name

    payload: dict[str, Any] = {
        "confirmation_dir": str(confirmation_dir),
        "system": system,
        "smoke": _smoke_enabled(),
        "seeds": seeds,
        "methods": methods,
        "theta_distribution": {"type": "categorical", "values": list(DISCRETE_THETA_VALUES)},
        "main_setup": {
            "process_noise_std": 0.05,
            "max_episode_steps": 512,
            "window_length": 50,
            "nominal_warmup_steps": 49,
            "id_update_interval": 10,
            "num_envs": TRAIN_NUM_ENVS,
            "vec_env_type": TRAIN_VEC_ENV_TYPE,
            "n_steps": TRAIN_N_STEPS,
            "batch_size": TRAIN_BATCH_SIZE,
            "rollout_transitions_per_update": TRAIN_NUM_ENVS * TRAIN_N_STEPS,
        },
        "gradual_phases": [
            {
                "phase": phase,
                "encoder_probability": encoder_prob,
                "privileged_probability": 1.0 - encoder_prob,
                "ent_coef": ent,
                "learning_rate": lr,
                "timesteps": _steps(steps),
            }
            for phase, encoder_prob, ent, lr, steps in GRADUAL_PHASES
        ],
        "calibration_refresh": {
            phase: {
                "phase": cal_phase,
                "timesteps": _steps(cal_steps),
                "naive_action_noise_std": list(CALIBRATION_NOISE_STD),
                "naive_action_noise_dist": list(CALIBRATION_NOISE_DIST),
            }
            for phase, (cal_phase, cal_steps) in CALIBRATION_AFTER.items()
        },
        "fine_tune_phases": [
            {
                "phase": phase,
                "encoder_probability": encoder_prob,
                "ent_coef": ent,
                "learning_rate": lr,
                "timesteps": _steps(steps),
            }
            for phase, encoder_prob, ent, lr, steps in FINE_TUNE_PHASES
        ],
        "experiments": names,
        "dirs": {method: {seed: str(path.resolve()) for seed, path in by_seed.items()} for method, by_seed in dirs.items()},
    }
    _write_manifest(run_root, payload)

    try:
        for seed in seeds:
            for method in methods:
                phase_rows = _train_method(
                    method=method,
                    seed=seed,
                    base_cfg=base_cfg,
                    exp_root=exp_root,
                    exp_name=names[method][seed],
                    exp_dir=dirs[method][seed],
                    system=system,
                )
                payload.setdefault("phase_logs", {}).setdefault(method, {})[str(seed)] = phase_rows
                _write_manifest(run_root, payload)
    finally:
        base.CONFIG_PATH.write_text(original_text)

    episodes_per_theta = 1 if _smoke_enabled() else 20
    prediction_sweeps: dict[int, Path] = {}
    ablation_sweeps: dict[int, Path] = {}
    constants_by_seed: dict[int, dict[str, float]] = {}
    for seed in seeds:
        target = dirs[methods[0]][seed]
        prediction_sweeps[seed] = _run_sweep(
            target=target,
            specs=_prediction_specs_for_seed(seed, dirs, methods),
            output_subdir=f"r22_calibrated_uncertainty_seed{seed}/predicted_final",
            episodes_per_theta=episodes_per_theta,
        )
        constants = _constant_uncertainties_from_prediction_sweep(
            prediction_sweeps[seed],
            dirs,
            methods,
            seed,
        )
        constants_by_seed[seed] = constants
        ablation_sweeps[seed] = _run_sweep(
            target=target,
            specs=_ablation_specs_for_seed(seed, dirs, methods, constants),
            output_subdir=f"r22_calibrated_uncertainty_seed{seed}/uncertainty_ablations",
            episodes_per_theta=episodes_per_theta,
        )

    payload["sweeps"] = {
        "predicted": {seed: str(path.resolve()) for seed, path in prediction_sweeps.items()},
        "uncertainty_ablations": {seed: str(path.resolve()) for seed, path in ablation_sweeps.items()},
    }
    payload["matched_constant_uncertainty_scaled"] = constants_by_seed
    rows = _collect_scorecard_rows(run_root, ablation_sweeps)
    aggregate = _aggregate_scorecards(run_root, rows)
    success = _success_summary(run_root, rows, methods, seeds)
    _plot_results(run_root, rows)
    payload["aggregate_scorecard"] = aggregate
    payload["success_summary"] = success
    _write_manifest(run_root, payload)
    if not _smoke_enabled():
        POINTER.write_text(str(run_root.resolve()) + "\n")
    print(f"Saved R22 calibrated uncertainty run to: {run_root}", flush=True)


if __name__ == "__main__":
    main()
