from __future__ import annotations

import os
from pathlib import Path

import yaml

from scalar_ab_lqr_utils import (
    DEFAULT_RANGE,
    SELECTION_ROOT,
    base_training_cfg,
    evaluate_controller_grid,
    run_training_stage,
    save_json,
    set_load,
    smoke_enabled,
    snapshot_phase_weights,
    timestamp,
)
from run_two_action_pipeline import CONFIG_PATH, _resolve_exp_root


SEED = 1
PRIVILEGED_STEPS = 2_000_000
ENCODER_STEPS = 1_500_000
STAGED_POLICY_PHASES = [
    ("S01", 0.01, 2e-4, 1_500_000),
    ("S02", 0.002, 1e-4, 1_500_000),
]
GRADUAL_PHASES = [
    ("P01", 0.25, 0.08, 3e-4, 1_000_000),
    ("P02", 0.50, 0.05, 3e-4, 1_000_000),
    ("P03", 0.75, 0.03, 2e-4, 1_000_000),
    ("P04", 1.00, 0.01, 1e-4, 1_500_000),
    ("P05", 1.00, 0.005, 1e-4, 1_500_000),
    ("P06", 1.00, 0.002, 1e-4, 1_000_000),
]


def _smoke() -> bool:
    return smoke_enabled("SCALAR_AB_NEURAL_RMA_SMOKE")


def _step_multiplier() -> float:
    if _smoke():
        return 1.0
    raw = os.environ.get("SCALAR_AB_NEURAL_RMA_STEP_MULT", os.environ.get("SCALAR_AB_STEP_MULT"))
    if raw is not None and raw != "":
        return float(raw)
    return 2.0 if os.environ.get("SCALAR_AB_LONG", "0").lower() in {"1", "true", "yes"} else 1.0


def _steps(steps: int) -> int:
    return 16_384 if _smoke() else int(round(float(steps) * _step_multiplier()))


def _checkpoint(exp_dir: Path, weights_name: str = "weights") -> Path:
    path = exp_dir / weights_name
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def _set_privileged(cfg: dict) -> None:
    params = cfg["model"]["params"]
    params["context_mode"] = "privileged"
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["condition_on_uncertainty"] = False
    params["privileged_uncertainty_mode"] = "zeros"
    params["privileged_context_probability"] = 0.0
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["learning_rate"] = 3e-4
    params["ent_coef"] = 0.01


def _set_staged_encoder(cfg: dict) -> None:
    params = cfg["model"]["params"]
    params["context_mode"] = "encoder_mle"
    params["freeze_ppo"] = True
    params["regression_coef"] = 1.0
    params["condition_on_uncertainty"] = False
    params["privileged_uncertainty_mode"] = "zeros"
    params["privileged_context_probability"] = 0.0
    params["naive_action_noise_std"] = [0.0, 0.3]
    params["naive_action_noise_dist"] = ["gaussian", "uniform"]
    params["learning_rate"] = 1e-5
    params["ent_coef"] = 0.0
    for cb in cfg.get("callbacks", []):
        if cb.get("name") == "SaveModelCallback":
            cb.setdefault("params", {})["best_metric"] = "regression_mse"


def _set_staged_policy(cfg: dict, *, ent: float, lr: float) -> None:
    params = cfg["model"]["params"]
    params["context_mode"] = "encoder_mle"
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["condition_on_uncertainty"] = False
    params["privileged_uncertainty_mode"] = "zeros"
    params["privileged_context_probability"] = 0.0
    params["naive_action_noise_std"] = 0.0
    params["learning_rate"] = float(lr)
    params["ent_coef"] = float(ent)


def _set_gradual(cfg: dict, *, method: str, encoder_probability: float, ent: float, lr: float, freeze_encoder: bool) -> None:
    params = cfg["model"]["params"]
    params["context_mode"] = "encoder_nll" if method == "nll" else "encoder_mle"
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0 if freeze_encoder else 1.0
    params["condition_on_uncertainty"] = method == "nll"
    params["privileged_uncertainty_mode"] = "predicted" if method == "nll" else "zeros"
    params["privileged_context_probability"] = 1.0 - float(encoder_probability)
    params["uncertainty_regularization_coef"] = 0.0
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["naive_action_noise_std"] = 0.0
    params["learning_rate"] = float(lr)
    params["ent_coef"] = float(ent)


def _train_staged(base_cfg: dict, exp_root: Path, names: dict[str, str], dirs: dict[str, Path]) -> None:
    cfg = base_training_cfg(base_cfg, exp_root, names["privileged"], _steps(PRIVILEGED_STEPS), DEFAULT_RANGE)
    _set_privileged(cfg)
    run_training_stage("scalar_ab_staged/privileged", cfg)

    cfg = base_training_cfg(base_cfg, exp_root, names["staged_encoder"], _steps(ENCODER_STEPS), DEFAULT_RANGE)
    _set_staged_encoder(cfg)
    set_load(cfg, dirs["privileged"])
    run_training_stage("scalar_ab_staged/encoder", cfg)

    for phase_idx, (phase, ent, lr, steps) in enumerate(STAGED_POLICY_PHASES):
        cfg = base_training_cfg(base_cfg, exp_root, names["staged_policy"], _steps(steps), DEFAULT_RANGE)
        _set_staged_policy(cfg, ent=ent, lr=lr)
        set_load(cfg, dirs["staged_encoder"] if phase_idx == 0 else dirs["staged_policy"])
        run_training_stage(f"scalar_ab_staged/policy/{phase}", cfg)
        snapshot_phase_weights(dirs["staged_policy"], phase)


def _train_gradual(base_cfg: dict, exp_root: Path, exp_name: str, exp_dir: Path, *, method: str) -> None:
    for phase_idx, (phase, enc_prob, ent, lr, steps) in enumerate(GRADUAL_PHASES):
        cfg = base_training_cfg(base_cfg, exp_root, exp_name, _steps(steps), DEFAULT_RANGE)
        _set_gradual(cfg, method=method, encoder_probability=enc_prob, ent=ent, lr=lr, freeze_encoder=(phase == "P06"))
        set_load(cfg, exp_dir if phase_idx > 0 else None)
        run_training_stage(f"scalar_ab_gradual_{method}/{phase}", cfg)
        snapshot_phase_weights(exp_dir, phase)


def main() -> None:
    original_text = CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_text)
    exp_root = _resolve_exp_root(base_cfg)
    stamp = timestamp()
    suffix = "_smoke" if _smoke() else ""
    names = {
        "privileged": f"s_{stamp}_scalar_ab_staged_privileged{suffix}",
        "staged_encoder": f"s_{stamp}_scalar_ab_staged_mle_encoder{suffix}",
        "staged_policy": f"s_{stamp}_scalar_ab_staged_vanilla_rma{suffix}",
        "gradual_mle": f"s_{stamp}_scalar_ab_gradual_mle{suffix}",
        "gradual_nll": f"s_{stamp}_scalar_ab_gradual_nll{suffix}",
    }
    dirs = {key: exp_root / value for key, value in names.items()}
    run_root = SELECTION_ROOT / f"neural_rma_{stamp}_stable_wide_b{suffix}"
    manifest = {
        "timestamp": stamp,
        "smoke": _smoke(),
        "seed": SEED,
        "param_range": DEFAULT_RANGE,
        "step_multiplier": _step_multiplier(),
        "dirs": {key: str(value.resolve()) for key, value in dirs.items()},
        "privileged_steps": _steps(PRIVILEGED_STEPS),
        "encoder_steps": _steps(ENCODER_STEPS),
        "staged_policy_phases": [
            {"phase": phase, "ent_coef": ent, "learning_rate": lr, "timesteps": _steps(steps)}
            for phase, ent, lr, steps in STAGED_POLICY_PHASES
        ],
        "gradual_phases": [
            {"phase": phase, "encoder_probability": prob, "ent_coef": ent, "learning_rate": lr, "timesteps": _steps(steps)}
            for phase, prob, ent, lr, steps in GRADUAL_PHASES
        ],
    }
    save_json(run_root / "scalar_ab_neural_rma.json", manifest)

    try:
        _train_staged(base_cfg, exp_root, names, dirs)
        _train_gradual(base_cfg, exp_root, names["gradual_mle"], dirs["gradual_mle"], method="mle")
        _train_gradual(base_cfg, exp_root, names["gradual_nll"], dirs["gradual_nll"], method="nll")
    finally:
        CONFIG_PATH.write_text(original_text)

    for key, weight in [
        ("staged_policy", "weights_S02"),
        ("gradual_mle", "weights_P06"),
        ("gradual_nll", "weights_P06"),
    ]:
        path = _checkpoint(dirs[key], weight)
        if not path.exists():
            raise FileNotFoundError(f"Missing expected checkpoint: {path}")

    specs = [
        {"label": "privileged", "kind": "ppo", "experiment": str(dirs["privileged"]), "weights_name": "weights"},
        {"label": "staged_vanilla_S02", "kind": "ppo", "experiment": str(dirs["staged_policy"]), "weights_name": "weights_S02"},
        {"label": "gradual_mle_P06", "kind": "ppo", "experiment": str(dirs["gradual_mle"]), "weights_name": "weights_P06"},
        {"label": "gradual_nll_P06", "kind": "ppo", "experiment": str(dirs["gradual_nll"]), "weights_name": "weights_P06"},
        {"label": "oracle_lqr", "kind": "oracle_lqr"},
    ]
    compact_n = 3 if _smoke() else 31
    compact_seeds = 1 if _smoke() else 3
    final_n = 3 if _smoke() else 51
    final_seeds = 1 if _smoke() else 5
    compact = evaluate_controller_grid(run_root / "scalar_ab_grid_compact", specs, grid_n=compact_n, seeds_per_pair=compact_seeds)
    final = evaluate_controller_grid(run_root / "scalar_ab_grid_final", specs, grid_n=final_n, seeds_per_pair=final_seeds)
    manifest["sweeps"] = {"compact": str(compact.resolve()), "final": str(final.resolve())}
    manifest["validation"] = {"checkpoints_produced": True, "scorecards_produced": True}
    save_json(run_root / "scalar_ab_neural_rma.json", manifest)
    if not _smoke():
        (SELECTION_ROOT / "latest_scalar_ab_neural_rma.txt").write_text(str(run_root.resolve()) + "\n")
    print(f"Saved scalar-ab neural RMA run to: {run_root}", flush=True)


if __name__ == "__main__":
    main()
