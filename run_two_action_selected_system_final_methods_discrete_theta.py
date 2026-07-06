from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import yaml

import run_two_action_selected_system_final_methods as base
from run_two_action_system_neural_screening import _set_system


SEED = base.SEED
SELECTED_SYSTEM_LABEL = base.SELECTED_SYSTEM_LABEL
DISCRETE_THETA_VALUES = [-0.25, 0.0, 0.25]
SELECTION_ROOT = base.SELECTION_ROOT

SHORT_ENCODER_EXTENSION_STEPS = 2_000_000
SHORT_STAGED_POLICY_PHASES = [
    ("S01", 0.01, 2e-4, 2_000_000),
    ("S02", 0.002, 1e-4, 2_000_000),
]
SHORT_GRADUAL_PHASES = [
    ("P01", 0.25, 0.10, 3e-4, 1_500_000),
    ("P02", 0.50, 0.08, 3e-4, 1_500_000),
    ("P03", 0.75, 0.05, 2e-4, 1_500_000),
    ("P04", 1.00, 0.02, 1e-4, 2_000_000),
    ("P05", 1.00, 0.005, 1e-4, 2_000_000),
    ("P06", 1.00, 0.002, 1e-4, 2_000_000),
]


def _smoke_enabled() -> bool:
    return os.environ.get("TWO_ACTION_DISCRETE_FINAL_SMOKE", "0").lower() in {"1", "true", "yes"}


def _full_budget_enabled() -> bool:
    return os.environ.get("TWO_ACTION_DISCRETE_FINAL_FULL_BUDGET", "0").lower() in {"1", "true", "yes"}


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    return None if value in {None, ""} else float(value)


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return None if value in {None, ""} else int(value)


def r22_variant_overrides() -> dict:
    return {
        "process_noise_std": _env_float("TWO_ACTION_R22_PROCESS_NOISE_STD"),
        "max_episode_steps": _env_int("TWO_ACTION_R22_MAX_EPISODE_STEPS"),
        "window_length": _env_int("TWO_ACTION_R22_WINDOW_LENGTH"),
        "id_update_interval": _env_int("TWO_ACTION_R22_ID_UPDATE_INTERVAL"),
    }


def apply_r22_variant_overrides(cfg: dict) -> None:
    overrides = r22_variant_overrides()
    lqr = cfg.setdefault("lqr_env", {})
    if overrides["process_noise_std"] is not None:
        lqr["process_noise_std"] = overrides["process_noise_std"]
    if overrides["max_episode_steps"] is not None:
        lqr["max_episode_steps"] = overrides["max_episode_steps"]

    params = cfg.get("model", {}).get("params", {})
    if overrides["window_length"] is not None:
        params["window_length"] = overrides["window_length"]
        params["nominal_warmup_steps"] = max(0, overrides["window_length"] - 1)
    if overrides["id_update_interval"] is not None:
        params["id_update_interval"] = overrides["id_update_interval"]


def _steps(steps: int) -> int:
    return 16_384 if _smoke_enabled() else int(steps)


def _set_discrete_theta_wrappers(cfg: dict) -> None:
    for wrapper in cfg.get("wrappers", []):
        if wrapper.get("name") in {"DomainRandomizationWrapper", "ChangingCartPoleDynamics"}:
            params = wrapper.setdefault("params", {})
            params["randomize_theta"] = True
            params["randomize_a"] = False
            params["randomize_b"] = False
            params["theta_mult_range"] = list(DISCRETE_THETA_VALUES)
            params["categorical"] = True


def _base_cfg(discrete_base_cfg: dict, exp_root: Path, exp_name: str, steps: int, system: dict) -> dict:
    cfg = base._base_stage_cfg(discrete_base_cfg, exp_root, exp_name, _steps(steps))
    _set_system(cfg, system)
    _set_discrete_theta_wrappers(cfg)
    apply_r22_variant_overrides(cfg)
    return cfg


def _run_sweep(
    *,
    target: Path,
    specs: list[dict],
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
    print(f"START discrete theta sweep: {output_subdir}", flush=True)
    base.subprocess.run(["python3", str(base.SWEEP_SCRIPT)], cwd=base.ROOT, env=env, check=True)
    sweep_dir = target / output_subdir
    if not (sweep_dir / "controller_scorecard.csv").exists():
        raise RuntimeError(f"Sweep did not produce its scorecard: {sweep_dir}")
    if (sweep_dir / "step_predictions.csv").exists():
        raise RuntimeError(f"Sweep unexpectedly saved step_predictions.csv: {sweep_dir}")
    return sweep_dir


def _install_discrete_overrides() -> None:
    base._base_cfg = _base_cfg
    base._smoke_enabled = _smoke_enabled
    base._steps = _steps
    original_set_mle_encoder = base._set_mle_encoder
    original_set_vanilla_policy = base._set_vanilla_policy
    original_set_gradual = base._set_gradual

    def _set_mle_encoder_with_overrides(cfg: dict, seed: int) -> None:
        original_set_mle_encoder(cfg, seed)
        apply_r22_variant_overrides(cfg)

    def _set_vanilla_policy_with_overrides(cfg: dict, seed: int) -> None:
        original_set_vanilla_policy(cfg, seed)
        apply_r22_variant_overrides(cfg)

    def _set_gradual_with_overrides(cfg: dict, **kwargs) -> None:
        original_set_gradual(cfg, **kwargs)
        apply_r22_variant_overrides(cfg)

    base._set_mle_encoder = _set_mle_encoder_with_overrides
    base._set_vanilla_policy = _set_vanilla_policy_with_overrides
    base._set_gradual = _set_gradual_with_overrides
    if _full_budget_enabled():
        base.ENCODER_EXTENSION_STEPS = base.ENCODER_EXTENSION_STEPS
        base.STAGED_POLICY_PHASES = list(base.STAGED_POLICY_PHASES)
        base.GRADUAL_PHASES = list(base.GRADUAL_PHASES)
    else:
        base.ENCODER_EXTENSION_STEPS = SHORT_ENCODER_EXTENSION_STEPS
        base.STAGED_POLICY_PHASES = list(SHORT_STAGED_POLICY_PHASES)
        base.GRADUAL_PHASES = list(SHORT_GRADUAL_PHASES)


def main() -> None:
    _install_discrete_overrides()
    confirmation_dir, system, confirmation_run = base._load_confirmation()
    original_text = base.CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_text)
    exp_root = base._resolve_exp_root(base_cfg)
    stamp = datetime.now().strftime("%m-%d__%H-%M")
    smoke_suffix = "_smoke" if _smoke_enabled() else ""
    budget_tag = "full" if _full_budget_enabled() else "short"
    run_root = (
        SELECTION_ROOT
        / f"final_methods_discrete_theta_{stamp}_{system['label']}_seed{SEED}_{budget_tag}{smoke_suffix}"
    )

    names = {
        "staged_encoder": f"s_{stamp}_{system['label']}_discrete_theta_staged_mle_encoder_{budget_tag}{smoke_suffix}",
        "staged_policy": f"s_{stamp}_{system['label']}_discrete_theta_staged_vanilla_rma_{budget_tag}{smoke_suffix}",
        "gradual_mle": f"s_{stamp}_{system['label']}_discrete_theta_gradual_mle_{budget_tag}{smoke_suffix}",
        "gradual_nll": f"s_{stamp}_{system['label']}_discrete_theta_gradual_nll_{budget_tag}{smoke_suffix}",
    }
    dirs = {key: exp_root / name for key, name in names.items()}
    payload = {
        "confirmation_dir": str(confirmation_dir),
        "system": system,
        "seed": SEED,
        "smoke": _smoke_enabled(),
        "budget": budget_tag,
        "theta_distribution": {
            "type": "categorical",
            "values": list(DISCRETE_THETA_VALUES),
        },
        "r22_variant_overrides": r22_variant_overrides(),
        "experiments": names,
        "dirs": {key: str(value.resolve()) for key, value in dirs.items()},
        "staged": {
            "source_privileged": confirmation_run["dirs"]["privileged"],
            "source_encoder": confirmation_run["dirs"]["encoder"],
            "encoder_extension_steps": _steps(base.ENCODER_EXTENSION_STEPS),
            "policy_phases": [
                {"phase": phase, "ent_coef": ent, "learning_rate": lr, "timesteps": _steps(steps)}
                for phase, ent, lr, steps in base.STAGED_POLICY_PHASES
            ],
        },
        "gradual_phases": [
            {
                "phase": phase,
                "encoder_probability": encoder_prob,
                "privileged_probability": 1.0 - encoder_prob,
                "encoder_frozen": phase == "P06",
                "ent_coef": ent,
                "learning_rate": lr,
                "timesteps": _steps(steps),
            }
            for phase, encoder_prob, ent, lr, steps in base.GRADUAL_PHASES
        ],
    }
    base._write_manifest(run_root, payload)

    try:
        base._train_staged(
            base_cfg=base_cfg,
            exp_root=exp_root,
            system=system,
            names=names,
            dirs=dirs,
            confirmation_run=confirmation_run,
        )
        base._train_gradual(
            method="mle",
            base_cfg=base_cfg,
            exp_root=exp_root,
            system=system,
            exp_name=names["gradual_mle"],
            exp_dir=dirs["gradual_mle"],
        )
        base._train_gradual(
            method="nll",
            base_cfg=base_cfg,
            exp_root=exp_root,
            system=system,
            exp_name=names["gradual_nll"],
            exp_dir=dirs["gradual_nll"],
        )
        base._validate_phase_checkpoints(dirs)
        base._validate_gradual_configs(dirs["gradual_mle"], dirs["gradual_nll"])
    finally:
        base.CONFIG_PATH.write_text(original_text)

    episodes_per_theta = 1 if _smoke_enabled() else 20
    compact_dir = _run_sweep(
        target=dirs["staged_policy"],
        specs=base._compact_specs(dirs, confirmation_run),
        output_subdir="theta_sweep_discrete_compact_all_phases",
        episodes_per_theta=episodes_per_theta,
    )
    full_dir = _run_sweep(
        target=dirs["staged_policy"],
        specs=base._full_specs(dirs, confirmation_run),
        output_subdir="theta_sweep_discrete_full_final_comparison",
        episodes_per_theta=episodes_per_theta,
    )
    payload["sweeps"] = {
        "compact": str(compact_dir.resolve()),
        "full_final_comparison": str(full_dir.resolve()),
    }
    payload["validation"] = {
        "selected_system_and_seed_match_confirmation": True,
        "categorical_theta_values": list(DISCRETE_THETA_VALUES),
        "staged_encoder_preserved_privileged_policy": True,
        "mle_P06_encoder_unchanged": True,
        "nll_P06_encoder_unchanged": True,
        "mle_and_nll_schedules_matched": True,
        "all_phase_checkpoints_produced": True,
        "scorecards_produced": True,
    }
    base._write_manifest(run_root, payload)
    if not _smoke_enabled():
        (SELECTION_ROOT / "latest_final_methods_discrete_theta.txt").write_text(str(run_root.resolve()) + "\n")
    print(f"Saved discrete-theta final-method run to: {run_root}", flush=True)


if __name__ == "__main__":
    main()
