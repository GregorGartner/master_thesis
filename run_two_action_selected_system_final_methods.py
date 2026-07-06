from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

import torch
import yaml

from run_two_action_gradual_encoder_curriculum import _set_gradual_encoder_curriculum
from run_two_action_gradual_mle_encoder_curriculum import _set_gradual_mle_curriculum
from run_two_action_pipeline import (
    CONFIG_PATH,
    ROOT,
    _base_stage_cfg,
    _resolve_exp_root,
    _run_with_config,
    _snapshot_phase_weights,
)
from run_two_action_system_neural_screening import (
    SELECTION_ROOT,
    _set_common,
    _set_mle_encoder,
    _set_system,
    _set_vanilla_policy,
)


SWEEP_SCRIPT = ROOT / "lqr_theta_sweep_eval.py"
SEED = 1
SELECTED_SYSTEM_LABEL = "r22_1p5"

ENCODER_EXTENSION_STEPS = 4_000_000
STAGED_POLICY_PHASES = [
    ("S01", 0.01, 2e-4, 4_000_000),
    ("S02", 0.002, 1e-4, 4_000_000),
]
GRADUAL_PHASES = [
    ("P01", 0.25, 0.10, 3e-4, 3_000_000),
    ("P02", 0.50, 0.08, 3e-4, 3_000_000),
    ("P03", 0.75, 0.05, 2e-4, 3_000_000),
    ("P04", 1.00, 0.02, 1e-4, 4_000_000),
    ("P05", 1.00, 0.005, 1e-4, 4_000_000),
    ("P06", 1.00, 0.002, 1e-4, 4_000_000),
]


def _smoke_enabled() -> bool:
    return os.environ.get("TWO_ACTION_FINAL_METHODS_SMOKE", "0").lower() in {"1", "true", "yes"}


def _steps(steps: int) -> int:
    return 16_384 if _smoke_enabled() else steps


def _checkpoint(exp_dir: Path, weights_name: str = "weights") -> Path:
    path = exp_dir / weights_name
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def _require_checkpoint(exp_dir: Path, weights_name: str = "weights") -> None:
    path = _checkpoint(exp_dir, weights_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing required checkpoint: {path}")


def _load_confirmation() -> tuple[Path, dict, dict]:
    pointer = SELECTION_ROOT / "latest_confirmation.txt"
    if not pointer.exists():
        raise FileNotFoundError("No latest confirmation pointer found. Run system confirmation first.")

    confirmation_dir = Path(pointer.read_text().strip()).resolve()
    freeze_path = confirmation_dir / "freeze_decision.json"
    run_path = confirmation_dir / "confirmation_run.json"
    if not freeze_path.exists() or not run_path.exists():
        raise RuntimeError(f"Confirmation is incomplete: {confirmation_dir}")

    with open(freeze_path, "r") as f:
        freeze = json.load(f)
    with open(run_path, "r") as f:
        confirmation = json.load(f)

    if not freeze.get("freeze_system", False):
        raise RuntimeError("Confirmation did not reproduce. Refusing to start final-method training.")
    if freeze["system"]["label"] != SELECTED_SYSTEM_LABEL:
        raise RuntimeError(
            f"Expected frozen system {SELECTED_SYSTEM_LABEL}, got {freeze['system']['label']}."
        )
    if int(confirmation["seed"]) != SEED:
        raise RuntimeError(f"Expected confirmation seed {SEED}, got {confirmation['seed']}.")

    run = confirmation["run"]
    for key in ["privileged", "encoder", "vanilla", "gradual"]:
        _require_checkpoint(Path(run["dirs"][key]))
    _require_checkpoint(Path(run["dirs"]["gradual"]), "weights_P05")
    return confirmation_dir, freeze["system"], run


def _base_cfg(base_cfg: dict, exp_root: Path, exp_name: str, steps: int, system: dict) -> dict:
    cfg = _base_stage_cfg(base_cfg, exp_root, exp_name, _steps(steps))
    _set_system(cfg, system)
    return cfg


def _set_load(cfg: dict, source: Path | None, weights_name: str = "weights") -> None:
    cfg["training"]["load_weights"] = source is not None
    cfg["training"]["load_weights_from"] = str(source.resolve()) if source is not None else None
    cfg["training"]["load_weights_name"] = weights_name
    cfg["training"]["load_encoder_only"] = False


def _policy_group_digest(checkpoint: Path, *, encoder: bool) -> str:
    with zipfile.ZipFile(checkpoint, "r") as archive:
        state = torch.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu")
    digest = hashlib.sha256()
    for name in sorted(state):
        if name.startswith("context_encoder.") != encoder:
            continue
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _assert_same_policy_group(before: Path, after: Path, *, encoder: bool, description: str) -> None:
    if _policy_group_digest(before, encoder=encoder) != _policy_group_digest(after, encoder=encoder):
        raise RuntimeError(f"Checkpoint integrity check failed: {description}.")


def _run_stage(label: str, cfg: dict) -> None:
    print(f"START {label}: {cfg['training']['experiment_name']} steps={cfg['total_timesteps']}", flush=True)
    _run_with_config(cfg)
    print(f"END   {label}: {cfg['training']['experiment_name']}", flush=True)


def _train_staged(
    *,
    base_cfg: dict,
    exp_root: Path,
    system: dict,
    names: dict,
    dirs: dict,
    confirmation_run: dict,
) -> None:
    source_privileged_dir = Path(confirmation_run["dirs"]["privileged"])
    source_encoder_dir = Path(confirmation_run["dirs"]["encoder"])
    _assert_same_policy_group(
        _checkpoint(source_privileged_dir),
        _checkpoint(source_encoder_dir),
        encoder=False,
        description="confirmation encoder stage did not preserve the privileged policy",
    )

    cfg = _base_cfg(base_cfg, exp_root, names["staged_encoder"], ENCODER_EXTENSION_STEPS, system)
    _set_mle_encoder(cfg, SEED)
    _set_load(cfg, source_encoder_dir)
    _run_stage("staged/E01_encoder_extension", cfg)
    _snapshot_phase_weights(dirs["staged_encoder"], "E01")
    _assert_same_policy_group(
        _checkpoint(source_encoder_dir),
        _checkpoint(dirs["staged_encoder"], "weights_E01"),
        encoder=False,
        description="frozen-policy encoder extension changed the privileged policy",
    )

    for phase_idx, (phase, ent, lr, steps) in enumerate(STAGED_POLICY_PHASES):
        cfg = _base_cfg(base_cfg, exp_root, names["staged_policy"], steps, system)
        _set_vanilla_policy(cfg, SEED)
        params = cfg["model"]["params"]
        params["ent_coef"] = float(ent)
        params["learning_rate"] = float(lr)
        source = dirs["staged_encoder"] if phase_idx == 0 else dirs["staged_policy"]
        _set_load(cfg, source)
        _run_stage(f"staged/{phase}", cfg)
        _snapshot_phase_weights(dirs["staged_policy"], phase)


def _set_gradual(
    cfg: dict,
    *,
    method: str,
    encoder_probability: float,
    ent: float,
    lr: float,
    freeze_encoder: bool,
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
    _set_common(cfg, SEED)
    params = cfg["model"]["params"]
    params["privileged_context_probability"] = 1.0 - float(encoder_probability)
    params["regression_coef"] = 0.0 if freeze_encoder else 1.0
    params["uncertainty_reward_penalty_coef"] = 0.0


def _train_gradual(
    *,
    method: str,
    base_cfg: dict,
    exp_root: Path,
    system: dict,
    exp_name: str,
    exp_dir: Path,
) -> None:
    for phase_idx, (phase, encoder_prob, ent, lr, steps) in enumerate(GRADUAL_PHASES):
        freeze_encoder = phase == "P06"
        cfg = _base_cfg(base_cfg, exp_root, exp_name, steps, system)
        _set_gradual(
            cfg,
            method=method,
            encoder_probability=encoder_prob,
            ent=ent,
            lr=lr,
            freeze_encoder=freeze_encoder,
        )
        _set_load(cfg, exp_dir if phase_idx > 0 else None)
        _run_stage(f"gradual_{method}/{phase}", cfg)
        _snapshot_phase_weights(exp_dir, phase)

        if freeze_encoder:
            _assert_same_policy_group(
                _checkpoint(exp_dir, "weights_P05"),
                _checkpoint(exp_dir, "weights_P06"),
                encoder=True,
                description=f"gradual {method} P06 changed the frozen encoder",
            )


def _validate_gradual_configs(mle_dir: Path, nll_dir: Path) -> None:
    with open(mle_dir / "config.yaml", "r") as f:
        mle = yaml.safe_load(f)["model"]["params"]
    with open(nll_dir / "config.yaml", "r") as f:
        nll = yaml.safe_load(f)["model"]["params"]

    matched_keys = [
        "seed",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "learning_rate",
        "ent_coef",
        "regression_coef",
        "privileged_context_probability",
        "detach_context_for_rl",
        "encoder_net_arch",
        "actor_net_arch",
        "critic_net_arch",
        "uncertainty_reward_penalty_coef",
    ]
    mismatched = [key for key in matched_keys if mle.get(key) != nll.get(key)]
    if mismatched:
        raise RuntimeError(f"Matched gradual MLE/NLL configs differ unexpectedly: {mismatched}")
    if mle["context_mode"] != "encoder_mle" or mle["condition_on_uncertainty"]:
        raise RuntimeError("Gradual MLE final config is not mean-only encoder_mle.")
    if nll["context_mode"] != "encoder_nll" or not nll["condition_on_uncertainty"]:
        raise RuntimeError("Gradual NLL final config is not uncertainty-conditioned encoder_nll.")


def _validate_phase_checkpoints(dirs: dict) -> None:
    _require_checkpoint(dirs["staged_encoder"], "weights_E01")
    for phase, *_ in STAGED_POLICY_PHASES:
        _require_checkpoint(dirs["staged_policy"], f"weights_{phase}")
    for method in ["mle", "nll"]:
        for phase, *_ in GRADUAL_PHASES:
            _require_checkpoint(dirs[f"gradual_{method}"], f"weights_{phase}")


def _ppo_spec(label: str, experiment: Path, weights_name: str = "weights") -> dict:
    return {
        "label": label,
        "kind": "ppo",
        "experiment": str(experiment.resolve()),
        "weights_name": weights_name,
    }


def _lqr_specs(target: Path) -> list[dict]:
    return [
        {"label": "nominal_lqr", "kind": "nominal_lqr", "experiment": str(target.resolve())},
        {"label": "oracle_lqr", "kind": "lqr", "experiment": str(target.resolve())},
    ]


def _compact_specs(dirs: dict, confirmation_run: dict) -> list[dict]:
    specs = [
        _ppo_spec("confirmation_privileged", Path(confirmation_run["dirs"]["privileged"])),
        _ppo_spec("confirmation_short_vanilla", Path(confirmation_run["dirs"]["vanilla"])),
        _ppo_spec("staged_encoder_E01", dirs["staged_encoder"], "weights_E01"),
    ]
    specs.extend(
        _ppo_spec(f"staged_{phase}", dirs["staged_policy"], f"weights_{phase}")
        for phase, *_ in STAGED_POLICY_PHASES
    )
    for method in ["mle", "nll"]:
        specs.extend(
            _ppo_spec(f"extended_{method}_{phase}", dirs[f"gradual_{method}"], f"weights_{phase}")
            for phase, *_ in GRADUAL_PHASES
        )
    return specs + _lqr_specs(dirs["staged_policy"])


def _full_specs(dirs: dict, confirmation_run: dict) -> list[dict]:
    specs = [
        _ppo_spec("privileged", Path(confirmation_run["dirs"]["privileged"])),
        _ppo_spec("short_vanilla_rma", Path(confirmation_run["dirs"]["vanilla"])),
        _ppo_spec("fully_trained_staged_S02", dirs["staged_policy"], "weights_S02"),
        _ppo_spec("short_gradual_mle_P05", Path(confirmation_run["dirs"]["gradual"]), "weights_P05"),
        _ppo_spec("extended_mle_P05", dirs["gradual_mle"], "weights_P05"),
        _ppo_spec("extended_mle_P06", dirs["gradual_mle"], "weights_P06"),
        _ppo_spec("extended_nll_P05", dirs["gradual_nll"], "weights_P05"),
        _ppo_spec("extended_nll_P06", dirs["gradual_nll"], "weights_P06"),
    ]
    return specs + _lqr_specs(dirs["staged_policy"])


def _run_sweep(
    *,
    target: Path,
    specs: list[dict],
    output_subdir: str,
    theta_points: int,
    episodes_per_theta: int,
) -> Path:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")
    env["THETA_SWEEP_TARGET_EXPERIMENT"] = str(target.resolve())
    env["THETA_SWEEP_OUTPUT_SUBDIR"] = output_subdir
    env["THETA_SWEEP_N_THETA_POINTS"] = str(theta_points)
    env["THETA_SWEEP_EPISODES_PER_THETA"] = str(episodes_per_theta)
    env["THETA_SWEEP_DETERMINISTIC"] = "1"
    env["THETA_SWEEP_RETURN_MODE"] = "quadratic"
    env["THETA_SWEEP_SAVE_TRAJECTORY_TRACE"] = "0"
    env["THETA_SWEEP_SAVE_STEP_LEVEL_CSV"] = "0"
    env["THETA_SWEEP_COLLECT_STEP_LEVEL_PREDICTIONS"] = "1"
    env["THETA_SWEEP_CONTROLLER_SPECS_JSON"] = json.dumps(specs)
    print(f"START theta sweep: {output_subdir}", flush=True)
    subprocess.run(["python3", str(SWEEP_SCRIPT)], cwd=ROOT, env=env, check=True)
    sweep_dir = target / output_subdir
    if not (sweep_dir / "controller_scorecard.csv").exists():
        raise RuntimeError(f"Sweep did not produce its scorecard: {sweep_dir}")
    if (sweep_dir / "step_predictions.csv").exists():
        raise RuntimeError(f"Sweep unexpectedly saved step_predictions.csv: {sweep_dir}")
    return sweep_dir


def _write_manifest(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "final_methods.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(path / "final_methods.yaml", "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main() -> None:
    confirmation_dir, system, confirmation_run = _load_confirmation()
    original_text = CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_text)
    exp_root = _resolve_exp_root(base_cfg)
    stamp = datetime.now().strftime("%m-%d__%H-%M")
    smoke_suffix = "_smoke" if _smoke_enabled() else ""
    run_root = SELECTION_ROOT / f"final_methods_{stamp}_{system['label']}_seed{SEED}{smoke_suffix}"

    names = {
        "staged_encoder": f"s_{stamp}_{system['label']}_final_staged_mle_encoder{smoke_suffix}",
        "staged_policy": f"s_{stamp}_{system['label']}_final_staged_vanilla_rma{smoke_suffix}",
        "gradual_mle": f"s_{stamp}_{system['label']}_final_gradual_mle{smoke_suffix}",
        "gradual_nll": f"s_{stamp}_{system['label']}_final_gradual_nll{smoke_suffix}",
    }
    dirs = {key: exp_root / name for key, name in names.items()}
    payload = {
        "confirmation_dir": str(confirmation_dir),
        "system": system,
        "seed": SEED,
        "smoke": _smoke_enabled(),
        "experiments": names,
        "dirs": {key: str(value.resolve()) for key, value in dirs.items()},
        "staged": {
            "source_privileged": confirmation_run["dirs"]["privileged"],
            "source_encoder": confirmation_run["dirs"]["encoder"],
            "encoder_extension_steps": _steps(ENCODER_EXTENSION_STEPS),
            "policy_phases": [
                {"phase": phase, "ent_coef": ent, "learning_rate": lr, "timesteps": _steps(steps)}
                for phase, ent, lr, steps in STAGED_POLICY_PHASES
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
            for phase, encoder_prob, ent, lr, steps in GRADUAL_PHASES
        ],
    }
    _write_manifest(run_root, payload)

    try:
        _train_staged(
            base_cfg=base_cfg,
            exp_root=exp_root,
            system=system,
            names=names,
            dirs=dirs,
            confirmation_run=confirmation_run,
        )
        _train_gradual(
            method="mle",
            base_cfg=base_cfg,
            exp_root=exp_root,
            system=system,
            exp_name=names["gradual_mle"],
            exp_dir=dirs["gradual_mle"],
        )
        _train_gradual(
            method="nll",
            base_cfg=base_cfg,
            exp_root=exp_root,
            system=system,
            exp_name=names["gradual_nll"],
            exp_dir=dirs["gradual_nll"],
        )
        _validate_phase_checkpoints(dirs)
        _validate_gradual_configs(dirs["gradual_mle"], dirs["gradual_nll"])
    finally:
        CONFIG_PATH.write_text(original_text)

    theta_points = 3 if _smoke_enabled() else 21
    episodes_per_theta = 1 if _smoke_enabled() else 10
    compact_dir = _run_sweep(
        target=dirs["staged_policy"],
        specs=_compact_specs(dirs, confirmation_run),
        output_subdir="theta_sweep_compact_all_phases",
        theta_points=theta_points,
        episodes_per_theta=episodes_per_theta,
    )
    full_dir = _run_sweep(
        target=dirs["staged_policy"],
        specs=_full_specs(dirs, confirmation_run),
        output_subdir="theta_sweep_full_final_comparison",
        theta_points=theta_points if _smoke_enabled() else 41,
        episodes_per_theta=episodes_per_theta if _smoke_enabled() else 20,
    )
    payload["sweeps"] = {
        "compact": str(compact_dir.resolve()),
        "full_final_comparison": str(full_dir.resolve()),
    }
    payload["validation"] = {
        "selected_system_and_seed_match_confirmation": True,
        "staged_encoder_preserved_privileged_policy": True,
        "mle_P06_encoder_unchanged": True,
        "nll_P06_encoder_unchanged": True,
        "mle_and_nll_schedules_matched": True,
        "all_phase_checkpoints_produced": True,
        "scorecards_produced": True,
    }
    _write_manifest(run_root, payload)
    if not _smoke_enabled():
        (SELECTION_ROOT / "latest_final_methods.txt").write_text(str(run_root.resolve()) + "\n")
    print(f"Saved final-method run to: {run_root}", flush=True)


if __name__ == "__main__":
    main()
