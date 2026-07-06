from __future__ import annotations

import copy
import os
from datetime import datetime
from pathlib import Path

import yaml

from run_two_action_pipeline import (
    CONFIG_PATH,
    TWO_ACTION_LQR,
    _base_stage_cfg,
    _resolve_exp_root,
    _run_with_config,
    _set_common_model_params,
    _snapshot_phase_weights,
)


ARCH = [128, 128]

DEFAULT_LATENT_DIMS = (1, 2, 4, 16)
DEFAULT_WINDOW_LENGTHS = (10, 25, 50, 100)
DEFAULT_MAX_EPISODE_STEPS = 512

PHASES = [
    ("P01", 0.1, 3e-4, 4_000_000),
    ("P02", 0.05, 3e-4, 4_000_000),
    ("P03", 0.02, 2e-4, 4_000_000),
    ("P04", 0.01, 1e-4, 4_000_000),
    ("P05", 0.005, 1e-4, 4_000_000),
]

SMOKE_PHASES = [
    ("P01", 0.02, 2e-4, 512),
    ("P02", 0.01, 2e-4, 512),
]


def _parse_int_tuple(raw: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if not raw:
        return default
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Parsed an empty integer list.")
    return values


def _smoke_enabled() -> bool:
    return os.environ.get("TWO_ACTION_E2E_MATRIX_SMOKE", "0").lower() in {"1", "true", "yes"}


def _selected_phases() -> list[tuple[str, float, float, int]]:
    if _smoke_enabled():
        return SMOKE_PHASES
    step_scale = float(os.environ.get("TWO_ACTION_E2E_MATRIX_STEP_SCALE", "1.0"))
    if step_scale <= 0:
        raise ValueError("TWO_ACTION_E2E_MATRIX_STEP_SCALE must be positive.")
    return [
        (phase, ent, lr, max(4096, int(round(steps * step_scale))))
        for phase, ent, lr, steps in PHASES
    ]


def _episode_steps_for_window(window_length: int) -> int:
    base_steps = int(os.environ.get("TWO_ACTION_E2E_MATRIX_EPISODE_STEPS", str(DEFAULT_MAX_EPISODE_STEPS)))
    long_window_steps = os.environ.get("TWO_ACTION_E2E_MATRIX_LONG_WINDOW_EPISODE_STEPS")
    if long_window_steps is None:
        return base_steps
    long_window_min = int(os.environ.get("TWO_ACTION_E2E_MATRIX_LONG_WINDOW_MIN", "100"))
    if int(window_length) >= long_window_min:
        return int(long_window_steps)
    return base_steps


def _set_end_to_end_rma(
    cfg: dict,
    *,
    ent: float,
    lr: float,
    latent_dim: int,
    window_length: int,
) -> None:
    params = cfg["model"]["params"]
    _set_common_model_params(cfg)

    params["context_mode"] = "encoder_nll"
    params["latent_dim"] = int(latent_dim)
    params["window_length"] = int(window_length)
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["condition_on_uncertainty"] = False
    params["privileged_uncertainty_mode"] = "zeros"
    params["uncertainty_regularization_coef"] = 0.0
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["uncertainty_penalty_metric"] = "std"
    params["detach_context_for_rl"] = False
    params["id_update_interval"] = 1
    params["nominal_warmup_steps"] = 0
    params["encoder_net_arch"] = list(ARCH)
    params["actor_net_arch"] = list(ARCH)
    params["critic_net_arch"] = list(ARCH)
    params["learning_rate"] = float(lr)
    params["ent_coef"] = float(ent)

    for cb in cfg.get("callbacks", []):
        if cb.get("name") == "LivePlotCallback":
            cb["enabled"] = True
        if cb.get("name") == "SaveModelCallback":
            cb.setdefault("params", {})
            cb["params"]["best_metric"] = "episode_reward"
            cb["params"]["save_on_training_end"] = True


def _write_pipeline_file(exp_dir: Path, payload: dict) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "two_action_e2e_rma_matrix.yaml", "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _run_stage(label: str, cfg: dict, exp_dir: Path, payload: dict) -> None:
    print(
        f"START {label}: {cfg['training']['experiment_name']} "
        f"steps={cfg['total_timesteps']}",
        flush=True,
    )
    _write_pipeline_file(exp_dir, payload)
    _run_with_config(cfg)
    print(f"END   {label}: {cfg['training']['experiment_name']}", flush=True)


def main() -> None:
    original_text = CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_text)
    exp_root = _resolve_exp_root(base_cfg)
    stamp = datetime.now().strftime("%m-%d__%H-%M")

    latent_dims = _parse_int_tuple(os.environ.get("TWO_ACTION_E2E_MATRIX_LATENTS"), DEFAULT_LATENT_DIMS)
    window_lengths = _parse_int_tuple(os.environ.get("TWO_ACTION_E2E_MATRIX_WINDOWS"), DEFAULT_WINDOW_LENGTHS)
    phases = _selected_phases()
    step_scale = None if _smoke_enabled() else float(os.environ.get("TWO_ACTION_E2E_MATRIX_STEP_SCALE", "1.0"))

    root_name = f"s_{stamp}_two_action_e2e_rma_latent_window_matrix"
    root_dir = exp_root / root_name
    root_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": stamp,
        "root_experiment": root_name,
        "initialization": "scratch",
        "two_action_lqr": TWO_ACTION_LQR,
        "model": "UnifiedContextPPO encoder_nll, no regression loss, RL gradients through latent",
        "arch": ARCH,
        "latent_dims": list(latent_dims),
        "window_lengths": list(window_lengths),
        "step_scale": step_scale,
        "phases": [
            {"phase": phase, "ent_coef": ent, "learning_rate": lr, "timesteps": steps}
            for phase, ent, lr, steps in phases
        ],
        "runs": [],
    }

    try:
        for latent_dim in latent_dims:
            for window_length in window_lengths:
                exp_name = (
                    f"s_{stamp}_two_action_e2e_rma"
                    f"_z{latent_dim}_w{window_length}"
                )
                exp_dir = exp_root / exp_name
                exp_dir.mkdir(parents=True, exist_ok=True)

                run_payload = copy.deepcopy(payload)
                run_payload["experiment"] = exp_name
                run_payload["latent_dim"] = int(latent_dim)
                run_payload["window_length"] = int(window_length)
                run_payload["max_episode_steps"] = _episode_steps_for_window(window_length)
                payload["runs"].append(
                    {
                        "experiment": exp_name,
                        "latent_dim": int(latent_dim),
                        "window_length": int(window_length),
                        "max_episode_steps": _episode_steps_for_window(window_length),
                    }
                )
                _write_pipeline_file(root_dir, payload)

                for phase_idx, (phase, ent, lr, steps) in enumerate(phases):
                    cfg = _base_stage_cfg(base_cfg, exp_root, exp_name, steps)
                    cfg["lqr_env"]["max_episode_steps"] = _episode_steps_for_window(window_length)
                    if _smoke_enabled():
                        cfg["lqr_env"]["max_episode_steps"] = 64
                    _set_end_to_end_rma(
                        cfg,
                        ent=ent,
                        lr=lr,
                        latent_dim=latent_dim,
                        window_length=window_length,
                    )

                    cfg["training"]["load_weights"] = phase_idx > 0
                    cfg["training"]["load_weights_from"] = str(exp_dir.resolve()) if phase_idx > 0 else None
                    cfg["training"]["load_weights_name"] = "weights"
                    cfg["training"]["load_encoder_only"] = False

                    _run_stage(
                        f"e2e_rma/z{latent_dim}_w{window_length}/{phase}",
                        cfg,
                        exp_dir,
                        run_payload,
                    )
                    _snapshot_phase_weights(exp_dir, phase)

    finally:
        _write_pipeline_file(root_dir, payload)
        CONFIG_PATH.write_text(original_text)


if __name__ == "__main__":
    main()
