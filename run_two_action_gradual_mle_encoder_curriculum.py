from __future__ import annotations

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


RUN_SUFFIX = "two_action_gradual_mle_encoder_curriculum"
ARCH = [128, 128]

PHASES = [
    ("P01", 0.25, 0.10, 3e-4, 4_000_000),
    ("P02", 0.50, 0.08, 3e-4, 4_000_000),
    ("P03", 0.75, 0.05, 2e-4, 4_000_000),
    ("P04", 1.00, 0.02, 1e-4, 4_000_000),
    ("P05", 1.00, 0.005, 1e-4, 4_000_000),
]


def _smoke_enabled() -> bool:
    return os.environ.get("TWO_ACTION_GRADUAL_MLE_SMOKE", "0").lower() in {"1", "true", "yes"}


def _selected_phases() -> list[tuple[str, float, float, float, int]]:
    if _smoke_enabled():
        return [(phase, encoder_prob, ent, lr, 16_384) for phase, encoder_prob, ent, lr, _ in PHASES]
    return PHASES


def _set_gradual_mle_curriculum(
    cfg: dict,
    *,
    encoder_probability: float,
    ent: float,
    lr: float,
) -> None:
    params = cfg["model"]["params"]
    _set_common_model_params(cfg)

    params["context_mode"] = "encoder_mle"
    params["freeze_ppo"] = False
    params["regression_coef"] = 1.0
    params["condition_on_uncertainty"] = False
    params["privileged_uncertainty_mode"] = "zeros"
    params["privileged_context_probability"] = 1.0 - float(encoder_probability)
    params["uncertainty_regularization_coef"] = 0.0
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["uncertainty_penalty_metric"] = "std"
    params["detach_context_for_rl"] = True

    params["n_steps"] = 8_192
    params["batch_size"] = 1_024
    params["n_epochs"] = 8
    params["gamma"] = 0.995
    params["learning_rate"] = float(lr)
    params["ent_coef"] = float(ent)

    params["encoder_net_arch"] = list(ARCH)
    params["actor_net_arch"] = list(ARCH)
    params["critic_net_arch"] = list(ARCH)

    for cb in cfg.get("callbacks", []):
        if cb.get("name") == "LivePlotCallback":
            cb["enabled"] = True
        if cb.get("name") == "SaveModelCallback":
            cb.setdefault("params", {})
            cb["params"]["best_metric"] = "episode_reward"
            cb["params"]["save_on_training_end"] = True


def _write_pipeline_file(exp_dir: Path, payload: dict) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "two_action_gradual_mle_encoder_curriculum.yaml", "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main() -> None:
    original_text = CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_text)
    exp_root = _resolve_exp_root(base_cfg)
    stamp = datetime.now().strftime("%m-%d__%H-%M")
    phases = _selected_phases()

    suffix = f"{RUN_SUFFIX}_smoke" if _smoke_enabled() else RUN_SUFFIX
    exp_name = f"s_{stamp}_{suffix}"
    exp_dir = exp_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": stamp,
        "experiment": exp_name,
        "initialization": "scratch",
        "two_action_lqr": TWO_ACTION_LQR,
        "method": "Episode-level curriculum from privileged theta to MLE encoder mean",
        "encoder_training": "MSE/MLE only; PPO context gradients detached",
        "condition_on_uncertainty": False,
        "uncertainty_reward_penalty": 0.0,
        "arch": ARCH,
        "episode_steps": 512,
        "rollout_steps": 8_192,
        "window_length": 50,
        "nominal_warmup_steps": 49,
        "id_update_interval": 10,
        "gamma": 0.995,
        "phases": [
            {
                "phase": phase,
                "encoder_probability": encoder_prob,
                "privileged_probability": 1.0 - encoder_prob,
                "ent_coef": ent,
                "learning_rate": lr,
                "timesteps": steps,
            }
            for phase, encoder_prob, ent, lr, steps in phases
        ],
    }

    try:
        for phase_idx, (phase, encoder_prob, ent, lr, steps) in enumerate(phases):
            cfg = _base_stage_cfg(base_cfg, exp_root, exp_name, steps)
            _set_gradual_mle_curriculum(
                cfg,
                encoder_probability=encoder_prob,
                ent=ent,
                lr=lr,
            )

            cfg["training"]["load_weights"] = phase_idx > 0
            cfg["training"]["load_weights_from"] = str(exp_dir.resolve()) if phase_idx > 0 else None
            cfg["training"]["load_weights_name"] = "weights"
            cfg["training"]["load_encoder_only"] = False

            print(
                f"START gradual_mle/{phase}: {exp_name} "
                f"encoder_probability={encoder_prob:.2f} steps={steps}",
                flush=True,
            )
            _write_pipeline_file(exp_dir, payload)
            _run_with_config(cfg)
            print(f"END   gradual_mle/{phase}: {exp_name}", flush=True)
            _snapshot_phase_weights(exp_dir, phase)

    finally:
        _write_pipeline_file(exp_dir, payload)
        CONFIG_PATH.write_text(original_text)


if __name__ == "__main__":
    main()
