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


RUN_SUFFIX = "two_action_transformer_policy"
ACTOR_CRITIC_ARCH = [128, 128]
LATENT_DIM = 32
TRANSFORMER_D_MODEL = 32
TRANSFORMER_N_HEADS = 4
TRANSFORMER_FF_DIM = 64
TRANSFORMER_N_LAYERS = 1
TRANSFORMER_DROPOUT = 0.0

PHASES = [
    ("P01", 0.05, 3e-4, 3_000_000),
    ("P02", 0.02, 3e-4, 3_000_000),
    ("P03", 0.01, 2e-4, 3_000_000),
    ("P04", 0.005, 1e-4, 3_000_000),
]


def _smoke_enabled() -> bool:
    return os.environ.get("TWO_ACTION_TRANSFORMER_SMOKE", "0").lower() in {"1", "true", "yes"}


def _selected_phases() -> list[tuple[str, float, float, int]]:
    if _smoke_enabled():
        return [
            ("P01", 0.05, 3e-4, 512),
            ("P02", 0.02, 3e-4, 512),
        ]
    return PHASES


def _set_transformer_policy(cfg: dict, *, ent: float, lr: float) -> None:
    params = cfg["model"]["params"]
    _set_common_model_params(cfg)

    params["context_mode"] = "encoder_mle"
    params["encoder_type"] = "transformer"
    params["latent_dim"] = LATENT_DIM
    params["freeze_ppo"] = False
    params["regression_coef"] = 0.0
    params["condition_on_uncertainty"] = False
    params["uncertainty_regularization_coef"] = 0.0
    params["uncertainty_reward_penalty_coef"] = 0.0
    params["detach_context_for_rl"] = False
    params["id_update_interval"] = 1
    params["nominal_warmup_steps"] = 0
    params["actor_net_arch"] = list(ACTOR_CRITIC_ARCH)
    params["critic_net_arch"] = list(ACTOR_CRITIC_ARCH)
    params["transformer_d_model"] = TRANSFORMER_D_MODEL
    params["transformer_n_heads"] = TRANSFORMER_N_HEADS
    params["transformer_ff_dim"] = TRANSFORMER_FF_DIM
    params["transformer_n_layers"] = TRANSFORMER_N_LAYERS
    params["transformer_dropout"] = TRANSFORMER_DROPOUT
    params["learning_rate"] = float(lr)
    params["ent_coef"] = float(ent)

    if _smoke_enabled():
        params["n_steps"] = 128
        params["batch_size"] = 64

    for cb in cfg.get("callbacks", []):
        if cb.get("name") == "LivePlotCallback":
            cb["enabled"] = True
        if cb.get("name") == "SaveModelCallback":
            cb.setdefault("params", {})
            cb["params"]["best_metric"] = "episode_reward"
            cb["params"]["save_on_training_end"] = True


def _write_pipeline_file(exp_dir: Path, payload: dict) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "two_action_transformer_policy.yaml", "w") as f:
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
        "model": "UnifiedContextPPO transformer history encoder, no regression loss",
        "history": "[x_i, u_i, delta_x_i] over the latest 50 completed transitions",
        "actor_critic_arch": ACTOR_CRITIC_ARCH,
        "latent_dim": LATENT_DIM,
        "transformer": {
            "d_model": TRANSFORMER_D_MODEL,
            "n_heads": TRANSFORMER_N_HEADS,
            "ff_dim": TRANSFORMER_FF_DIM,
            "n_layers": TRANSFORMER_N_LAYERS,
            "dropout": TRANSFORMER_DROPOUT,
            "pooling": "masked_mean",
        },
        "phases": [
            {"phase": phase, "ent_coef": ent, "learning_rate": lr, "timesteps": steps}
            for phase, ent, lr, steps in phases
        ],
    }

    try:
        for phase_idx, (phase, ent, lr, steps) in enumerate(phases):
            cfg = _base_stage_cfg(base_cfg, exp_root, exp_name, steps)
            if _smoke_enabled():
                cfg["lqr_env"]["max_episode_steps"] = 64
            _set_transformer_policy(cfg, ent=ent, lr=lr)

            cfg["training"]["load_weights"] = phase_idx > 0
            cfg["training"]["load_weights_from"] = str(exp_dir.resolve()) if phase_idx > 0 else None
            cfg["training"]["load_weights_name"] = "weights"
            cfg["training"]["load_encoder_only"] = False

            _run_stage(f"transformer/{phase}", cfg, exp_dir, payload)
            _snapshot_phase_weights(exp_dir, phase)

    finally:
        _write_pipeline_file(exp_dir, payload)
        CONFIG_PATH.write_text(original_text)


if __name__ == "__main__":
    main()
