from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from run_two_action_pipeline import (
    CONFIG_PATH,
    TWO_ACTION_LQR,
    _base_stage_cfg,
    _resolve_exp_root,
    _run_with_config,
    _set_encoder,
    _set_no_unc_policy,
    _set_privileged,
    _set_unc_policy,
    _snapshot_phase_weights,
)


RUN_CONTINUATION_BRANCHES = True
RUN_LARGE_NET_COMPARISON = True

SOURCE_UNC_RUN = "s_05-13__18-54_two_action_uncertainty_policy"

CONTINUATION_STEPS = 5_000_000
CONTINUATION_BRANCHES = [
    ("from_P04_pen0p20_ent0", "weights_P04", 0.20, 0.0, 5e-5),
    ("from_P04_pen0p175_ent0", "weights_P04", 0.175, 0.0, 5e-5),
    ("from_P04_pen0p175_ent0p002", "weights_P04", 0.175, 0.002, 5e-5),
    ("from_P05_pen0p175_ent0", "weights_P05", 0.175, 0.0, 5e-5),
    ("from_P05_pen0p15_ent0", "weights_P05", 0.15, 0.0, 5e-5),
    ("from_P05_pen0p15_ent0p002", "weights_P05", 0.15, 0.002, 5e-5),
]

LARGE_ARCH = [128, 128]
LARGE_PRIVILEGED_STEPS = 8_000_000
LARGE_ENCODER_STEPS = 6_000_000

LARGE_NO_UNC_PHASES = [
    ("P01", 0.20, 8e-4, 2_000_000),
    ("P02", 0.05, 4e-4, 2_000_000),
    ("P03", 0.01, 3e-4, 2_000_000),
    ("P04", 0.00, 1e-4, 2_000_000),
    ("P05", 0.00, 1e-4, 2_000_000),
]

LARGE_UNC_PHASES = [
    ("P01", 0.50, 0.03, 3e-4, 3_000_000),
    ("P02", 0.35, 0.03, 3e-4, 3_000_000),
    ("P03", 0.25, 0.02, 2e-4, 3_000_000),
    ("P04", 0.20, 0.02, 2e-4, 3_000_000),
    ("P05", 0.175, 0.02, 2e-4, 3_000_000),
]


def _set_arch(cfg: dict, arch: list[int]) -> None:
    params = cfg["model"]["params"]
    params["actor_net_arch"] = list(arch)
    params["critic_net_arch"] = list(arch)
    params["encoder_net_arch"] = list(arch)


def _write_schedule(schedule_dir: Path, payload: dict) -> None:
    schedule_dir.mkdir(parents=True, exist_ok=True)
    with open(schedule_dir / "matrix_schedule.yaml", "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _run_stage(label: str, cfg: dict, schedule_dir: Path, payload: dict) -> None:
    print(f"START {label}: {cfg['training']['experiment_name']} steps={cfg['total_timesteps']}", flush=True)
    _write_schedule(schedule_dir, payload)
    _run_with_config(cfg)
    print(f"END   {label}: {cfg['training']['experiment_name']}", flush=True)


def main() -> None:
    original_text = CONFIG_PATH.read_text()
    base_cfg = yaml.safe_load(original_text)
    exp_root = _resolve_exp_root(base_cfg)
    stamp = datetime.now().strftime("%m-%d__%H-%M")
    schedule_dir = exp_root / f"s_{stamp}_two_action_next_matrix"
    source_unc_dir = exp_root / SOURCE_UNC_RUN

    payload = {
        "timestamp": stamp,
        "two_action_lqr": TWO_ACTION_LQR,
        "source_uncertainty_run": str(source_unc_dir.resolve()),
        "runs": [],
    }

    try:
        if RUN_CONTINUATION_BRANCHES:
            if not source_unc_dir.exists():
                raise FileNotFoundError(f"Missing source run: {source_unc_dir}")

            for label, weight_name, penalty, ent, lr in CONTINUATION_BRANCHES:
                if not (source_unc_dir / f"{weight_name}.zip").exists():
                    raise FileNotFoundError(f"Missing {weight_name}.zip in {source_unc_dir}")

                exp_name = f"s_{stamp}_two_action_unc_cont_{label}"
                cfg = _base_stage_cfg(base_cfg, exp_root, exp_name, CONTINUATION_STEPS)
                _set_unc_policy(cfg, penalty=penalty, ent=ent, lr=lr)
                cfg["training"]["load_weights"] = True
                cfg["training"]["load_weights_from"] = str(source_unc_dir.resolve())
                cfg["training"]["load_weights_name"] = weight_name
                cfg["training"]["load_encoder_only"] = False

                payload["runs"].append({
                    "name": exp_name,
                    "type": "continuation_branch",
                    "source": SOURCE_UNC_RUN,
                    "source_weights": weight_name,
                    "penalty": penalty,
                    "ent_coef": ent,
                    "learning_rate": lr,
                    "timesteps": CONTINUATION_STEPS,
                    "arch": [64, 64],
                })
                _run_stage(f"continuation/{label}", cfg, schedule_dir, payload)

        if RUN_LARGE_NET_COMPARISON:
            names = {
                "privileged": f"s_{stamp}_two_action_large_privileged",
                "encoder": f"s_{stamp}_two_action_large_nll_encoder_rand_exp",
                "no_unc": f"s_{stamp}_two_action_large_no_uncertainty_baseline",
                "unc": f"s_{stamp}_two_action_large_uncertainty_policy",
            }
            dirs = {key: exp_root / value for key, value in names.items()}

            cfg = _base_stage_cfg(base_cfg, exp_root, names["privileged"], LARGE_PRIVILEGED_STEPS)
            _set_privileged(cfg)
            _set_arch(cfg, LARGE_ARCH)
            cfg["training"]["load_weights"] = False
            cfg["training"]["load_weights_from"] = None
            cfg["training"]["load_encoder_only"] = False
            payload["runs"].append({
                "name": names["privileged"],
                "type": "large_privileged",
                "arch": LARGE_ARCH,
                "timesteps": LARGE_PRIVILEGED_STEPS,
                "learning_rate": cfg["model"]["params"]["learning_rate"],
                "ent_coef": cfg["model"]["params"]["ent_coef"],
            })
            _run_stage("large/privileged", cfg, schedule_dir, payload)

            cfg = _base_stage_cfg(base_cfg, exp_root, names["encoder"], LARGE_ENCODER_STEPS)
            _set_encoder(cfg)
            _set_arch(cfg, LARGE_ARCH)
            cfg["training"]["load_weights"] = True
            cfg["training"]["load_weights_from"] = str(dirs["privileged"].resolve())
            cfg["training"]["load_weights_name"] = "weights"
            cfg["training"]["load_encoder_only"] = False
            payload["runs"].append({
                "name": names["encoder"],
                "type": "large_encoder_nll",
                "arch": LARGE_ARCH,
                "load_weights_from": names["privileged"],
                "load_weights_name": "weights",
                "timesteps": LARGE_ENCODER_STEPS,
                "learning_rate": cfg["model"]["params"]["learning_rate"],
                "naive_action_noise_std": cfg["model"]["params"]["naive_action_noise_std"],
                "naive_action_noise_dist": cfg["model"]["params"]["naive_action_noise_dist"],
            })
            _run_stage("large/encoder", cfg, schedule_dir, payload)

            no_unc_dir = dirs["no_unc"]
            no_unc_dir.mkdir(parents=True, exist_ok=True)
            payload["runs"].append({
                "name": names["no_unc"],
                "type": "large_no_uncertainty_baseline",
                "arch": LARGE_ARCH,
                "load_encoder_from": names["encoder"],
                "phases": [
                    {"phase": p, "ent_coef": ent, "learning_rate": lr, "timesteps": steps}
                    for p, ent, lr, steps in LARGE_NO_UNC_PHASES
                ],
            })
            for phase_idx, (phase, ent, lr, steps) in enumerate(LARGE_NO_UNC_PHASES):
                cfg = _base_stage_cfg(base_cfg, exp_root, names["no_unc"], steps)
                _set_no_unc_policy(cfg, ent=ent, lr=lr)
                _set_arch(cfg, LARGE_ARCH)
                cfg["training"]["load_weights"] = True
                cfg["training"]["load_weights_from"] = str(dirs["encoder"].resolve()) if phase_idx == 0 else str(no_unc_dir.resolve())
                cfg["training"]["load_weights_name"] = "weights_best" if phase_idx == 0 else "weights"
                cfg["training"]["load_encoder_only"] = phase_idx == 0
                _run_stage(f"large/no_uncertainty/{phase}", cfg, schedule_dir, payload)
                _snapshot_phase_weights(no_unc_dir, phase)

            unc_dir = dirs["unc"]
            unc_dir.mkdir(parents=True, exist_ok=True)
            payload["runs"].append({
                "name": names["unc"],
                "type": "large_uncertainty_policy",
                "arch": LARGE_ARCH,
                "load_encoder_from": names["encoder"],
                "uncertainty_penalty_metric": "std",
                "phases": [
                    {"phase": p, "penalty": pen, "ent_coef": ent, "learning_rate": lr, "timesteps": steps}
                    for p, pen, ent, lr, steps in LARGE_UNC_PHASES
                ],
            })
            for phase_idx, (phase, penalty, ent, lr, steps) in enumerate(LARGE_UNC_PHASES):
                cfg = _base_stage_cfg(base_cfg, exp_root, names["unc"], steps)
                _set_unc_policy(cfg, penalty=penalty, ent=ent, lr=lr)
                _set_arch(cfg, LARGE_ARCH)
                cfg["training"]["load_weights"] = True
                cfg["training"]["load_weights_from"] = str(dirs["encoder"].resolve()) if phase_idx == 0 else str(unc_dir.resolve())
                cfg["training"]["load_weights_name"] = "weights_best" if phase_idx == 0 else "weights"
                cfg["training"]["load_encoder_only"] = phase_idx == 0
                _run_stage(f"large/uncertainty/{phase}", cfg, schedule_dir, payload)
                _snapshot_phase_weights(unc_dir, phase)

    finally:
        _write_schedule(schedule_dir, payload)
        CONFIG_PATH.write_text(original_text)


if __name__ == "__main__":
    main()
