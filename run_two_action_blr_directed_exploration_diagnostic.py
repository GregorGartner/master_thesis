from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import lqr_theta_sweep_eval as sweep
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import yaml


ROOT = Path(__file__).resolve().parent
SELECTION_ROOT = ROOT / "experiments" / "two_action_system_selection"
BLR_POINTER = SELECTION_ROOT / "latest_blr_ppo.txt"
SWEEP_SCRIPT = ROOT / "lqr_theta_sweep_eval.py"

TAIL_THRESHOLD = 0.15
CENTER_THRESHOLD = 0.05
TRACE_THETAS = (-0.25, -0.125, 0.0, 0.125, 0.25)
WINDOW_LENGTH = 50
UPDATE_INTERVAL = 10


def _smoke_enabled() -> bool:
    return os.environ.get("TWO_ACTION_BLR_DIRECTED_EXPLORATION_SMOKE", "0").lower() in {
        "1",
        "true",
        "yes",
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_manifest() -> tuple[Path, dict[str, Any], dict[str, Path]]:
    if not BLR_POINTER.exists():
        raise FileNotFoundError("No BLR+PPO run found. Run run_two_action_selected_system_blr_ppo.py first.")
    run_root = Path(BLR_POINTER.read_text().strip()).resolve()
    manifest_path = run_root / "blr_ppo.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    dirs = {key: Path(value).resolve() for key, value in manifest["dirs"].items()}
    required = {
        "mean": dirs["mean"] / "weights_P06.zip",
        "mean_std": dirs["mean_std"] / "weights_P02.zip",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required BLR checkpoints: {missing}")
    return run_root, manifest, dirs


def _matched_uncertainty(dirs: dict[str, Path]) -> tuple[float, float, float]:
    aggregate = dirs["mean"] / "theta_sweep_blr_compact_all_phases" / "theta_sweep_aggregate.csv"
    rows = [
        row
        for row in _read_csv(aggregate)
        if row["controller"] == "blr_mean_std_P02"
        and row.get("pred_theta_std_tail_mean_mean") not in {None, ""}
    ]
    if not rows:
        raise RuntimeError(f"Could not derive P02 uncertainty distribution from {aggregate}")
    raw = np.asarray([float(row["pred_theta_std_tail_mean_mean"]) for row in rows], dtype=np.float64)
    with open(dirs["mean_std"] / "config.yaml") as f:
        params = yaml.safe_load(f)["model"]["params"]
    scale = float(params.get("z_scale", 1.0))
    mean = float(np.mean(raw) * scale)
    std = float(np.std(raw) * scale)
    low = max(0.0, mean - math.sqrt(3.0) * std)
    high = mean + math.sqrt(3.0) * std
    if high <= low:
        high = low + 1e-6
    return mean, low, high


def _specs(dirs: dict[str, Path], matched: tuple[float, float, float]) -> list[dict[str, Any]]:
    mean, low, high = matched
    mean_std = str(dirs["mean_std"])
    return [
        {
            "label": "blr_mean_P06",
            "kind": "ppo",
            "experiment": str(dirs["mean"]),
            "weights_name": "weights_P06",
        },
        {
            "label": "blr_mean_std_P02_predicted",
            "kind": "ppo",
            "experiment": mean_std,
            "weights_name": "weights_P02",
            "uncertainty_override": "predicted",
        },
        {
            "label": "blr_mean_std_P02_reflected",
            "kind": "ppo",
            "experiment": mean_std,
            "weights_name": "weights_P02",
            "uncertainty_override": "reflected",
            "uncertainty_value": mean,
        },
        {
            "label": "blr_mean_std_P02_constant",
            "kind": "ppo",
            "experiment": mean_std,
            "weights_name": "weights_P02",
            "uncertainty_override": "constant",
            "uncertainty_value": mean,
        },
        {
            "label": "blr_mean_std_P02_zero",
            "kind": "ppo",
            "experiment": mean_std,
            "weights_name": "weights_P02",
            "uncertainty_override": "zeros",
        },
        {"label": "nominal_lqr", "kind": "nominal_lqr", "experiment": str(dirs["mean"])},
        {"label": "oracle_lqr", "kind": "lqr", "experiment": str(dirs["mean"])},
    ]


def _run_sweep(
    target: Path,
    specs: list[dict[str, Any]],
    output_subdir: str,
    *,
    theta_points: int,
    episodes_per_theta: int,
    trace: bool,
) -> Path:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")
    env["THETA_SWEEP_TARGET_EXPERIMENT"] = str(target)
    env["THETA_SWEEP_OUTPUT_SUBDIR"] = output_subdir
    env["THETA_SWEEP_N_THETA_POINTS"] = str(theta_points)
    env["THETA_SWEEP_EPISODES_PER_THETA"] = str(episodes_per_theta)
    env["THETA_SWEEP_DETERMINISTIC"] = "1"
    env["THETA_SWEEP_RETURN_MODE"] = "quadratic"
    env["THETA_SWEEP_SAVE_TRAJECTORY_TRACE"] = "1" if trace else "0"
    env["THETA_SWEEP_SAVE_STEP_LEVEL_CSV"] = "0"
    env["THETA_SWEEP_COLLECT_STEP_LEVEL_PREDICTIONS"] = "1"
    env["THETA_SWEEP_CONTROLLER_SPECS_JSON"] = json.dumps(specs)
    if trace:
        trace_thetas = (-0.25, 0.0, 0.25) if _smoke_enabled() else TRACE_THETAS
        env["THETA_SWEEP_THETA_VALUES"] = ",".join(str(value) for value in trace_thetas)
    print(f"START diagnostic sweep: {output_subdir}", flush=True)
    subprocess.run(["python3", str(SWEEP_SCRIPT)], cwd=ROOT, env=env, check=True)
    output = target / output_subdir
    if not (output / "controller_scorecard.csv").exists():
        raise RuntimeError(f"Diagnostic sweep did not finish: {output}")
    return output


def _mean(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
    return float(np.mean(values)) if values else float("nan")


def _ratio_of_sums(rows: list[dict[str, str]], numerator: str, denominator: str) -> float:
    num = sum(float(row[numerator]) for row in rows)
    den = sum(float(row[denominator]) for row in rows)
    return float(num / max(den, 1e-12))


def _build_summary_scorecard(full_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    episodes = _read_csv(full_dir / "episode_summary.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in episodes:
        grouped[row["controller"]].append(row)

    out: list[dict[str, Any]] = []
    for controller, rows in sorted(grouped.items()):
        tail = [row for row in rows if abs(float(row["theta"])) >= TAIL_THRESHOLD]
        center = [row for row in rows if abs(float(row["theta"])) <= CENTER_THRESHOLD]
        out.append(
            {
                "controller": controller,
                "num_episodes": len(rows),
                "mean_return": _mean(rows, "episode_return"),
                "tail_mean_return": _mean(tail, "episode_return"),
                "center_mean_return": _mean(center, "episode_return"),
                "mean_theta_rmse": _mean(rows, "theta_rmse_tail"),
                "mean_action_cost": _mean(rows, "episode_action_cost"),
                "mean_theta_sensitivity": _mean(rows, "episode_theta_sensitivity_sq"),
                "information_per_action_cost": _ratio_of_sums(
                    rows, "episode_theta_sensitivity_sq", "episode_action_cost"
                ),
                "mean_nominal_manifold_deviation_r": _mean(rows, "episode_nominal_deviation_r"),
                "information_per_nominal_manifold_deviation_r": _ratio_of_sums(
                    rows, "episode_theta_sensitivity_sq", "episode_nominal_deviation_r"
                ),
            }
        )
    _write_csv(output_dir / "directed_exploration_scorecard.csv", out)
    return out


def _build_window_rows(trace_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    trace = _read_csv(trace_dir / "trajectory_trace.csv")
    grouped: dict[tuple[str, float, int], list[dict[str, str]]] = defaultdict(list)
    for row in trace:
        grouped[(row["controller"], float(row["theta"]), int(row["seed"]))].append(row)

    out: list[dict[str, Any]] = []
    for (controller, theta, seed), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["step"]))
        for end in range(WINDOW_LENGTH, len(rows) + 1, UPDATE_INTERVAL):
            window = rows[end - WINDOW_LENGTH : end]
            last = window[-1]
            theta_hat = last.get("latent_cached_raw_0", "")
            theta_std = last.get("latent_cached_std_raw_0", "")
            action_cost = sum(float(row["action_cost"]) for row in window)
            info = sum(float(row["theta_sensitivity_sq"]) for row in window)
            nominal_dev_r = sum(float(row["nominal_deviation_r"]) for row in window)
            out.append(
                {
                    "controller": controller,
                    "theta": theta,
                    "seed": seed,
                    "window_end_step": int(last["step"]),
                    "window_theta_sensitivity": info,
                    "window_action_cost": action_cost,
                    "window_nominal_deviation_r": nominal_dev_r,
                    "information_per_action_cost": info / max(action_cost, 1e-12),
                    "information_per_nominal_deviation_r": info / max(nominal_dev_r, 1e-12),
                    "posterior_theta": theta_hat,
                    "posterior_abs_error": (
                        abs(float(theta_hat) - theta) if theta_hat not in {None, ""} else ""
                    ),
                    "posterior_std": theta_std,
                }
            )
    _write_csv(output_dir / "window_diagnostics.csv", out)
    return out


def _plot_information_efficiency(scorecard: list[dict[str, Any]], output_dir: Path) -> None:
    rows = [row for row in scorecard if row["controller"] not in {"nominal_lqr", "oracle_lqr"}]
    short_labels = {
        "blr_mean_P06": "mean P06",
        "blr_mean_std_P02_predicted": "mean+std predicted",
        "blr_mean_std_P02_reflected": "mean+std reflected",
        "blr_mean_std_P02_constant": "mean+std constant",
        "blr_mean_std_P02_zero": "mean+std zero",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for row in rows:
        label = short_labels.get(row["controller"], row["controller"])
        axes[0].scatter(row["mean_action_cost"], row["mean_theta_sensitivity"], s=70, label=label)
        axes[1].bar(label, row["information_per_action_cost"])
    axes[0].set_title("Information Generated vs Actual Action Cost")
    axes[0].set_xlabel("mean episode action cost")
    axes[0].set_ylabel("mean episode theta sensitivity")
    axes[1].set_title("Information per Action Cost")
    axes[1].set_ylabel("sum theta sensitivity / sum action cost")
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "information_efficiency.png", dpi=200)
    plt.close(fig)


def _plot_window_diagnostics(window_rows: list[dict[str, Any]], output_dir: Path) -> None:
    relevant = [
        row
        for row in window_rows
        if row["controller"].startswith("blr_") and row["posterior_abs_error"] != ""
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in relevant:
        groups[row["controller"]].append(row)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for controller, rows in sorted(groups.items()):
        axes[0].scatter(
            [row["window_theta_sensitivity"] for row in rows],
            [row["posterior_abs_error"] for row in rows],
            s=8,
            alpha=0.2,
            label=controller,
        )
        std_rows = [row for row in rows if row["posterior_std"] != ""]
        if std_rows:
            axes[1].scatter(
                [row["window_theta_sensitivity"] for row in std_rows],
                [float(row["posterior_std"]) for row in std_rows],
                s=8,
                alpha=0.2,
                label=controller,
            )
    axes[0].set_title("Active-Window Information vs Posterior Error")
    axes[0].set_xlabel("theta sensitivity in current 50-step window")
    axes[0].set_ylabel("absolute posterior mean error")
    axes[1].set_title("Active-Window Information vs Policy Uncertainty Input")
    axes[1].set_xlabel("theta sensitivity in current 50-step window")
    axes[1].set_ylabel("cached uncertainty input (raw theta units)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "window_information_and_identification.png", dpi=200)
    plt.close(fig)


def _plot_trace_behavior(trace_dir: Path, output_dir: Path) -> None:
    trace = _read_csv(trace_dir / "trajectory_trace.csv")
    groups: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in trace:
        groups[row["controller"]][int(row["step"])].append(row)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for controller, by_step in sorted(groups.items()):
        steps = sorted(by_step)
        oracle_delta = [
            np.mean(
                [
                    math.sqrt(
                        sum(float(value) ** 2 for key, value in row.items() if key.startswith("u_delta_lqr"))
                    )
                    for row in by_step[step]
                ]
            )
            for step in steps
        ]
        nominal_delta = [
            np.mean(
                [
                    math.sqrt(
                        sum(
                            float(value) ** 2
                            for key, value in row.items()
                            if key.startswith("u_delta_nominal")
                        )
                    )
                    for row in by_step[step]
                ]
            )
            for step in steps
        ]
        axes[0, 0].plot(steps, oracle_delta, label=controller)
        axes[0, 1].plot(steps, nominal_delta, label=controller)
        for action_idx, ax in enumerate(axes[1]):
            key = f"u{action_idx}"
            if key in next(iter(by_step.values()))[0]:
                values = [
                    np.mean([abs(float(row[key])) for row in by_step[step]])
                    for step in steps
                ]
                ax.plot(steps, values, label=controller)
    axes[0, 0].set_title("Distance From Oracle-LQR Action")
    axes[0, 1].set_title("Distance From Nominal-Manifold Action")
    axes[1, 0].set_title("Mean Absolute Action Component 1")
    axes[1, 1].set_title("Mean Absolute Action Component 2")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_dir / "control_behavior_and_actions.png", dpi=200)
    plt.close(fig)


def _fixed_state_intervention(
    dirs: dict[str, Path],
    trace_dir: Path,
    output_dir: Path,
) -> None:
    target_cfg = sweep._load_yaml(dirs["mean"] / "config.yaml")
    runtime = sweep.build_controller(
        {
            "label": "fixed_state",
            "kind": "ppo",
            "experiment": str(dirs["mean_std"]),
            "weights_name": "weights_P02",
            "uncertainty_override": "predicted",
        },
        target_cfg,
    )
    if runtime is None:
        raise RuntimeError("Could not load BLR mean+std P02 for fixed-state intervention.")
    model = runtime.model
    trace = [
        row
        for row in _read_csv(trace_dir / "trajectory_trace.csv")
        if row["controller"] == "blr_mean_std_P02_predicted"
        and int(row["step"]) >= WINDOW_LENGTH
        and int(row["step"]) % UPDATE_INTERVAL == 0
    ]
    system_cfg = target_cfg["lqr_env"]
    a_nom = np.asarray(system_cfg["A"], dtype=np.float64)
    b_nom = np.asarray(system_cfg["B"], dtype=np.float64)
    q = np.asarray(system_cfg["Q"], dtype=np.float64)
    r = np.asarray(system_cfg["R"], dtype=np.float64)
    p_nom = sweep.solve_discrete_are(a_nom, b_nom, q, r)
    k_nom = np.linalg.solve(r + b_nom.T @ p_nom @ b_nom, b_nom.T @ p_nom @ a_nom)
    delta_b = np.asarray(system_cfg["delta_B"], dtype=np.float64)
    factors = (0.0, 0.5, 1.0, 1.5)
    out: list[dict[str, Any]] = []
    model.policy.eval()
    with th.no_grad():
        for row in trace:
            step = int(row["step"])
            current_u = np.asarray([float(row[f"u{i}"]) for i in range(model.action_space.shape[0])])
            x = np.asarray([float(row[f"x{i}"]) for i in range(k_nom.shape[1])], dtype=np.float32)
            policy_obs_cols = sorted(
                (key for key in row if key.startswith("policy_obs")),
                key=lambda key: int(key.removeprefix("policy_obs")),
            )
            obs = np.asarray([float(row[key]) for key in policy_obs_cols], dtype=np.float32).reshape(1, -1)
            mean = float(row["latent_cached_scaled_0"])
            predicted_std = float(row["latent_cached_std_scaled_0"])
            oracle = np.asarray(
                [float(row[f"u_lqr{i}"]) for i in range(model.action_space.shape[0])]
            )
            nominal = np.asarray(
                [float(row[f"u_nominal{i}"]) for i in range(model.action_space.shape[0])]
            )
            for factor in factors:
                context = th.tensor([[mean, factor * predicted_std]], device=model.device, dtype=th.float32)
                action_th, _, _ = model.policy.forward_with_z(
                    th.as_tensor(obs, device=model.device, dtype=th.float32),
                    context,
                    deterministic=True,
                )
                action = action_th.cpu().numpy().reshape(-1)
                action = np.clip(action, model.action_space.low, model.action_space.high)
                mismatch = k_nom @ x.astype(np.float64) + action
                nominal_delta = action - nominal
                out.append(
                    {
                        "theta": row["theta"],
                        "seed": row["seed"],
                        "step": step,
                        "uncertainty_factor": factor,
                        "action0": action[0],
                        "action1": action[1],
                        "distance_to_observed_predicted_action": float(np.linalg.norm(action - current_u)),
                        "distance_to_oracle_lqr": float(np.linalg.norm(action - oracle)),
                        "distance_to_nominal_lqr": float(np.linalg.norm(action - nominal)),
                        "theta_sensitivity_sq": float(np.sum(np.square(delta_b @ mismatch))),
                        "action_cost": float(action @ r @ action),
                        "nominal_deviation_r": float(nominal_delta @ r @ nominal_delta),
                    }
                )
    runtime.env.close()
    reproduction_errors = [
        float(row["distance_to_observed_predicted_action"])
        for row in out
        if float(row["uncertainty_factor"]) == 1.0
    ]
    if reproduction_errors and max(reproduction_errors) > 1e-6:
        raise RuntimeError(
            "Fixed-state intervention failed to reproduce the recorded predicted-uncertainty action."
        )
    _write_csv(output_dir / "fixed_state_intervention.csv", out)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for metric, ax, title in [
        ("distance_to_observed_predicted_action", axes[0, 0], "Action Change From Predicted-Std Decision"),
        ("theta_sensitivity_sq", axes[0, 1], "Immediate Theta Sensitivity"),
        ("action_cost", axes[1, 0], "Immediate Action Cost"),
        ("distance_to_oracle_lqr", axes[1, 1], "Distance From Oracle-LQR Action"),
    ]:
        means = [
            np.mean([float(row[metric]) for row in out if float(row["uncertainty_factor"]) == factor])
            for factor in factors
        ]
        ax.plot(factors, means, marker="o")
        ax.set_title(title)
        ax.set_xlabel("predicted uncertainty multiplier")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fixed_state_uncertainty_response.png", dpi=200)
    plt.close(fig)


def main() -> None:
    run_root, manifest, dirs = _load_manifest()
    matched = _matched_uncertainty(dirs)
    specs = _specs(dirs, matched)
    diagnostic_root = dirs["mean"] / "blr_directed_exploration_diagnostic"
    diagnostic_root.mkdir(parents=True, exist_ok=True)

    full_dir = _run_sweep(
        dirs["mean"],
        specs,
        "blr_directed_exploration_diagnostic/full_sweep",
        theta_points=3 if _smoke_enabled() else 41,
        episodes_per_theta=1 if _smoke_enabled() else 20,
        trace=False,
    )
    trace_dir = _run_sweep(
        dirs["mean"],
        specs,
        "blr_directed_exploration_diagnostic/focused_trace",
        theta_points=3 if _smoke_enabled() else len(TRACE_THETAS),
        episodes_per_theta=1 if _smoke_enabled() else 20,
        trace=True,
    )

    scorecard = _build_summary_scorecard(full_dir, diagnostic_root)
    windows = _build_window_rows(trace_dir, diagnostic_root)
    _plot_information_efficiency(scorecard, diagnostic_root)
    _plot_window_diagnostics(windows, diagnostic_root)
    _plot_trace_behavior(trace_dir, diagnostic_root)
    _fixed_state_intervention(dirs, trace_dir, diagnostic_root)

    payload = {
        "source_blr_run": str(run_root),
        "source_blr_manifest": manifest,
        "smoke": _smoke_enabled(),
        "matched_uncertainty_scaled": {
            "mean": matched[0],
            "reference_low": matched[1],
            "reference_high": matched[2],
        },
        "controllers": specs,
        "full_sweep": str(full_dir),
        "focused_trace": str(trace_dir),
        "primary_metric": "sum(theta_sensitivity_sq) / sum(actual_action_cost)",
    }
    with open(diagnostic_root / "diagnostic_manifest.json", "w") as f:
        json.dump(payload, f, indent=2)
    if not _smoke_enabled():
        (SELECTION_ROOT / "latest_blr_directed_exploration.txt").write_text(
            str(diagnostic_root.resolve()) + "\n"
        )
    print(f"Saved BLR directed-exploration diagnostic to: {diagnostic_root}", flush=True)


if __name__ == "__main__":
    main()
