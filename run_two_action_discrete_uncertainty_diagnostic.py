from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import yaml

import lqr_theta_sweep_eval as sweep


ROOT = Path(__file__).resolve().parent
SELECTION_ROOT = ROOT / "experiments" / "two_action_system_selection"
DISCRETE_POINTER = SELECTION_ROOT / "latest_final_methods_discrete_theta.txt"
SWEEP_SCRIPT = ROOT / "lqr_theta_sweep_eval.py"

TAIL_THRESHOLD = 0.15
CENTER_THRESHOLD = 0.05
THETA_VALUES = (-0.25, 0.0, 0.25)
WINDOW_LENGTH = int(os.environ.get("TWO_ACTION_R22_WINDOW_LENGTH", "50"))
UPDATE_INTERVAL = int(os.environ.get("TWO_ACTION_R22_ID_UPDATE_INTERVAL", "10"))


def _smoke_enabled() -> bool:
    return os.environ.get("TWO_ACTION_DISCRETE_UNCERTAINTY_DIAG_SMOKE", "0").lower() in {
        "1",
        "true",
        "yes",
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
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
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_manifest() -> tuple[Path, dict[str, Any], dict[str, Path]]:
    if not DISCRETE_POINTER.exists():
        raise FileNotFoundError(
            "No discrete-theta final-methods run found. Expected "
            f"{DISCRETE_POINTER}."
        )
    run_root = Path(DISCRETE_POINTER.read_text().strip()).resolve()
    manifest_path = run_root / "final_methods.json"
    with manifest_path.open() as f:
        manifest = json.load(f)
    dirs = {key: Path(value).resolve() for key, value in manifest["dirs"].items()}
    return run_root, manifest, dirs


def _phase(env_name: str, default: str) -> str:
    phase = os.environ.get(env_name, default).strip()
    if not phase:
        raise ValueError(f"{env_name} cannot be empty.")
    return phase


def _matched_uncertainty(manifest: dict[str, Any], nll_label: str, nll_dir: Path) -> tuple[float, float, float]:
    compact = Path(manifest["sweeps"]["compact"]).resolve() / "theta_sweep_aggregate.csv"
    rows = [
        row
        for row in _read_csv(compact)
        if row["controller"] == nll_label
        and row.get("pred_theta_std_tail_mean_mean") not in {None, ""}
    ]
    if not rows:
        raise RuntimeError(f"Could not derive uncertainty distribution for {nll_label} from {compact}")
    raw = np.asarray([float(row["pred_theta_std_tail_mean_mean"]) for row in rows], dtype=np.float64)
    with (nll_dir / "config.yaml").open() as f:
        params = yaml.safe_load(f)["model"]["params"]
    scale = float(params.get("z_scale", 1.0))
    mean = float(np.mean(raw) * scale)
    std = float(np.std(raw) * scale)
    low = max(0.0, mean - math.sqrt(3.0) * std)
    high = mean + math.sqrt(3.0) * std
    if high <= low:
        high = low + 1e-6
    return mean, low, high


def _specs(
    dirs: dict[str, Path],
    *,
    mle_phase: str,
    nll_phase: str,
    matched: tuple[float, float, float],
) -> list[dict[str, Any]]:
    mean, _low, _high = matched
    mle_dir = str(dirs["gradual_mle"])
    nll_dir = str(dirs["gradual_nll"])
    return [
        {
            "label": f"discrete_mle_{mle_phase}",
            "kind": "ppo",
            "experiment": mle_dir,
            "weights_name": f"weights_{mle_phase}",
        },
        {
            "label": f"discrete_nll_{nll_phase}_predicted",
            "kind": "ppo",
            "experiment": nll_dir,
            "weights_name": f"weights_{nll_phase}",
            "uncertainty_override": "predicted",
        },
        {
            "label": f"discrete_nll_{nll_phase}_reflected",
            "kind": "ppo",
            "experiment": nll_dir,
            "weights_name": f"weights_{nll_phase}",
            "uncertainty_override": "reflected",
            "uncertainty_value": mean,
        },
        {
            "label": f"discrete_nll_{nll_phase}_constant",
            "kind": "ppo",
            "experiment": nll_dir,
            "weights_name": f"weights_{nll_phase}",
            "uncertainty_override": "constant",
            "uncertainty_value": mean,
        },
        {
            "label": f"discrete_nll_{nll_phase}_zero",
            "kind": "ppo",
            "experiment": nll_dir,
            "weights_name": f"weights_{nll_phase}",
            "uncertainty_override": "zeros",
        },
        {"label": "nominal_lqr", "kind": "nominal_lqr", "experiment": mle_dir},
        {"label": "oracle_lqr", "kind": "lqr", "experiment": mle_dir},
    ]


def _run_sweep(
    target: Path,
    specs: list[dict[str, Any]],
    output_subdir: str,
    *,
    episodes_per_theta: int,
    trace: bool,
) -> Path:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")
    env["THETA_SWEEP_TARGET_EXPERIMENT"] = str(target)
    env["THETA_SWEEP_OUTPUT_SUBDIR"] = output_subdir
    env["THETA_SWEEP_THETA_VALUES"] = ",".join(str(value) for value in THETA_VALUES)
    env["THETA_SWEEP_N_THETA_POINTS"] = str(len(THETA_VALUES))
    env["THETA_SWEEP_EPISODES_PER_THETA"] = str(episodes_per_theta)
    env["THETA_SWEEP_DETERMINISTIC"] = "1"
    env["THETA_SWEEP_RETURN_MODE"] = "quadratic"
    env["THETA_SWEEP_SAVE_TRAJECTORY_TRACE"] = "1" if trace else "0"
    env["THETA_SWEEP_SAVE_STEP_LEVEL_CSV"] = "0"
    env["THETA_SWEEP_COLLECT_STEP_LEVEL_PREDICTIONS"] = "1"
    env["THETA_SWEEP_CONTROLLER_SPECS_JSON"] = json.dumps(specs)
    print(f"START discrete uncertainty diagnostic sweep: {output_subdir}", flush=True)
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
                "mean_state_cost": _mean(rows, "episode_state_cost"),
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for row in rows:
        label = row["controller"].replace("discrete_", "").replace("_", " ")
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
        if row["controller"].startswith("discrete_") and row["posterior_abs_error"] != ""
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
    axes[0].set_xlabel(f"theta sensitivity in current {WINDOW_LENGTH}-step window")
    axes[0].set_ylabel("absolute posterior mean error")
    axes[1].set_title("Active-Window Information vs Policy Uncertainty Input")
    axes[1].set_xlabel(f"theta sensitivity in current {WINDOW_LENGTH}-step window")
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
        theta_info = [
            np.mean([float(row["theta_sensitivity_sq"]) for row in by_step[step]])
            for step in steps
        ]
        axes[0, 0].plot(steps, theta_info, label=controller)
        axes[0, 1].plot(steps, nominal_delta, label=controller)
        for action_idx, ax in enumerate(axes[1]):
            key = f"u{action_idx}"
            if key in next(iter(by_step.values()))[0]:
                values = [
                    np.mean([abs(float(row[key])) for row in by_step[step]])
                    for step in steps
                ]
                ax.plot(steps, values, label=controller)
    axes[0, 0].set_title("Immediate Theta Sensitivity")
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
    *,
    nll_phase: str,
) -> None:
    target_cfg = sweep._load_yaml(dirs["gradual_mle"] / "config.yaml")
    nll_label = f"discrete_nll_{nll_phase}_predicted"
    runtime = sweep.build_controller(
        {
            "label": "fixed_state",
            "kind": "ppo",
            "experiment": str(dirs["gradual_nll"]),
            "weights_name": f"weights_{nll_phase}",
            "uncertainty_override": "predicted",
        },
        target_cfg,
    )
    if runtime is None:
        raise RuntimeError(f"Could not load NLL {nll_phase} for fixed-state intervention.")
    model = runtime.model
    trace = [
        row
        for row in _read_csv(trace_dir / "trajectory_trace.csv")
        if row["controller"] == nll_label
        and int(row["step"]) >= WINDOW_LENGTH
        and int(row["step"]) % UPDATE_INTERVAL == 0
    ]
    if not trace:
        raise RuntimeError(f"No trace rows found for fixed-state intervention controller {nll_label}.")

    system_cfg = target_cfg["lqr_env"]
    r = np.asarray(system_cfg["R"], dtype=np.float64)
    delta_b = np.asarray(system_cfg["delta_B"], dtype=np.float64)

    factors = (0.0, 0.5, 1.0, 1.5)
    out: list[dict[str, Any]] = []
    model.policy.eval()
    with th.no_grad():
        for row in trace:
            step = int(row["step"])
            current_u = np.asarray([float(row[f"u{i}"]) for i in range(model.action_space.shape[0])])
            policy_obs_cols = sorted(
                (key for key in row if key.startswith("policy_obs")),
                key=lambda key: int(key.removeprefix("policy_obs")),
            )
            obs = np.asarray([float(row[key]) for key in policy_obs_cols], dtype=np.float32).reshape(1, -1)
            mean = float(row["latent_cached_scaled_0"])
            predicted_std = float(row["latent_cached_std_scaled_0"])
            nominal = np.asarray([float(row[f"u_nominal{i}"]) for i in range(model.action_space.shape[0])])
            for factor in factors:
                context = th.tensor([[mean, factor * predicted_std]], device=model.device, dtype=th.float32)
                action_th, _, _ = model.policy.forward_with_z(
                    th.as_tensor(obs, device=model.device, dtype=th.float32),
                    context,
                    deterministic=True,
                )
                action = action_th.cpu().numpy().reshape(-1)
                action = np.clip(action, model.action_space.low, model.action_space.high)
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
                        "distance_to_nominal_lqr": float(np.linalg.norm(action - nominal)),
                        "theta_sensitivity_sq": float(np.sum(np.square(delta_b @ nominal_delta))),
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
        ("distance_to_nominal_lqr", axes[1, 1], "Distance From Nominal-Manifold Action"),
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


def _run_fixed_state_intervention_subprocess(nll_phase: str) -> None:
    env = os.environ.copy()
    env["TWO_ACTION_DISCRETE_DIAG_FIXED_ONLY"] = "1"
    env["TWO_ACTION_DISCRETE_DIAG_NLL_PHASE"] = nll_phase
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp")
    subprocess.run(["python3", str(Path(__file__).resolve())], cwd=ROOT, env=env, check=True)


def main() -> None:
    run_root, manifest, dirs = _load_manifest()
    mle_phase = _phase("TWO_ACTION_DISCRETE_DIAG_MLE_PHASE", "P06")
    nll_phase = _phase("TWO_ACTION_DISCRETE_DIAG_NLL_PHASE", "P06")
    diagnostic_root = dirs["gradual_nll"] / f"discrete_uncertainty_diagnostic_{nll_phase.lower()}"
    if os.environ.get("TWO_ACTION_DISCRETE_DIAG_FIXED_ONLY", "0").lower() in {"1", "true", "yes"}:
        trace_dir = diagnostic_root / "focused_trace"
        _fixed_state_intervention(dirs, trace_dir, diagnostic_root, nll_phase=nll_phase)
        print(f"Saved fixed-state intervention to: {diagnostic_root}", flush=True)
        return

    nll_label = f"extended_nll_{nll_phase}"
    matched = _matched_uncertainty(manifest, nll_label, dirs["gradual_nll"])
    specs = _specs(dirs, mle_phase=mle_phase, nll_phase=nll_phase, matched=matched)

    diagnostic_root.mkdir(parents=True, exist_ok=True)

    full_dir = _run_sweep(
        dirs["gradual_nll"],
        specs,
        f"discrete_uncertainty_diagnostic_{nll_phase.lower()}/full_sweep",
        episodes_per_theta=1 if _smoke_enabled() else 20,
        trace=False,
    )
    trace_dir = _run_sweep(
        dirs["gradual_nll"],
        specs,
        f"discrete_uncertainty_diagnostic_{nll_phase.lower()}/focused_trace",
        episodes_per_theta=1 if _smoke_enabled() else 20,
        trace=True,
    )

    scorecard = _build_summary_scorecard(full_dir, diagnostic_root)
    windows = _build_window_rows(trace_dir, diagnostic_root)
    _plot_information_efficiency(scorecard, diagnostic_root)
    _plot_window_diagnostics(windows, diagnostic_root)
    _plot_trace_behavior(trace_dir, diagnostic_root)
    _run_fixed_state_intervention_subprocess(nll_phase)

    payload = {
        "source_final_methods_run": str(run_root),
        "source_final_methods_manifest": manifest,
        "smoke": _smoke_enabled(),
        "mle_phase": mle_phase,
        "nll_phase": nll_phase,
        "matched_uncertainty_scaled": {
            "mean": matched[0],
            "reference_low": matched[1],
            "reference_high": matched[2],
        },
        "theta_values": list(THETA_VALUES),
        "controllers": specs,
        "full_sweep": str(full_dir),
        "focused_trace": str(trace_dir),
        "primary_metric": "sum(theta_sensitivity_sq) / sum(actual_action_cost)",
    }
    with (diagnostic_root / "diagnostic_manifest.json").open("w") as f:
        json.dump(payload, f, indent=2)
    if not _smoke_enabled():
        (SELECTION_ROOT / "latest_discrete_uncertainty_diagnostic.txt").write_text(
            str(diagnostic_root.resolve()) + "\n"
        )
    print(f"Saved discrete-theta uncertainty diagnostic to: {diagnostic_root}", flush=True)


if __name__ == "__main__":
    main()
