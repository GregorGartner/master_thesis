from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import closed_form_system_search as search


OUTPUT_ROOT = Path("experiments") / "two_action_system_selection"

A0 = np.array([[0.98, 0.0], [0.0, 0.95]], dtype=np.float64)
B0 = np.array([[0.1, 0.03], [1.0, 0.2]], dtype=np.float64)
DELTA_B = np.array([[0.5, 0.02], [0.15, 1.0]], dtype=np.float64)
Q = np.diag([2.0, 1.0]).astype(np.float64)
R22_VALUES = (0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 2.75, 3.5, 5.0)
DITHER_STDS = (0.0, 0.02, 0.05, 0.1, 0.2)

THETA_MAX = 0.25
THETA_POINTS = int(os.environ.get("TWO_ACTION_SCREEN_THETA_POINTS", "21"))
SEED_COUNT = int(os.environ.get("TWO_ACTION_SCREEN_SEED_COUNT", "10"))
HORIZON = 512
WINDOW_LENGTH = 50
NOMINAL_WARMUP_STEPS = 49
ID_UPDATE_INTERVAL = 10
PROCESS_NOISE_STD = 0.05
INITIAL_STATE_LOW = -0.3
INITIAL_STATE_HIGH = 0.3
ACTION_LOW = -30.0
ACTION_HIGH = 30.0

TAIL_THRESHOLD = 0.15
CENTER_THRESHOLD = 0.05


def _label(r22: float) -> str:
    return f"r22_{r22:g}".replace(".", "p")


def _family(r22: float) -> search.Family:
    r = np.diag([5.0, float(r22)]).astype(np.float64)
    k_nom, _ = search.lqr_gain(A0, B0, Q, r)
    return search.Family(
        A0=A0.copy(),
        B0=B0.copy(),
        DeltaB=DELTA_B.copy(),
        Q=Q.copy(),
        R=r,
        K_nom=k_nom,
        DeltaA=DELTA_B @ k_nom,
    )


def _system_dict(r22: float, family: search.Family) -> dict:
    return {
        "label": _label(r22),
        "r22": float(r22),
        "A": family.A0.tolist(),
        "B": family.B0.tolist(),
        "delta_B": family.DeltaB.tolist(),
        "Q": family.Q.tolist(),
        "R": family.R.tolist(),
        "theta_max": THETA_MAX,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict]) -> dict:
    returns = np.asarray([row["return"] for row in rows], dtype=np.float64)
    thetas = np.asarray([row["theta"] for row in rows], dtype=np.float64)
    failures = np.asarray([row["failed"] for row in rows], dtype=np.float64)
    theta_rmse_values = np.asarray([row["theta_rmse"] for row in rows], dtype=np.float64)
    tail = np.abs(thetas) >= TAIL_THRESHOLD
    center = np.abs(thetas) <= CENTER_THRESHOLD
    neg_tail = thetas <= -TAIL_THRESHOLD
    pos_tail = thetas >= TAIL_THRESHOLD
    return {
        "mean_return": float(np.mean(returns)),
        "tail_return": float(np.mean(returns[tail])),
        "center_return": float(np.mean(returns[center])),
        "negative_tail_return": float(np.mean(returns[neg_tail])),
        "positive_tail_return": float(np.mean(returns[pos_tail])),
        "theta_rmse": float(np.sqrt(np.mean(theta_rmse_values))),
        "info_proxy_mean": float(np.mean([row["info_proxy_mean"] for row in rows])),
        "theta_sensitivity_sq_mean": float(np.mean([row["theta_sensitivity_sq_mean"] for row in rows])),
        "failure_rate": float(np.mean(failures)),
    }


def _rollout(
    family: search.Family,
    *,
    theta: float,
    seed: int,
    controller: str,
    dither_std: float = 0.0,
) -> dict:
    process_rng = np.random.default_rng(seed)
    dither_rng = np.random.default_rng(seed + 1_000_003)
    a_true, b_true = search.system_at_theta(family, theta)
    x = process_rng.uniform(INITIAL_STATE_LOW, INITIAL_STATE_HIGH, size=2)
    noises = process_rng.normal(0.0, PROCESS_NOISE_STD, size=(HORIZON, 2))
    window: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    total_cost = 0.0
    theta_errors_sq: list[float] = []
    info_values: list[float] = []
    sensitivity_values: list[float] = []
    held_theta = 0.0
    held_k = family.K_nom
    oracle_k = search.lqr_gain(a_true, b_true, family.Q, family.R)[0]
    failed = False

    for step in range(HORIZON):
        if controller == "oracle_lqr":
            k = oracle_k
        elif controller == "nominal_lqr":
            k = family.K_nom
        elif controller == "ce_gls_dither":
            if step == 0 or step % ID_UPDATE_INTERVAL == 0:
                theta_hat, _ = search.gls_theta_posterior(
                    family,
                    window,
                    prior_mean=0.0,
                    prior_var=THETA_MAX**2,
                )
                held_theta = float(np.clip(theta_hat, -THETA_MAX, THETA_MAX))
                a_hat, b_hat = search.system_at_theta(family, held_theta)
                held_k, _ = search.lqr_gain(a_hat, b_hat, family.Q, family.R)
            k = held_k
        else:
            raise ValueError(f"Unknown controller: {controller}")

        u = -(k @ x)
        if controller == "ce_gls_dither" and dither_std > 0.0:
            u = u.copy()
            u[1] += dither_rng.normal(0.0, dither_std)
        u = np.clip(u, ACTION_LOW, ACTION_HIGH)

        if np.linalg.norm(x) > search.MAX_ROLLOUT_STATE_NORM:
            failed = True
            break

        mismatch = family.K_nom @ x + u
        info_values.append(float(mismatch @ mismatch))
        sensitivity = family.DeltaB @ mismatch
        sensitivity_values.append(float(sensitivity @ sensitivity))
        total_cost += search.physical_stage_cost(x, u, family.Q, family.R)

        x_next = a_true @ x + b_true @ u + noises[step]
        window.append((x.copy(), u.copy(), x_next.copy()))
        if len(window) > WINDOW_LENGTH:
            window.pop(0)
        if controller == "ce_gls_dither" and step >= WINDOW_LENGTH:
            theta_errors_sq.append((held_theta - theta) ** 2)
        x = x_next

    if failed:
        total_cost += 1e6
    return {
        "theta": float(theta),
        "seed": int(seed),
        "controller": controller,
        "dither_std": float(dither_std),
        "return": -float(total_cost),
        "theta_rmse": float(np.mean(theta_errors_sq)) if theta_errors_sq else float(theta**2),
        "info_proxy_mean": float(np.mean(info_values)) if info_values else float("nan"),
        "theta_sensitivity_sq_mean": float(np.mean(sensitivity_values)) if sensitivity_values else float("nan"),
        "failed": bool(failed),
    }


def _evaluate(
    family: search.Family,
    controller: str,
    *,
    dither_std: float = 0.0,
) -> tuple[dict, list[dict]]:
    theta_grid = np.linspace(-THETA_MAX, THETA_MAX, THETA_POINTS)
    rows = []
    for theta in theta_grid:
        for seed in range(SEED_COUNT):
            rows.append(
                _rollout(
                    family,
                    theta=float(theta),
                    seed=seed,
                    controller=controller,
                    dither_std=dither_std,
                )
            )
    return _summary(rows), rows


def _static_metrics(family: search.Family) -> dict:
    theta_grid = np.linspace(-THETA_MAX, THETA_MAX, THETA_POINTS)
    ranks = []
    controllability_conds = []
    closed_loop_rhos = []
    gain_differences = []
    for theta in theta_grid:
        a, b = search.system_at_theta(family, float(theta))
        controllability = search.controllability_matrix(a, b)
        ranks.append(int(np.linalg.matrix_rank(controllability)))
        controllability_conds.append(float(np.linalg.cond(controllability)))
        k, _ = search.lqr_gain(a, b, family.Q, family.R)
        closed_loop_rhos.append(float(np.max(np.abs(np.linalg.eigvals(a - b @ k)))))
        gain_differences.append(float(np.linalg.norm(k - family.K_nom)))

    column_ratios = np.sum(family.DeltaB * family.DeltaB, axis=0) / np.diag(family.R)
    generalized_eigs = np.linalg.eigvals(np.linalg.solve(family.R, family.DeltaB.T @ family.DeltaB)).real
    return {
        "full_controllability": bool(min(ranks) == 2),
        "max_controllability_cond": float(max(controllability_conds)),
        "max_oracle_closed_loop_rho": float(max(closed_loop_rhos)),
        "mean_oracle_gain_variation": float(np.mean(gain_differences)),
        "info_cost_ratio_action_0": float(column_ratios[0]),
        "info_cost_ratio_action_1": float(column_ratios[1]),
        "max_generalized_info_cost_eig": float(max(generalized_eigs)),
    }


def _plot_returns(output: Path, rows: list[dict], title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = sorted({(row["controller"], row["dither_std"]) for row in rows})
    for controller, dither_std in labels:
        subset = [row for row in rows if row["controller"] == controller and row["dither_std"] == dither_std]
        theta_values = sorted({row["theta"] for row in subset})
        means = [np.mean([row["return"] for row in subset if row["theta"] == theta]) for theta in theta_values]
        label = controller if controller != "ce_gls_dither" else f"CE GLS + dither {dither_std:g}"
        ax.plot(theta_values, means, linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("theta")
    ax.set_ylabel("quadratic return")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / filename, dpi=180)
    plt.close(fig)


def main() -> None:
    timestamp = datetime.now().strftime("%m-%d__%H-%M")
    output = OUTPUT_ROOT / f"screening_{timestamp}"
    output.mkdir(parents=True, exist_ok=True)

    analytic_rows: list[dict] = []
    analytic_rollouts: list[dict] = []
    families: dict[str, search.Family] = {}
    systems: dict[str, dict] = {}

    for r22 in tqdm(R22_VALUES, desc="analytic systems"):
        family = _family(r22)
        label = _label(r22)
        families[label] = family
        systems[label] = _system_dict(r22, family)
        static = _static_metrics(family)
        nominal, nominal_rows = _evaluate(family, "nominal_lqr")
        oracle, oracle_rows = _evaluate(family, "oracle_lqr")
        for row in nominal_rows + oracle_rows:
            analytic_rollouts.append({"system": label, **row})

        mean_gap = oracle["mean_return"] - nominal["mean_return"]
        tail_gap = oracle["tail_return"] - nominal["tail_return"]
        tail_asymmetry = abs(
            oracle["negative_tail_return"] - oracle["positive_tail_return"]
        )
        stable = (
            static["full_controllability"]
            and static["max_controllability_cond"] <= 1e6
            and static["max_oracle_closed_loop_rho"] < 0.995
            and oracle["failure_rate"] == 0.0
            and nominal["failure_rate"] == 0.0
        )
        moderate_info_cost_ratio = (
            0.4 <= static["max_generalized_info_cost_eig"] <= 2.5
        )
        nominal_horizontal_error = abs(
            nominal["negative_tail_return"] - nominal["positive_tail_return"]
        )
        analytic_score = (
            min(mean_gap, tail_gap)
            - 0.25 * tail_asymmetry
            - 0.5 * abs(np.log(max(static["max_generalized_info_cost_eig"], 1e-12)))
        )
        analytic_rows.append(
            {
                "system": label,
                "r22": float(r22),
                **static,
                "nominal_mean_return": nominal["mean_return"],
                "nominal_tail_return": nominal["tail_return"],
                "nominal_horizontal_error": nominal_horizontal_error,
                "oracle_mean_return": oracle["mean_return"],
                "oracle_tail_return": oracle["tail_return"],
                "oracle_center_return": oracle["center_return"],
                "oracle_negative_tail_return": oracle["negative_tail_return"],
                "oracle_positive_tail_return": oracle["positive_tail_return"],
                "mean_oracle_gap": mean_gap,
                "tail_oracle_gap": tail_gap,
                "oracle_tail_asymmetry": tail_asymmetry,
                "stable": stable,
                "moderate_info_cost_ratio": moderate_info_cost_ratio,
                "analytic_score": analytic_score,
            }
        )

    eligible = [
        row
        for row in analytic_rows
        if row["stable"]
        and row["moderate_info_cost_ratio"]
        and row["nominal_horizontal_error"] < 1e-5
        and row["mean_oracle_gap"] >= 3.0
        and row["tail_oracle_gap"] >= 3.0
    ]
    if len(eligible) < 3:
        raise RuntimeError(f"Only {len(eligible)} systems passed analytic screening.")
    shortlist = sorted(eligible, key=lambda row: row["analytic_score"], reverse=True)[:3]

    dither_rows: list[dict] = []
    dither_rollouts: list[dict] = []
    dither_selection_rows: list[dict] = []
    analytic_by_label = {row["system"]: row for row in analytic_rows}
    for candidate in tqdm(shortlist, desc="dither systems"):
        label = candidate["system"]
        family = families[label]
        summaries: dict[float, dict] = {}
        for dither_std in DITHER_STDS:
            summary, rows = _evaluate(family, "ce_gls_dither", dither_std=dither_std)
            summaries[dither_std] = summary
            dither_rows.append({"system": label, "dither_std": dither_std, **summary})
            for row in rows:
                dither_rollouts.append({"system": label, **row})

        zero = summaries[0.0]
        gap = max(analytic_by_label[label]["mean_oracle_gap"], 1e-8)
        moderate = []
        for dither_std in (0.02, 0.05, 0.1):
            summary = summaries[dither_std]
            id_gain_fraction = (zero["theta_rmse"] - summary["theta_rmse"]) / max(
                zero["theta_rmse"], 1e-8
            )
            return_cost_fraction = max(0.0, zero["mean_return"] - summary["mean_return"]) / gap
            score = id_gain_fraction - 0.5 * return_cost_fraction
            moderate.append((score, dither_std, id_gain_fraction, return_cost_fraction, summary))
        best_score, best_std, id_gain_fraction, return_cost_fraction, best_summary = max(moderate)
        dither_gap_closure = (
            best_summary["mean_return"] - analytic_by_label[label]["nominal_mean_return"]
        ) / gap
        dither_selection_rows.append(
            {
                "system": label,
                "r22": candidate["r22"],
                "analytic_score": candidate["analytic_score"],
                "zero_dither_theta_rmse": zero["theta_rmse"],
                "best_moderate_dither_std": best_std,
                "best_moderate_theta_rmse": best_summary["theta_rmse"],
                "best_moderate_mean_return": best_summary["mean_return"],
                "identification_gain_fraction": id_gain_fraction,
                "return_cost_fraction_of_oracle_gap": return_cost_fraction,
                "dither_oracle_gap_closure": dither_gap_closure,
                "dither_score": best_score,
                "dither_failure_rate": best_summary["failure_rate"],
            }
        )

    feasible = [
        row
        for row in dither_selection_rows
        if row["zero_dither_theta_rmse"] >= 0.1
        and row["identification_gain_fraction"] >= 0.2
        and row["return_cost_fraction_of_oracle_gap"] <= 1.0
        and row["dither_oracle_gap_closure"] <= 0.75
        and row["dither_failure_rate"] == 0.0
    ]
    pool = feasible if len(feasible) >= 2 else dither_selection_rows
    selected = sorted(pool, key=lambda row: row["dither_score"], reverse=True)[:2]

    _write_csv(output / "analytic_scorecard.csv", analytic_rows)
    _write_csv(output / "analytic_rollouts.csv", analytic_rollouts)
    _write_csv(output / "dither_scorecard.csv", dither_rows)
    _write_csv(output / "dither_rollouts.csv", dither_rollouts)
    _write_csv(output / "dither_selection.csv", dither_selection_rows)
    _plot_returns(output, analytic_rollouts, "Nominal and Oracle LQR", "analytic_return_vs_theta.png")
    _plot_returns(output, dither_rollouts, "CE GLS With Action-2 Dither", "dither_return_vs_theta.png")

    payload = {
        "timestamp": timestamp,
        "protocol": {
            "theta_max": THETA_MAX,
            "theta_points": THETA_POINTS,
            "seed_count": SEED_COUNT,
            "horizon": HORIZON,
            "window_length": WINDOW_LENGTH,
            "nominal_warmup_steps": NOMINAL_WARMUP_STEPS,
            "id_update_interval": ID_UPDATE_INTERVAL,
            "process_noise_std": PROCESS_NOISE_STD,
        },
        "all_systems": [systems[_label(r22)] for r22 in R22_VALUES],
        "analytic_shortlist": [systems[row["system"]] for row in shortlist],
        "selected_for_neural_screening": [systems[row["system"]] for row in selected],
    }
    with open(output / "selected_systems.json", "w") as f:
        json.dump(payload, f, indent=2)
    latest = OUTPUT_ROOT / "latest_screening.txt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(output.resolve()) + "\n")

    print(f"Saved system screening to: {output}", flush=True)
    print("Analytic shortlist:", ", ".join(row["system"] for row in shortlist), flush=True)
    print("Selected for neural screening:", ", ".join(row["system"] for row in selected), flush=True)


if __name__ == "__main__":
    main()
