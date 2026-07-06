from __future__ import annotations

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scalar_ab_lqr_utils import (
    DEFAULT_RANGE,
    EPISODE_STEPS,
    FALLBACK_RANGE,
    INITIAL_STATE_HIGH,
    INITIAL_STATE_LOW,
    PRIOR_MEAN,
    PRIOR_VAR,
    PROCESS_NOISE_STD,
    Q,
    R,
    SELECTION_ROOT,
    WINDOW_LENGTH,
    closed_loop_signature,
    discounted_cost_for_gain,
    save_json,
    scalar_gain,
    smoke_enabled,
    timestamp,
    write_csv,
)


def _grid_n() -> int:
    return 41 if smoke_enabled("SCALAR_AB_DIAGNOSTICS_SMOKE") else 201


def _rollout_n() -> int:
    return 2 if smoke_enabled("SCALAR_AB_DIAGNOSTICS_SMOKE") else 20


def _posterior_from_window(z_rows: list[list[float]], y_rows: list[float]) -> tuple[np.ndarray, np.ndarray]:
    mu0 = np.asarray(PRIOR_MEAN, dtype=np.float64)
    var0 = np.asarray(PRIOR_VAR, dtype=np.float64)
    precision = np.diag(1.0 / var0)
    rhs = precision @ mu0
    if z_rows:
        z = np.asarray(z_rows[-WINDOW_LENGTH:], dtype=np.float64)
        y = np.asarray(y_rows[-WINDOW_LENGTH:], dtype=np.float64)
        inv_var = 1.0 / max(PROCESS_NOISE_STD**2, 1e-10)
        precision = precision + inv_var * (z.T @ z)
        rhs = rhs + inv_var * (z.T @ y)
    cov = np.linalg.inv(precision)
    mean = np.linalg.solve(precision, rhs)
    return mean, cov


def _simulate_ce_blr(a: float, b: float, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x = float(rng.uniform(INITIAL_STATE_LOW, INITIAL_STATE_HIGH))
    z_rows: list[list[float]] = []
    y_rows: list[float] = []
    total_cost = 0.0
    for _ in range(EPISODE_STEPS):
        mean, _ = _posterior_from_window(z_rows, y_rows)
        k = scalar_gain(float(mean[0]), float(mean[1]))
        u = -k * x
        total_cost += Q * x * x + R * u * u
        noise = float(rng.normal(0.0, PROCESS_NOISE_STD))
        x_next = a * x + b * u + noise
        z_rows.append([x, u])
        y_rows.append(x_next)
        x = x_next
    mean, cov = _posterior_from_window(z_rows, y_rows)
    gram = np.asarray(z_rows, dtype=np.float64).T @ np.asarray(z_rows, dtype=np.float64)
    eig = np.linalg.eigvalsh(gram + 1e-12 * np.eye(2))
    cov_eig = np.linalg.eigvalsh(cov)
    k_true = scalar_gain(a, b)
    k_hat = scalar_gain(float(mean[0]), float(mean[1]))
    return {
        "a": float(a),
        "b": float(b),
        "seed": int(seed),
        "return": float(-total_cost),
        "final_mean_a": float(mean[0]),
        "final_mean_b": float(mean[1]),
        "final_std_a": float(np.sqrt(max(cov[0, 0], 0.0))),
        "final_std_b": float(np.sqrt(max(cov[1, 1], 0.0))),
        "final_corr": float(cov[0, 1] / np.sqrt(max(cov[0, 0] * cov[1, 1], 1e-12))),
        "gain_error": float(abs(k_hat - k_true)),
        "gram_lambda_min": float(eig[0]),
        "gram_condition": float(eig[-1] / max(eig[0], 1e-12)),
        "cov_lambda_min": float(cov_eig[0]),
        "cov_lambda_max": float(cov_eig[-1]),
    }


def _analytic_rows(param_range: dict[str, list[float]], n: int) -> list[dict[str, float]]:
    rows = []
    for a in np.linspace(param_range["a"][0], param_range["a"][1], n):
        for b in np.linspace(param_range["b"][0], param_range["b"][1], n):
            k = scalar_gain(float(a), float(b))
            rows.append(
                {
                    "a": float(a),
                    "b": float(b),
                    "K": float(k),
                    "closed_loop": closed_loop_signature(float(a), float(b), k),
                    "optimal_cost_x1": discounted_cost_for_gain(float(a), float(b), k),
                }
            )
    return rows


def _ambiguous_pairs(rows: list[dict[str, float]], max_pairs: int = 200) -> list[dict[str, float]]:
    sorted_rows = sorted(rows, key=lambda row: row["closed_loop"])
    pairs = []
    for i, row in enumerate(sorted_rows):
        for other in sorted_rows[i + 1 : i + 80]:
            dc = abs(other["closed_loop"] - row["closed_loop"])
            if dc > 0.005:
                break
            excess_1 = discounted_cost_for_gain(row["a"], row["b"], other["K"]) - row["optimal_cost_x1"]
            excess_2 = discounted_cost_for_gain(other["a"], other["b"], row["K"]) - other["optimal_cost_x1"]
            pairs.append(
                {
                    "a1": row["a"],
                    "b1": row["b"],
                    "K1": row["K"],
                    "c1": row["closed_loop"],
                    "a2": other["a"],
                    "b2": other["b"],
                    "K2": other["K"],
                    "c2": other["closed_loop"],
                    "delta_closed_loop": dc,
                    "delta_K": abs(other["K"] - row["K"]),
                    "max_excess_cost": max(excess_1, excess_2),
                    "excess_cost_1_wrong_K2": excess_1,
                    "excess_cost_2_wrong_K1": excess_2,
                }
            )
    return sorted(pairs, key=lambda row: row["max_excess_cost"], reverse=True)[:max_pairs]


def _plot_analytic(rows: list[dict[str, float]], out: Path, metric: str) -> None:
    a_vals = sorted({row["a"] for row in rows})
    b_vals = sorted({row["b"] for row in rows})
    grid = np.full((len(a_vals), len(b_vals)), np.nan)
    a_idx = {v: i for i, v in enumerate(a_vals)}
    b_idx = {v: i for i, v in enumerate(b_vals)}
    for row in rows:
        grid[a_idx[row["a"]], b_idx[row["b"]]] = row[metric]
    plt.figure(figsize=(8, 6))
    plt.imshow(grid, origin="lower", aspect="auto", extent=[min(b_vals), max(b_vals), min(a_vals), max(a_vals)])
    plt.colorbar(label=metric)
    plt.xlabel("b")
    plt.ylabel("a")
    plt.title(metric)
    plt.tight_layout()
    plt.savefig(out / f"{metric}_heatmap.png", dpi=180)
    plt.close()


def main() -> None:
    smoke = smoke_enabled("SCALAR_AB_DIAGNOSTICS_SMOKE")
    root = SELECTION_ROOT / f"diagnostics_{timestamp()}_stable_wide_b{'_smoke' if smoke else ''}"
    root.mkdir(parents=True, exist_ok=True)
    param_range = DEFAULT_RANGE
    analytic = _analytic_rows(param_range, _grid_n())
    pairs = _ambiguous_pairs(analytic)
    write_csv(root / "analytic_grid.csv", analytic)
    write_csv(root / "ambiguous_pairs.csv", pairs)
    for metric in ["K", "closed_loop", "optimal_cost_x1"]:
        _plot_analytic(analytic, root, metric)

    rollout_rows = []
    eval_points = [(row["a1"], row["b1"]) for row in pairs[:10]] + [(row["a2"], row["b2"]) for row in pairs[:10]]
    if not eval_points:
        eval_points = [
            (param_range["a"][0], param_range["b"][0]),
            (np.mean(param_range["a"]), np.mean(param_range["b"])),
            (param_range["a"][1], param_range["b"][1]),
        ]
    for a, b in eval_points:
        for seed in range(_rollout_n()):
            rollout_rows.append(_simulate_ce_blr(float(a), float(b), 12345 + seed))
    write_csv(root / "blr_ce_lqr_rollouts.csv", rollout_rows)
    save_json(
        root / "diagnostics_manifest.json",
        {
            "param_range": param_range,
            "fallback_range": FALLBACK_RANGE,
            "grid_n": _grid_n(),
            "rollout_seeds": _rollout_n(),
            "smoke": smoke,
            "outputs": ["analytic_grid.csv", "ambiguous_pairs.csv", "blr_ce_lqr_rollouts.csv"],
        },
    )
    if not smoke:
        (SELECTION_ROOT / "latest_scalar_ab_diagnostics.txt").write_text(str(root.resolve()) + "\n")
    print(f"Saved scalar-ab diagnostics to: {root}", flush=True)


if __name__ == "__main__":
    main()
