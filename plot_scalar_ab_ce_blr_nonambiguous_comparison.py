from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

from run_scalar_ab_lqr_diagnostics import _simulate_ce_blr
from scalar_ab_lqr_utils import (
    SELECTION_ROOT,
    discounted_cost_for_gain,
    write_csv,
)


TOL_CLOSED_LOOP = 0.005
MAX_NEIGHBORS = 80
N_ROLLOUT_SEEDS = 20


def _latest_diagnostics_dir() -> Path:
    pointer = SELECTION_ROOT / "latest_scalar_ab_diagnostics.txt"
    if not pointer.exists():
        raise FileNotFoundError(f"Missing diagnostics pointer: {pointer}")
    return Path(pointer.read_text().strip())


def _nearest_grid_row(analytic: pd.DataFrame, a: float, b: float) -> pd.Series:
    idx = ((analytic["a"] - a) ** 2 + (analytic["b"] - b) ** 2).idxmin()
    return analytic.loc[idx]


def _ambiguity_scores(analytic: pd.DataFrame) -> pd.DataFrame:
    rows = analytic.sort_values("closed_loop").reset_index(drop=True).copy()
    score = np.zeros(len(rows), dtype=np.float64)
    close_count = np.zeros(len(rows), dtype=np.int64)
    for i in range(len(rows)):
        row = rows.iloc[i]
        upper = min(len(rows), i + MAX_NEIGHBORS + 1)
        for j in range(i + 1, upper):
            other = rows.iloc[j]
            dc = float(other["closed_loop"] - row["closed_loop"])
            if dc > TOL_CLOSED_LOOP:
                break
            excess_i = discounted_cost_for_gain(float(row["a"]), float(row["b"]), float(other["K"])) - float(row["optimal_cost_x1"])
            excess_j = discounted_cost_for_gain(float(other["a"]), float(other["b"]), float(row["K"])) - float(other["optimal_cost_x1"])
            pair_score = max(excess_i, excess_j)
            score[i] = max(score[i], pair_score)
            score[j] = max(score[j], pair_score)
            close_count[i] += 1
            close_count[j] += 1
    rows["ambiguity_score"] = score
    rows["close_signature_neighbor_count"] = close_count
    return rows.sort_values(["a", "b"]).reset_index(drop=True)


def _farthest_subset(candidates: pd.DataFrame, n: int, min_dist: float = 0.10) -> pd.DataFrame:
    selected = []
    for _, row in candidates.sort_values(["ambiguity_score", "close_signature_neighbor_count"]).iterrows():
        point = np.asarray([float(row["a"]), float(row["b"])], dtype=np.float64)
        if all(np.linalg.norm(point - np.asarray([float(sel["a"]), float(sel["b"])], dtype=np.float64)) >= min_dist for sel in selected):
            selected.append(row.to_dict())
        if len(selected) >= n:
            break
    return pd.DataFrame(selected)


def _selected_nonambiguous_points(scored: pd.DataFrame, analytic: pd.DataFrame) -> pd.DataFrame:
    a_min, a_max = float(analytic["a"].min()), float(analytic["a"].max())
    b_min, b_max = float(analytic["b"].min()), float(analytic["b"].max())

    interior = scored[
        (scored["a"] > a_min + 0.04)
        & (scored["a"] < a_max - 0.04)
        & (scored["b"] > b_min + 0.15)
        & (scored["b"] < b_max - 0.15)
    ]
    easy = _farthest_subset(interior, n=6, min_dist=0.12)
    easy["category"] = "easy_interior"
    easy["name"] = [f"easy_{i+1}" for i in range(len(easy))]

    high_b_pool = scored[
        (scored["a"] > a_min + 0.04)
        & (scored["a"] < a_max - 0.04)
        & (scored["b"] > b_max - 0.25)
    ]
    high_b = _farthest_subset(high_b_pool, n=3, min_dist=0.10)
    high_b["category"] = "high_b_easy"
    high_b["name"] = [f"high_b_{i+1}" for i in range(len(high_b))]

    target_specs = [
        ("center", (a_min + a_max) / 2.0, (b_min + b_max) / 2.0),
        ("low_a_low_b_corner", a_min, b_min),
        ("low_a_high_b_corner", a_min, b_max),
        ("high_a_low_b_corner", a_max, b_min),
        ("high_a_high_b_corner", a_max, b_max),
    ]
    representative_rows = []
    for name, a, b in target_specs:
        row = _nearest_grid_row(scored, a, b).to_dict()
        row["category"] = "representative"
        row["name"] = name
        representative_rows.append(row)
    representative = pd.DataFrame(representative_rows)

    selected = pd.concat([easy, high_b, representative], ignore_index=True)
    selected = selected.drop_duplicates(subset=["a", "b"]).reset_index(drop=True)
    return selected


def _rollout_selected(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in selected.iterrows():
        for seed in range(N_ROLLOUT_SEEDS):
            out = _simulate_ce_blr(float(row["a"]), float(row["b"]), 22345 + seed)
            out["category"] = row["category"]
            out["name"] = row["name"]
            out["ambiguity_score"] = float(row["ambiguity_score"])
            out["close_signature_neighbor_count"] = int(row["close_signature_neighbor_count"])
            rows.append(out)
    return pd.DataFrame(rows)


def _summarize_rollouts(rows: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "return",
        "final_mean_a",
        "final_mean_b",
        "final_std_a",
        "final_std_b",
        "final_corr",
        "gain_error",
        "gram_lambda_min",
        "gram_condition",
        "cov_lambda_min",
        "cov_lambda_max",
        "ambiguity_score",
        "close_signature_neighbor_count",
    ]
    group_cols = ["category", "name", "a", "b"]
    available_numeric = [col for col in numeric_cols if col in rows.columns]
    summary = rows.groupby(group_cols, dropna=False)[available_numeric].mean().reset_index()
    metrics = []
    for _, row in summary.iterrows():
        mu = np.asarray([row["final_mean_a"], row["final_mean_b"]], dtype=np.float64)
        true = np.asarray([row["a"], row["b"]], dtype=np.float64)
        sa, sb, corr = float(row["final_std_a"]), float(row["final_std_b"]), float(row["final_corr"])
        cov = np.asarray([[sa * sa, corr * sa * sb], [corr * sa * sb, sb * sb]], dtype=np.float64)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-12)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        diff = true - mu
        mahal = float(np.sqrt(diff @ np.linalg.inv(cov) @ diff))
        coords = eigvecs.T @ diff / np.sqrt(eigvals)
        angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
        metrics.append(
            {
                "mean_error": float(np.linalg.norm(diff)),
                "mean_error_a": float(diff[0]),
                "mean_error_b": float(diff[1]),
                "mahalanobis_to_true": mahal,
                "inside_2std": bool(mahal <= 2.0),
                "ellipse_area_2std": float(4.0 * np.pi * np.sqrt(np.linalg.det(cov))),
                "cov_condition": float(eigvals[0] / max(eigvals[1], 1e-12)),
                "major_axis_angle_deg": angle,
                "true_coord_major_std": float(coords[0]),
                "true_coord_minor_std": float(coords[1]),
            }
        )
    return pd.concat([summary.reset_index(drop=True), pd.DataFrame(metrics)], axis=1)


def _ellipse_for_row(row: pd.Series, *, radius: float = 2.0, **kwargs) -> Ellipse:
    sa, sb, corr = float(row["final_std_a"]), float(row["final_std_b"]), float(row["final_corr"])
    cov = np.asarray([[sa * sa, corr * sa * sb], [corr * sa * sb, sb * sb]], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1e-12)
    eigvecs = eigvecs[:, order]
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    return Ellipse(
        xy=(float(row["final_mean_a"]), float(row["final_mean_b"])),
        width=2.0 * radius * np.sqrt(eigvals[0]),
        height=2.0 * radius * np.sqrt(eigvals[1]),
        angle=angle,
        fill=False,
        **kwargs,
    )


def _plot_ambiguity_score_map(scored: pd.DataFrame, selected: pd.DataFrame, out: Path) -> None:
    a_vals = np.sort(scored["a"].unique())
    b_vals = np.sort(scored["b"].unique())
    grid = scored.pivot(index="b", columns="a", values="ambiguity_score").sort_index().to_numpy()
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[a_vals.min(), a_vals.max(), b_vals.min(), b_vals.max()],
    )
    fig.colorbar(im, ax=ax, label="max wrong-controller excess among close-signature neighbors")
    colors = _category_colors()
    for cat, subset in selected.groupby("category"):
        ax.scatter(subset["a"], subset["b"], marker="x", s=70, linewidths=2.0, label=cat, color=colors.get(cat, "k"))
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_title("Scalar (a,b) ambiguity score and selected non-ambiguous comparison points")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "scalar_ab_ambiguity_score_selected_points.png", dpi=180)
    plt.close(fig)


def _category_colors() -> dict[str, str]:
    return {
        "ambiguous_endpoint": "crimson",
        "easy_interior": "tab:blue",
        "high_b_easy": "tab:purple",
        "representative": "tab:green",
    }


def _plot_ellipses(summary: pd.DataFrame, out_path: Path, title: str, categories: list[str] | None = None) -> None:
    plot_df = summary if categories is None else summary[summary["category"].isin(categories)]
    colors = _category_colors()
    fig, ax = plt.subplots(figsize=(10, 8))
    for cat, subset in plot_df.groupby("category"):
        color = colors.get(cat, "k")
        for _, row in subset.iterrows():
            ax.add_patch(_ellipse_for_row(row, edgecolor=color, linewidth=1.8, alpha=0.75))
            ax.scatter(row["a"], row["b"], marker="x", s=55, color=color, linewidths=2.0)
            ax.scatter(row["final_mean_a"], row["final_mean_b"], marker="o", s=35, facecolors="none", edgecolors=color, linewidths=1.5)
            ax.plot([row["a"], row["final_mean_a"]], [row["b"], row["final_mean_b"]], color=color, alpha=0.35, linewidth=1.0)
        ax.scatter([], [], marker="x", color=color, label=f"{cat}: true x")
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_title(title + "\n'x'=true parameter, 'o'=posterior mean, ellipse=2 std posterior")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_metric_summary(summary: pd.DataFrame, out: Path) -> None:
    metrics = [
        ("mean_error", "posterior mean error"),
        ("mahalanobis_to_true", "true-vs-mean Mahalanobis distance"),
        ("ellipse_area_2std", "2-std ellipse area"),
        ("cov_condition", "posterior covariance condition"),
        ("gram_lambda_min", "Gram lambda_min"),
        ("final_corr", "posterior corr(a,b)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = _category_colors()
    categories = ["ambiguous_endpoint", "easy_interior", "high_b_easy", "representative"]
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        data = [summary.loc[summary["category"] == cat, metric].dropna().to_numpy() for cat in categories]
        bp = ax.boxplot(data, tick_labels=categories, patch_artist=True, showmeans=True)
        for patch, cat in zip(bp["boxes"], categories):
            patch.set_facecolor(colors.get(cat, "gray"))
            patch.set_alpha(0.35)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=25)
        if metric in {"ellipse_area_2std", "cov_condition", "gram_lambda_min"}:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("CE-BLR posterior geometry: ambiguous endpoints vs non-ambiguous comparison systems")
    fig.tight_layout()
    fig.savefig(out / "scalar_ab_ce_blr_posterior_metric_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-dir", type=Path, default=None)
    args = parser.parse_args()
    diagnostics_dir = args.diagnostics_dir or _latest_diagnostics_dir()
    out = diagnostics_dir / "nonambiguous_ce_blr_comparison"
    out.mkdir(parents=True, exist_ok=True)

    analytic = pd.read_csv(diagnostics_dir / "analytic_grid.csv")
    scored = _ambiguity_scores(analytic)
    selected = _selected_nonambiguous_points(scored, analytic)
    write_csv(out / "selected_nonambiguous_points.csv", selected.to_dict("records"))
    _plot_ambiguity_score_map(scored, selected, out)

    nonamb_rollouts = _rollout_selected(selected)
    nonamb_rollouts.to_csv(out / "nonambiguous_blr_ce_lqr_rollouts.csv", index=False)

    ambiguous = pd.read_csv(diagnostics_dir / "blr_ce_lqr_rollouts.csv")
    ambiguous = ambiguous.copy()
    ambiguous["category"] = "ambiguous_endpoint"
    ambiguous["name"] = [f"ambiguous_{a:.4f}_{b:.4f}" for a, b in zip(ambiguous["a"], ambiguous["b"])]
    # Drop repeated endpoint memberships before summary by letting the groupby average over all available seeds.
    combined = pd.concat([ambiguous, nonamb_rollouts], ignore_index=True, sort=False)
    summary = _summarize_rollouts(combined)
    summary.to_csv(out / "ce_blr_posterior_comparison_summary.csv", index=False)

    _plot_ellipses(
        summary,
        out / "scalar_ab_ce_blr_ellipses_ambiguous_vs_easy.png",
        "CE-BLR posterior ellipses: ambiguous endpoints vs easy interior systems",
        ["ambiguous_endpoint", "easy_interior"],
    )
    _plot_ellipses(
        summary,
        out / "scalar_ab_ce_blr_ellipses_representative.png",
        "CE-BLR posterior ellipses: representative corners/center and high-b points",
        ["representative", "high_b_easy"],
    )
    _plot_ellipses(
        summary,
        out / "scalar_ab_ce_blr_ellipses_all_selected.png",
        "CE-BLR posterior ellipses: all selected systems",
        None,
    )
    _plot_metric_summary(summary, out)

    print(f"Saved CE-BLR non-ambiguous comparison to: {out}", flush=True)


if __name__ == "__main__":
    main()
