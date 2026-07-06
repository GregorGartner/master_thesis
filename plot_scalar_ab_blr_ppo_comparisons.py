#!/usr/bin/env python3
"""Comparison plots for scalar (a,b) BLR/PPO and neural RMA runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONTROLLER_LABELS = {
    "privileged": "Privileged PPO",
    "blr_mean_P06": "BLR mean P06",
    "blr_mean_cov_P06": "BLR mean+cov P06",
    "staged_vanilla_S02": "Staged vanilla S02",
    "gradual_mle_P06": "Gradual MLE P06",
    "gradual_nll_P06": "Gradual NLL P06",
    "oracle_lqr": "Oracle LQR",
}

CONTROLLER_ORDER = [
    "privileged",
    "staged_vanilla_S02",
    "gradual_mle_P06",
    "gradual_nll_P06",
    "blr_mean_P06",
    "blr_mean_cov_P06",
    "oracle_lqr",
]

CONTROLLER_COLORS = {
    "privileged": "0.45",
    "staged_vanilla_S02": "tab:purple",
    "gradual_mle_P06": "tab:blue",
    "gradual_nll_P06": "tab:green",
    "blr_mean_P06": "tab:blue",
    "blr_mean_cov_P06": "tab:green",
    "oracle_lqr": "tab:orange",
}


def _pivot(df: pd.DataFrame, controller: str, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = df[df["controller"] == controller]
    a_vals = np.sort(sub["a"].unique())
    b_vals = np.sort(sub["b"].unique())
    grid = sub.pivot(index="b", columns="a", values=metric).loc[b_vals, a_vals].values
    return a_vals, b_vals, grid


def _metric_values(df: pd.DataFrame, metric: str, controllers: list[str]) -> np.ndarray:
    vals = []
    for controller in controllers:
        vals.append(df.loc[df["controller"] == controller, metric].to_numpy())
    return np.concatenate(vals)


def _robust_limits(vals: np.ndarray, *, center_zero: bool = False) -> tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    if center_zero:
        lim = float(np.nanpercentile(np.abs(vals), 98))
        if lim <= 0:
            lim = float(np.nanmax(np.abs(vals))) or 1.0
        return -lim, lim
    lo, hi = np.nanpercentile(vals, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def _add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log10_gram_lambda_min_mean"] = np.log10(np.maximum(df["gram_lambda_min_mean"], 1e-12))
    df["log10_u_over_x_var_mean"] = np.log10(np.maximum(df["u_over_x_var_mean"], 1e-12))
    return df


def plot_shared_scale_grid(df: pd.DataFrame, out: Path, tag: str) -> None:
    controllers = [c for c in CONTROLLER_ORDER if c in set(df["controller"])]
    metrics = [
        ("return_mean", "Mean return", "viridis"),
        ("state_cost_mean", "State cost", "magma_r"),
        ("action_cost_mean", "Action cost", "magma_r"),
        ("gram_logdet_mean", "Gram logdet", "viridis"),
        ("log10_u_over_x_var_mean", "log10 Var(u/x)", "viridis"),
    ]
    fig, axs = plt.subplots(len(metrics), len(controllers), figsize=(4.2 * len(controllers), 3.4 * len(metrics)), constrained_layout=True)
    if len(metrics) == 1:
        axs = np.array([axs])
    if len(controllers) == 1:
        axs = axs[:, None]

    for row, (metric, label, cmap) in enumerate(metrics):
        vals = _metric_values(df, metric, controllers)
        vmin, vmax = _robust_limits(vals)
        last_im = None
        for col, controller in enumerate(controllers):
            ax = axs[row, col]
            a_vals, b_vals, grid = _pivot(df, controller, metric)
            extent = [a_vals.min(), a_vals.max(), b_vals.min(), b_vals.max()]
            last_im = ax.imshow(grid, origin="lower", extent=extent, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            if row == 0:
                ax.set_title(CONTROLLER_LABELS.get(controller, controller))
            if col == 0:
                ax.set_ylabel(f"{label}\nb")
            else:
                ax.set_yticklabels([])
            if row == len(metrics) - 1:
                ax.set_xlabel("a")
            else:
                ax.set_xticklabels([])
        fig.colorbar(last_im, ax=axs[row, :].tolist(), label=label, shrink=0.88)
    fig.suptitle(f"Scalar (a,b) Controller Comparison ({tag}) - shared color scale per row", fontsize=16)
    fig.savefig(out / f"scalar_ab_shared_scale_comparison_{tag}.png", dpi=180)
    plt.close(fig)


def _controller_metric_frame(df: pd.DataFrame, controller: str, metric: str) -> pd.DataFrame:
    return df[df["controller"] == controller][["a", "b", metric]].rename(columns={metric: controller})


def _difference_df(df: pd.DataFrame, metric: str, left: str, right: str) -> pd.DataFrame:
    ldf = _controller_metric_frame(df, left, metric)
    rdf = _controller_metric_frame(df, right, metric)
    merged = ldf.merge(rdf, on=["a", "b"], how="inner")
    merged["diff"] = merged[left] - merged[right]
    return merged


def _available_comparisons(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    candidates = [
        ("blr_mean_cov_P06", "blr_mean_P06", "mean+cov - mean"),
        ("blr_mean_P06", "privileged", "mean - privileged"),
        ("blr_mean_cov_P06", "privileged", "mean+cov - privileged"),
        ("gradual_nll_P06", "gradual_mle_P06", "NLL - MLE"),
        ("staged_vanilla_S02", "privileged", "staged - privileged"),
        ("gradual_mle_P06", "privileged", "MLE - privileged"),
        ("gradual_nll_P06", "privileged", "NLL - privileged"),
    ]
    available = set(df["controller"])
    return [(l, r, label) for l, r, label in candidates if l in available and r in available]


def plot_difference_grid(df: pd.DataFrame, out: Path, tag: str) -> None:
    comparisons = _available_comparisons(df)
    if not comparisons:
        return
    metrics = [
        ("return_mean", "Return delta"),
        ("state_cost_mean", "State-cost delta"),
        ("action_cost_mean", "Action-cost delta"),
        ("gram_logdet_mean", "Gram-logdet delta"),
        ("log10_u_over_x_var_mean", "log10 Var(u/x) delta"),
    ]
    fig, axs = plt.subplots(len(metrics), len(comparisons), figsize=(4.6 * len(comparisons), 3.3 * len(metrics)), constrained_layout=True)
    if len(metrics) == 1:
        axs = np.array([axs])
    if len(comparisons) == 1:
        axs = axs[:, None]

    for row, (metric, metric_label) in enumerate(metrics):
        all_diffs = []
        diff_frames = []
        for left, right, _ in comparisons:
            ddf = _difference_df(df, metric, left, right)
            diff_frames.append(ddf)
            all_diffs.append(ddf["diff"].to_numpy())
        vmin, vmax = _robust_limits(np.concatenate(all_diffs), center_zero=True)
        last_im = None
        for col, ((_, _, comp_label), ddf) in enumerate(zip(comparisons, diff_frames)):
            ax = axs[row, col]
            a_vals = np.sort(ddf["a"].unique())
            b_vals = np.sort(ddf["b"].unique())
            grid = ddf.pivot(index="b", columns="a", values="diff").loc[b_vals, a_vals].values
            extent = [a_vals.min(), a_vals.max(), b_vals.min(), b_vals.max()]
            last_im = ax.imshow(grid, origin="lower", extent=extent, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
            if row == 0:
                ax.set_title(comp_label)
            if col == 0:
                ax.set_ylabel(f"{metric_label}\nb")
            else:
                ax.set_yticklabels([])
            if row == len(metrics) - 1:
                ax.set_xlabel("a")
            else:
                ax.set_xticklabels([])
        fig.colorbar(last_im, ax=axs[row, :].tolist(), label=metric_label, shrink=0.88)
    fig.suptitle(f"Scalar (a,b) Controller Difference Maps ({tag})", fontsize=16)
    fig.savefig(out / f"scalar_ab_difference_maps_{tag}.png", dpi=180)
    plt.close(fig)


def _primary_return_pair(df: pd.DataFrame) -> tuple[str, str, str, str, str] | None:
    available = set(df["controller"])
    if {"blr_mean_P06", "blr_mean_cov_P06"}.issubset(available):
        return (
            "blr_mean_P06",
            "blr_mean_cov_P06",
            "BLR mean",
            "BLR mean+cov",
            "mean_cov",
        )
    if {"gradual_mle_P06", "gradual_nll_P06"}.issubset(available):
        return (
            "gradual_mle_P06",
            "gradual_nll_P06",
            "Gradual MLE",
            "Gradual NLL",
            "nll",
        )
    return None


def plot_return_scatter(df: pd.DataFrame, out: Path, tag: str) -> None:
    pair = _primary_return_pair(df)
    if pair is None:
        return
    left, right, left_label, right_label, file_tag = pair
    left_df = _controller_metric_frame(df, left, "return_mean")
    right_df = _controller_metric_frame(df, right, "return_mean")
    info = df[df["controller"] == right][["a", "b", "gram_logdet_mean"]].rename(
        columns={"gram_logdet_mean": "right_gram_logdet"}
    )
    merged = left_df.merge(right_df, on=["a", "b"]).merge(info, on=["a", "b"])
    merged["return_delta"] = merged[right] - merged[left]

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    lo = float(min(merged[left].min(), merged[right].min()))
    hi = float(max(merged[left].max(), merged[right].max()))
    pad = 0.03 * (hi - lo)
    sc = axs[0].scatter(
        merged[left],
        merged[right],
        c=merged["right_gram_logdet"],
        cmap="viridis",
        s=24,
        alpha=0.82,
    )
    axs[0].plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=1.5)
    axs[0].set_xlim(lo - pad, hi + pad)
    axs[0].set_ylim(lo - pad, hi + pad)
    axs[0].set_xlabel(f"{left_label} return")
    axs[0].set_ylabel(f"{right_label} return")
    axs[0].set_title("Return comparison by grid point")
    axs[0].grid(True, alpha=0.25)
    fig.colorbar(sc, ax=axs[0], label=f"{right_label} Gram logdet")

    axs[1].hist(merged["return_delta"], bins=40, color="tab:blue", alpha=0.82)
    axs[1].axvline(0, color="black", lw=1.5)
    axs[1].axvline(merged["return_delta"].mean(), color="crimson", lw=2, label=f"mean {merged['return_delta'].mean():.4f}")
    axs[1].set_xlabel(f"{right_label} return - {left_label} return")
    axs[1].set_ylabel("grid points")
    axs[1].set_title("Distribution of return advantage")
    axs[1].grid(True, alpha=0.25)
    axs[1].legend()
    fig.suptitle(f"Scalar (a,b) {left_label} vs {right_label} ({tag})", fontsize=15)
    fig.savefig(out / f"scalar_ab_{file_tag}_return_scatter_{tag}.png", dpi=180)
    plt.close(fig)


def _nearest_rows(df: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
    grid = df[["a", "b"]].drop_duplicates().copy()
    grid["dist2"] = (grid["a"] - a) ** 2 + (grid["b"] - b) ** 2
    nearest = grid.sort_values("dist2").iloc[0]
    return df[(df["a"] == nearest["a"]) & (df["b"] == nearest["b"])].copy()


def plot_ambiguous_pair_focus(df: pd.DataFrame, diagnostics_dir: Path | None, out: Path, tag: str) -> None:
    if diagnostics_dir is None:
        return
    pair_path = diagnostics_dir / "ambiguous_pairs.csv"
    if not pair_path.exists():
        return
    pairs = pd.read_csv(pair_path).sort_values("max_excess_cost", ascending=False).head(25)
    rows = []
    for _, pair in pairs.iterrows():
        for side in ["1", "2"]:
            near = _nearest_rows(df, float(pair[f"a{side}"]), float(pair[f"b{side}"]))
            near = near.assign(pair_rank=len(rows), source_a=float(pair[f"a{side}"]), source_b=float(pair[f"b{side}"]))
            rows.append(near)
    focus = pd.concat(rows, ignore_index=True)
    focus_path = out / f"scalar_ab_ambiguous_focus_points_{tag}.csv"
    focus.to_csv(focus_path, index=False)

    controllers = [c for c in CONTROLLER_ORDER if c in set(focus["controller"])]
    metrics = [
        ("return_mean", "Mean return", "higher better"),
        ("state_cost_mean", "State cost", "lower better"),
        ("action_cost_mean", "Action cost", "lower better"),
        ("gram_logdet_mean", "Gram logdet", "higher means richer regressors"),
    ]
    summary = focus.groupby("controller")[["return_mean", "state_cost_mean", "action_cost_mean", "gram_logdet_mean"]].agg(["mean", "std"])

    fig, axs = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 5.0), constrained_layout=True)
    if len(metrics) == 1:
        axs = [axs]
    x = np.arange(len(controllers))
    for ax, (metric, label, note) in zip(axs, metrics):
        means = [summary.loc[c, (metric, "mean")] for c in controllers]
        stds = [summary.loc[c, (metric, "std")] for c in controllers]
        colors = [CONTROLLER_COLORS.get(c, "tab:gray") for c in controllers]
        ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.86)
        ax.set_xticks(x)
        ax.set_xticklabels([CONTROLLER_LABELS.get(c, c) for c in controllers], rotation=35, ha="right")
        ax.set_title(label)
        ax.set_ylabel(note)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"Top ambiguous-pair endpoints, nearest evaluated grid points ({tag})", fontsize=15)
    fig.savefig(out / f"scalar_ab_ambiguous_pair_focus_{tag}.png", dpi=180)
    plt.close(fig)


def plot_for_grid(grid_dir: Path, diagnostics_dir: Path | None) -> None:
    df = pd.read_csv(grid_dir / "scalar_ab_grid_aggregate.csv")
    df = _add_derived_metrics(df)
    tag = grid_dir.name.replace("scalar_ab_grid_", "")
    plot_shared_scale_grid(df, grid_dir, tag)
    plot_difference_grid(df, grid_dir, tag)
    plot_return_scatter(df, grid_dir, tag)
    plot_ambiguous_pair_focus(df, diagnostics_dir, grid_dir, tag)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="Scalar-ab BLR+PPO run directory.")
    parser.add_argument("--diagnostics-dir", type=Path, default=None, help="Optional scalar-ab diagnostics directory with ambiguous_pairs.csv.")
    parser.add_argument("--grid", choices=["compact", "final", "both"], default="both")
    args = parser.parse_args()

    grids = []
    if args.grid in {"compact", "both"}:
        grids.append(args.run_dir / "scalar_ab_grid_compact")
    if args.grid in {"final", "both"}:
        grids.append(args.run_dir / "scalar_ab_grid_final")

    for grid_dir in grids:
        if not (grid_dir / "scalar_ab_grid_aggregate.csv").exists():
            print(f"Skipping missing grid: {grid_dir}")
            continue
        plot_for_grid(grid_dir, args.diagnostics_dir)
        print(f"Wrote comparison plots in {grid_dir}")


if __name__ == "__main__":
    main()
