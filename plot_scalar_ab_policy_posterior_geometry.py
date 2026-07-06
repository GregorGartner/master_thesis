from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

from scalar_ab_lqr_utils import (
    PRIOR_MEAN,
    PRIOR_VAR,
    PROCESS_NOISE_STD,
    SELECTION_ROOT,
    WINDOW_LENGTH,
)


def _read_pointer(name: str) -> Path:
    path = SELECTION_ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Missing pointer: {path}")
    return Path(path.read_text().strip())


def _posterior_from_transitions(x: np.ndarray, u: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu0 = np.asarray(PRIOR_MEAN, dtype=np.float64)
    var0 = np.asarray(PRIOR_VAR, dtype=np.float64)
    precision = np.diag(1.0 / var0)
    rhs = precision @ mu0
    if len(y):
        z = np.column_stack([x[-WINDOW_LENGTH:], u[-WINDOW_LENGTH:]]).astype(np.float64)
        yy = y[-WINDOW_LENGTH:].astype(np.float64)
        inv_var = 1.0 / max(PROCESS_NOISE_STD**2, 1e-10)
        precision = precision + inv_var * (z.T @ z)
        rhs = rhs + inv_var * (z.T @ yy)
    cov = np.linalg.inv(precision)
    mean = np.linalg.solve(precision, rhs)
    return mean, cov


def _posterior_metrics(true_a: float, true_b: float, mean: np.ndarray, cov: np.ndarray) -> dict[str, float | bool]:
    true = np.asarray([true_a, true_b], dtype=np.float64)
    diff = true - mean
    sa = float(np.sqrt(max(cov[0, 0], 0.0)))
    sb = float(np.sqrt(max(cov[1, 1], 0.0)))
    corr = float(cov[0, 1] / np.sqrt(max(cov[0, 0] * cov[1, 1], 1e-12)))
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-12)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    mahal = float(np.sqrt(diff @ np.linalg.inv(cov) @ diff))
    coords = eigvecs.T @ diff / np.sqrt(eigvals)
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    return {
        "final_mean_a": float(mean[0]),
        "final_mean_b": float(mean[1]),
        "final_std_a": sa,
        "final_std_b": sb,
        "final_corr": corr,
        "mean_error": float(np.linalg.norm(diff)),
        "mean_error_a": float(diff[0]),
        "mean_error_b": float(diff[1]),
        "mahalanobis_to_true": mahal,
        "inside_2std": bool(mahal <= 2.0),
        "ellipse_area_2std": float(4.0 * np.pi * np.sqrt(max(np.linalg.det(cov), 1e-24))),
        "cov_condition": float(eigvals[0] / max(eigvals[1], 1e-12)),
        "major_axis_angle_deg": angle,
        "true_coord_major_std": float(coords[0]),
        "true_coord_minor_std": float(coords[1]),
    }


def _gram_metrics(x: np.ndarray, u: np.ndarray) -> dict[str, float]:
    z_full = np.column_stack([x, u]).astype(np.float64)
    z_window = z_full[-WINDOW_LENGTH:]
    out = {}
    for prefix, z in [("full", z_full), ("window", z_window)]:
        gram = z.T @ z if z.size else np.zeros((2, 2), dtype=np.float64)
        eig = np.linalg.eigvalsh(gram + 1e-12 * np.eye(2))
        ratio = z[:, 1] / np.maximum(np.abs(z[:, 0]), 1e-8) if z.size else np.asarray([])
        out[f"{prefix}_gram_lambda_min"] = float(eig[0])
        out[f"{prefix}_gram_lambda_max"] = float(eig[-1])
        out[f"{prefix}_gram_condition"] = float(eig[-1] / max(eig[0], 1e-12))
        out[f"{prefix}_gram_logdet"] = float(np.linalg.slogdet(gram + 1e-8 * np.eye(2))[1])
        out[f"{prefix}_u_over_x_var"] = float(np.var(ratio)) if ratio.size else 0.0
    return out


def _reconstruct_posteriors(trace_path: Path, out_csv: Path) -> pd.DataFrame:
    if out_csv.exists():
        return pd.read_csv(out_csv)
    cols = ["controller", "a", "b", "seed", "step", "x", "u"]
    trace = pd.read_csv(trace_path, usecols=cols)
    rows = []
    group_cols = ["controller", "a", "b", "seed"]
    for (controller, a, b, seed), ep in trace.groupby(group_cols, sort=False):
        ep = ep.sort_values("step")
        xs = ep["x"].to_numpy(dtype=np.float64)
        us = ep["u"].to_numpy(dtype=np.float64)
        if len(xs) < 2:
            continue
        z_x = xs[:-1]
        z_u = us[:-1]
        y = xs[1:]
        mean, cov = _posterior_from_transitions(z_x, z_u, y)
        row = {
            "controller": controller,
            "a": float(a),
            "b": float(b),
            "seed": int(seed),
        }
        row.update(_posterior_metrics(float(a), float(b), mean, cov))
        row.update(_gram_metrics(z_x, z_u))
        rows.append(row)
    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def _aggregate(rows: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    numeric = [
        col
        for col in rows.columns
        if col not in {"controller", "a", "b", "seed", "inside_2std"}
        and pd.api.types.is_numeric_dtype(rows[col])
    ]
    agg = rows.groupby(["controller", "a", "b"], dropna=False)[numeric].mean().reset_index()
    inside = rows.groupby(["controller", "a", "b"], dropna=False)["inside_2std"].mean().reset_index(name="inside_2std_rate")
    agg = agg.merge(inside, on=["controller", "a", "b"], how="left")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv, index=False)
    return agg


def _nearest_available_points(targets: pd.DataFrame, available: pd.DataFrame) -> pd.DataFrame:
    points = available[["a", "b"]].drop_duplicates().reset_index(drop=True)
    selected = []
    for _, target in targets.iterrows():
        d2 = (points["a"] - float(target["a"])) ** 2 + (points["b"] - float(target["b"])) ** 2
        row = points.loc[d2.idxmin()].to_dict()
        row["target_a"] = float(target["a"])
        row["target_b"] = float(target["b"])
        row["category"] = str(target.get("category", "selected"))
        row["name"] = str(target.get("name", f"{target['a']:.3f}_{target['b']:.3f}"))
        selected.append(row)
    return pd.DataFrame(selected).drop_duplicates(subset=["a", "b", "category", "name"]).reset_index(drop=True)


def _selected_targets(diagnostics_dir: Path, available: pd.DataFrame) -> pd.DataFrame:
    targets = []
    pairs_path = diagnostics_dir / "ambiguous_pairs.csv"
    if pairs_path.exists():
        pairs = pd.read_csv(pairs_path).head(8)
        for i, row in pairs.iterrows():
            targets.append({"a": row["a1"], "b": row["b1"], "category": "ambiguous_endpoint", "name": f"amb_{i+1}a"})
            targets.append({"a": row["a2"], "b": row["b2"], "category": "ambiguous_endpoint", "name": f"amb_{i+1}b"})
    easy_path = diagnostics_dir / "nonambiguous_ce_blr_comparison" / "selected_nonambiguous_points.csv"
    if easy_path.exists():
        easy = pd.read_csv(easy_path)
        easy = easy[easy["category"].isin(["easy_interior", "high_b_easy", "representative"])]
        targets.extend(easy[["a", "b", "category", "name"]].to_dict("records"))
    targets_df = pd.DataFrame(targets).drop_duplicates(subset=["a", "b", "category", "name"])
    return _nearest_available_points(targets_df, available)


def _ellipse_for_row(row: pd.Series, *, radius: float = 2.0, color: str = "k") -> Ellipse:
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
        edgecolor=color,
        linewidth=1.5,
        alpha=0.7,
    )


def _controller_colors(controllers: list[str]) -> dict[str, str]:
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {controller: palette[i % len(palette)] for i, controller in enumerate(controllers)}


def _plot_metric_boxplots(rows: pd.DataFrame, out: Path, title: str) -> None:
    controllers = list(rows["controller"].drop_duplicates())
    metrics = [
        ("mean_error", "posterior mean error", "linear"),
        ("mahalanobis_to_true", "true-vs-mean Mahalanobis", "linear"),
        ("ellipse_area_2std", "2-std ellipse area", "log"),
        ("cov_condition", "posterior covariance condition", "log"),
        ("window_gram_lambda_min", "window Gram lambda_min", "log"),
        ("window_u_over_x_var", "window Var(u/x)", "log"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (metric, label, scale) in zip(axes.ravel(), metrics):
        data = [rows.loc[rows["controller"] == c, metric].dropna().to_numpy() for c in controllers]
        ax.boxplot(data, tick_labels=controllers, showmeans=True, patch_artist=True)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=25)
        if scale == "log":
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / "policy_blr_posterior_metric_boxplots.png", dpi=180)
    plt.close(fig)


def _plot_metric_difference(rows: pd.DataFrame, out: Path, left: str, right: str, title: str) -> None:
    pivot = rows.pivot_table(index=["a", "b", "seed"], columns="controller")
    rows_out = []
    metrics = ["mean_error", "ellipse_area_2std", "window_gram_lambda_min", "window_u_over_x_var", "mahalanobis_to_true"]
    for metric in metrics:
        if (metric, left) not in pivot.columns or (metric, right) not in pivot.columns:
            continue
        delta = pivot[(metric, right)] - pivot[(metric, left)]
        rows_out.extend({"metric": metric, "delta": float(v)} for v in delta.dropna().to_numpy())
    if not rows_out:
        return
    df = pd.DataFrame(rows_out)
    df.to_csv(out / f"policy_blr_posterior_deltas_{right}_minus_{left}.csv", index=False)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for ax, metric in zip(np.ravel(axes), metrics):
        vals = df.loc[df["metric"] == metric, "delta"].to_numpy()
        if vals.size == 0:
            ax.axis("off")
            continue
        ax.hist(vals, bins=40, alpha=0.85)
        ax.axvline(0.0, color="k", linewidth=1.0)
        ax.axvline(float(np.mean(vals)), color="crimson", linestyle="--", label=f"mean {np.mean(vals):.3g}")
        ax.set_title(f"{right} - {left}\n{metric}")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / f"policy_blr_posterior_deltas_{right}_minus_{left}.png", dpi=180)
    plt.close(fig)


def _plot_selected_ellipses(agg: pd.DataFrame, selected: pd.DataFrame, out: Path, title: str, category: str) -> None:
    controllers = list(agg["controller"].drop_duplicates())
    colors = _controller_colors(controllers)
    selected_cat = selected[selected["category"] == category]
    if selected_cat.empty:
        return
    fig, axes = plt.subplots(1, len(controllers), figsize=(5 * len(controllers), 5), squeeze=False)
    for ax, controller in zip(axes.ravel(), controllers):
        sub = agg[agg["controller"] == controller]
        merged = selected_cat.merge(sub, on=["a", "b"], how="inner")
        color = colors[controller]
        for _, row in merged.iterrows():
            ax.add_patch(_ellipse_for_row(row, color=color))
            ax.scatter(row["a"], row["b"], marker="x", s=55, linewidths=2.0, color=color)
            ax.scatter(row["final_mean_a"], row["final_mean_b"], marker="o", s=35, facecolors="none", edgecolors=color, linewidths=1.5)
            ax.plot([row["a"], row["final_mean_a"]], [row["b"], row["final_mean_b"]], color=color, alpha=0.35, linewidth=1.0)
        ax.set_title(controller)
        ax.set_xlabel("a")
        ax.set_ylabel("b")
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(title + f"\n{category}: 'x'=true, 'o'=BLR mean from policy trajectory, ellipse=2 std")
    fig.tight_layout()
    fig.savefig(out / f"policy_blr_posterior_ellipses_{category}.png", dpi=180)
    plt.close(fig)


def _plot_heatmaps(agg: pd.DataFrame, out: Path, title: str) -> None:
    controllers = list(agg["controller"].drop_duplicates())
    metrics = [
        ("ellipse_area_2std", "2-std ellipse area", "viridis"),
        ("window_gram_lambda_min", "window Gram lambda_min", "viridis"),
        ("mean_error", "posterior mean error", "magma"),
    ]
    for metric, label, cmap in metrics:
        vals = agg[metric].to_numpy(dtype=np.float64)
        vmin, vmax = np.nanpercentile(vals, [2, 98])
        fig, axes = plt.subplots(1, len(controllers), figsize=(5 * len(controllers), 4), squeeze=False)
        for ax, controller in zip(axes.ravel(), controllers):
            sub = agg[agg["controller"] == controller]
            a_vals = np.sort(sub["a"].unique())
            b_vals = np.sort(sub["b"].unique())
            grid = sub.pivot(index="b", columns="a", values=metric).sort_index().to_numpy()
            im = ax.imshow(
                grid,
                origin="lower",
                aspect="auto",
                extent=[a_vals.min(), a_vals.max(), b_vals.min(), b_vals.max()],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            ax.set_title(controller)
            ax.set_xlabel("a")
            ax.set_ylabel("b")
        fig.colorbar(im, ax=axes.ravel().tolist(), label=label)
        fig.suptitle(f"{title}: {label} (shared color scale)")
        fig.tight_layout()
        fig.savefig(out / f"policy_blr_posterior_heatmap_{metric}.png", dpi=180)
        plt.close(fig)


def _run_one(root: Path, diagnostics_dir: Path, label: str) -> None:
    grid_dir = root / "scalar_ab_grid_final"
    trace_path = grid_dir / "trajectory_trace.csv"
    out = grid_dir / "policy_blr_posterior_geometry"
    out.mkdir(parents=True, exist_ok=True)
    episodes = _reconstruct_posteriors(trace_path, out / "policy_blr_posterior_episode_summary.csv")
    agg = _aggregate(episodes, out / "policy_blr_posterior_grid_aggregate.csv")
    selected = _selected_targets(diagnostics_dir, agg)
    selected.to_csv(out / "selected_policy_posterior_points.csv", index=False)

    _plot_metric_boxplots(episodes, out, f"{label}: reconstructed BLR posterior from policy trajectories")
    _plot_heatmaps(agg, out, f"{label}: reconstructed BLR posterior")
    for category in ["ambiguous_endpoint", "easy_interior", "high_b_easy", "representative"]:
        _plot_selected_ellipses(agg, selected, out, f"{label}: reconstructed BLR posterior ellipses", category)

    controllers = set(episodes["controller"])
    if {"blr_mean_P06", "blr_mean_cov_P06"}.issubset(controllers):
        _plot_metric_difference(episodes, out, "blr_mean_P06", "blr_mean_cov_P06", f"{label}: mean+cov trajectory info minus mean-only")
    if {"gradual_mle_P06", "gradual_nll_P06"}.issubset(controllers):
        _plot_metric_difference(episodes, out, "gradual_mle_P06", "gradual_nll_P06", f"{label}: NLL trajectory info minus MLE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-dir", type=Path, default=None)
    parser.add_argument("--blr-root", type=Path, default=None)
    parser.add_argument("--neural-root", type=Path, default=None)
    args = parser.parse_args()
    diagnostics_dir = args.diagnostics_dir or _read_pointer("latest_scalar_ab_diagnostics.txt")
    blr_root = args.blr_root or _read_pointer("latest_scalar_ab_blr_ppo.txt")
    neural_root = args.neural_root or _read_pointer("latest_scalar_ab_neural_rma.txt")
    _run_one(blr_root, diagnostics_dir, "Scalar (a,b) BLR+PPO")
    _run_one(neural_root, diagnostics_dir, "Scalar (a,b) neural RMA")
    print("Saved policy posterior geometry plots.", flush=True)


if __name__ == "__main__":
    main()
