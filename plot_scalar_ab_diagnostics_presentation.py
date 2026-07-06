from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _as_float(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _grid(rows: list[dict[str, str]], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_vals = np.asarray(sorted({float(row["a"]) for row in rows}), dtype=np.float64)
    b_vals = np.asarray(sorted({float(row["b"]) for row in rows}), dtype=np.float64)
    ai = {value: idx for idx, value in enumerate(a_vals)}
    bi = {value: idx for idx, value in enumerate(b_vals)}
    values = np.full((len(b_vals), len(a_vals)), np.nan, dtype=np.float64)
    for row in rows:
        values[bi[float(row["b"])], ai[float(row["a"])]] = float(row[key])
    return a_vals, b_vals, values


def _group_rollouts(rows: list[dict[str, str]]) -> dict[tuple[float, float], list[dict[str, str]]]:
    grouped: dict[tuple[float, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["a"]), float(row["b"]))].append(row)
    return dict(grouped)


def _group_metrics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (a, b), group in sorted(_group_rollouts(rows).items()):
        cov_lmin = _as_float(group, "cov_lambda_min")
        cov_lmax = _as_float(group, "cov_lambda_max")
        std_a = _as_float(group, "final_std_a")
        std_b = _as_float(group, "final_std_b")
        area = math.pi * 2.0 * np.sqrt(cov_lmin) * 2.0 * np.sqrt(cov_lmax)
        out.append(
            {
                "a": a,
                "b": b,
                "mean_a": float(np.mean(_as_float(group, "final_mean_a"))),
                "mean_b": float(np.mean(_as_float(group, "final_mean_b"))),
                "std_a": float(np.mean(std_a)),
                "std_b": float(np.mean(std_b)),
                "corr": float(np.mean(_as_float(group, "final_corr"))),
                "gram_lambda_min": float(np.mean(_as_float(group, "gram_lambda_min"))),
                "gram_condition": float(np.mean(_as_float(group, "gram_condition"))),
                "cov_condition": float(np.mean(cov_lmax / np.maximum(cov_lmin, 1e-12))),
                "ellipse_area": float(np.mean(area)),
                "gain_error": float(np.mean(_as_float(group, "gain_error"))),
            }
        )
    return out


def _plot_ellipse(ax, mean_a: float, mean_b: float, std_a: float, std_b: float, corr: float, *, color: str) -> None:
    cov = np.asarray(
        [
            [std_a**2, corr * std_a * std_b],
            [corr * std_a * std_b, std_b**2],
        ],
        dtype=np.float64,
    )
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    ellipse = Ellipse(
        (mean_a, mean_b),
        width=4.0 * math.sqrt(eigvals[0]),
        height=4.0 * math.sqrt(eigvals[1]),
        angle=angle,
        fill=False,
        lw=2.0,
        color=color,
        alpha=0.9,
    )
    ax.add_patch(ellipse)


def _plot_ambiguous_pair_metrics(pairs: list[dict[str, str]], out: Path) -> None:
    specs = [
        ("delta_closed_loop", "|delta c|", "Near-identical closed-loop signatures", "tab:green"),
        ("delta_K", "|delta K|", "Different optimal gains", "tab:purple"),
        ("max_excess_cost", "max excess cost", "Wrong-controller cost is visible", "tab:blue"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, (key, xlabel, title, color) in zip(axes, specs):
        values = _as_float(pairs, key)
        median = float(np.median(values))
        p95 = float(np.quantile(values, 0.95))
        ax.hist(values, bins=32, color=color, alpha=0.78)
        ax.axvline(median, color="black", lw=2.2, label=f"median {median:.4g}")
        ax.axvline(p95, color="crimson", lw=2.2, ls="--", label=f"p95 {p95:.4g}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle("Close-Signature Ambiguous Pair Metrics", fontsize=15)
    fig.tight_layout()
    fig.savefig(out / "scalar_ab_ambiguous_pair_metrics.png", dpi=180)
    plt.close(fig)


def _plot_takeaways(analytic: list[dict[str, str]], pairs: list[dict[str, str]], rollouts: list[dict[str, str]], out: Path) -> None:
    a_vals, b_vals, closed_loop = _grid(analytic, "closed_loop")
    metrics = _group_metrics(rollouts)
    pair_delta_c = _as_float(pairs, "delta_closed_loop")
    pair_delta_k = _as_float(pairs, "delta_K")
    pair_excess = _as_float(pairs, "max_excess_cost")

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    extent = [float(a_vals.min()), float(a_vals.max()), float(b_vals.min()), float(b_vals.max())]
    im = axes[0, 0].imshow(closed_loop, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    for row in pairs[:200]:
        axes[0, 0].plot(
            [float(row["a1"]), float(row["a2"])],
            [float(row["b1"]), float(row["b2"])],
            color="white",
            lw=0.5,
            alpha=0.45,
        )
    axes[0, 0].scatter(_as_float(pairs, "a1"), _as_float(pairs, "b1"), s=10, color="tab:orange", label="pair endpoint 1")
    axes[0, 0].scatter(_as_float(pairs, "a2"), _as_float(pairs, "b2"), s=10, color="tab:red", label="pair endpoint 2")
    axes[0, 0].set_title("Ambiguous pairs lie on near-equal closed-loop bands")
    axes[0, 0].set_xlabel("a")
    axes[0, 0].set_ylabel("b")
    axes[0, 0].legend(fontsize=8)
    fig.colorbar(im, ax=axes[0, 0], label="closed-loop signature c = a-bK*(a,b)")

    sc = axes[0, 1].scatter(pair_delta_c, pair_delta_k, c=pair_excess, cmap="magma", s=32, alpha=0.85)
    axes[0, 1].set_title("Small signature gap, large gain gap")
    axes[0, 1].set_xlabel("|delta closed-loop signature|")
    axes[0, 1].set_ylabel("|delta optimal gain|")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].text(
        0.03,
        0.96,
        f"median |dc|={np.median(pair_delta_c):.4f}\n"
        f"median |dK|={np.median(pair_delta_k):.3f}\n"
        f"median excess={np.median(pair_excess):.3f}",
        transform=axes[0, 1].transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.7"),
    )
    fig.colorbar(sc, ax=axes[0, 1], label="max wrong-controller excess cost")

    axes[1, 0].hist(pair_excess, bins=32, color="tab:blue", alpha=0.75)
    axes[1, 0].axvline(np.median(pair_excess), color="black", lw=2.2, label="median")
    axes[1, 0].set_title("Ambiguity is control-relevant")
    axes[1, 0].set_xlabel("max wrong-controller excess cost")
    axes[1, 0].set_ylabel("number of close-signature pairs")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend()

    gram_lmin = np.asarray([row["gram_lambda_min"] for row in metrics])
    gram_cond = np.asarray([row["gram_condition"] for row in metrics])
    corr = np.asarray([row["corr"] for row in metrics])
    sc2 = axes[1, 1].scatter(gram_lmin, gram_cond, c=corr, cmap="coolwarm", s=60, alpha=0.9)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("BLR sees weak/rank-poor information")
    axes[1, 1].set_xlabel("smallest Gram eigenvalue")
    axes[1, 1].set_ylabel("Gram condition number")
    axes[1, 1].grid(True, alpha=0.25, which="both")
    axes[1, 1].text(
        0.03,
        0.96,
        f"mean lambda_min={np.mean(gram_lmin):.4f}\n"
        f"mean cond={np.mean(gram_cond):.0f}\n"
        f"mean corr={np.mean(corr):.2f}",
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.7"),
    )
    fig.colorbar(sc2, ax=axes[1, 1], label="posterior corr(a,b)")

    fig.suptitle("Scalar (a,b) Diagnostics: Ambiguity + Directional BLR Uncertainty", fontsize=17)
    fig.tight_layout()
    fig.savefig(out / "scalar_ab_diagnostics_takeaways.png", dpi=180)
    plt.close(fig)


def _plot_posterior_ellipses(analytic: list[dict[str, str]], rollouts: list[dict[str, str]], out: Path) -> None:
    a_vals, b_vals, closed_loop = _grid(analytic, "closed_loop")
    metrics = _group_metrics(rollouts)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(metrics), 1)))
    extent = [float(a_vals.min()), float(a_vals.max()), float(b_vals.min()), float(b_vals.max())]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))
    im = axes[0].imshow(closed_loop, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    for idx, row in enumerate(metrics):
        color = colors[idx]
        axes[0].scatter(row["a"], row["b"], marker="x", s=90, color=color)
    axes[0].set_title("CE-BLR rollout systems are top ambiguous-pair endpoints")
    axes[0].set_xlabel("a")
    axes[0].set_ylabel("b")
    fig.colorbar(im, ax=axes[0], label="closed-loop signature c=a-bK*")

    for idx, row in enumerate(metrics):
        color = colors[idx]
        axes[1].scatter(row["a"], row["b"], marker="x", s=80, color=color)
        axes[1].scatter(row["mean_a"], row["mean_b"], marker="o", s=45, color=color)
        axes[1].plot([row["a"], row["mean_a"]], [row["b"], row["mean_b"]], color=color, lw=1.2, alpha=0.45)
        _plot_ellipse(
            axes[1],
            row["mean_a"],
            row["mean_b"],
            row["std_a"],
            row["std_b"],
            row["corr"],
            color=color,
        )
    axes[1].set_title("Expanded view of final BLR posterior ellipses")
    axes[1].set_xlabel("a")
    axes[1].set_ylabel("b")
    axes[1].grid(True, alpha=0.25)
    axes[1].text(
        0.03,
        0.97,
        "x = true parameter\no = posterior mean\nellipse = 2 std posterior\nline = posterior bias",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"),
    )
    fig.suptitle("Scalar (a,b) CE-BLR Posterior Geometry on Ambiguous Systems", fontsize=16)
    fig.tight_layout()
    fig.savefig(out / "scalar_ab_blr_posterior_ellipses_expanded.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(closed_loop, origin="lower", extent=extent, aspect="auto", cmap="Greys", alpha=0.22)
    for idx, row in enumerate(metrics):
        color = colors[idx]
        ax.scatter(row["a"], row["b"], marker="x", s=90, color=color)
        ax.scatter(row["mean_a"], row["mean_b"], marker="o", s=45, color=color)
        _plot_ellipse(ax, row["mean_a"], row["mean_b"], row["std_a"], row["std_b"], row["corr"], color=color)
    ax.set_title("Representative final BLR posteriors remain elongated")
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.grid(True, alpha=0.2)
    ax.text(
        0.02,
        0.98,
        "x = true parameter, circle = posterior mean, ellipse = 2 std posterior",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"),
    )
    fig.colorbar(im, ax=ax, label="closed-loop signature c")
    fig.tight_layout()
    fig.savefig(out / "scalar_ab_blr_posterior_ellipses.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    args = parser.parse_args()
    diagnostics = args.diagnostics_dir.resolve()
    analytic = _read_csv(diagnostics / "analytic_grid.csv")
    pairs = _read_csv(diagnostics / "ambiguous_pairs.csv")
    rollouts = _read_csv(diagnostics / "blr_ce_lqr_rollouts.csv")
    _plot_ambiguous_pair_metrics(pairs, diagnostics)
    _plot_takeaways(analytic, pairs, rollouts, diagnostics)
    _plot_posterior_ellipses(analytic, rollouts, diagnostics)
    print(f"Wrote scalar (a,b) presentation diagnostics to {diagnostics}")


if __name__ == "__main__":
    main()
