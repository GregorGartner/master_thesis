from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CSV_PATH = Path("experiments/stabler_lqr_2026-03-15_13-46-42_privileged/test/eval_episodes.csv")
DEFAULT_OUTPUT_DIR = Path("experiments/stabler_lqr_2026-03-15_13-46-42_privileged/test/analysis_plots")
N_BINS = 8
LOW_REWARD_QUANTILE = 0.1
MIN_CELL_COUNT_FOR_ALERT = 3
INFEASIBLE_MEAN_REWARD_QUANTILE = 0.25
INFEASIBLE_LOW_REWARD_FRACTION = 0.6
INFEASIBLE_MIN_CELL_COUNT = 3


def pretty_label(column_name: str) -> str:
    base_name = column_name[:-5] if column_name.endswith("_norm") else column_name
    return base_name.replace("_", " ").title()


def save_current_figure(filename: str) -> None:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DEFAULT_OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")


def add_trend_line(ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray) -> None:
    if x_values.size < 2 or float(np.std(x_values)) == 0.0:
        return
    slope, intercept = np.polyfit(x_values, y_values, 1)
    x_line = np.linspace(float(x_values.min()), float(x_values.max()), 200)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="tab:orange", linewidth=2, label="linear fit")


def compute_binned_means(x_values: np.ndarray, y_values: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    if x_values.size == 0:
        return np.array([]), np.array([])

    x_min = float(x_values.min())
    x_max = float(x_values.max())
    if x_min == x_max:
        return np.array([x_min]), np.array([float(np.mean(y_values))])

    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_ids = np.digitize(x_values, bin_edges[1:-1], right=False)

    bin_centers: list[float] = []
    bin_means: list[float] = []
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if not np.any(mask):
            continue
        lo = bin_edges[bin_index]
        hi = bin_edges[bin_index + 1]
        bin_centers.append(0.5 * (lo + hi))
        bin_means.append(float(np.mean(y_values[mask])))

    return np.asarray(bin_centers, dtype=float), np.asarray(bin_means, dtype=float)


def compute_pairwise_bin_stats(
    x_values: np.ndarray,
    y_values: np.ndarray,
    reward_values: np.ndarray,
    low_reward_mask: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_min = float(x_values.min())
    x_max = float(x_values.max())
    y_min = float(y_values.min())
    y_max = float(y_values.max())

    if x_min == x_max:
        x_edges = np.array([x_min - 0.5, x_max + 0.5], dtype=float)
    else:
        x_edges = np.linspace(x_min, x_max, n_bins + 1)

    if y_min == y_max:
        y_edges = np.array([y_min - 0.5, y_max + 0.5], dtype=float)
    else:
        y_edges = np.linspace(y_min, y_max, n_bins + 1)

    n_x_bins = len(x_edges) - 1
    n_y_bins = len(y_edges) - 1

    x_ids = np.clip(np.digitize(x_values, x_edges[1:-1], right=False), 0, n_x_bins - 1)
    y_ids = np.clip(np.digitize(y_values, y_edges[1:-1], right=False), 0, n_y_bins - 1)

    reward_sums = np.zeros((n_y_bins, n_x_bins), dtype=float)
    counts = np.zeros((n_y_bins, n_x_bins), dtype=int)
    low_counts = np.zeros((n_y_bins, n_x_bins), dtype=int)

    for x_idx, y_idx, reward, is_low in zip(x_ids, y_ids, reward_values, low_reward_mask):
        reward_sums[y_idx, x_idx] += float(reward)
        counts[y_idx, x_idx] += 1
        low_counts[y_idx, x_idx] += int(bool(is_low))

    mean_rewards = np.full((n_y_bins, n_x_bins), np.nan, dtype=float)
    low_reward_fraction = np.full((n_y_bins, n_x_bins), np.nan, dtype=float)
    valid_mask = counts > 0
    mean_rewards[valid_mask] = reward_sums[valid_mask] / counts[valid_mask]
    low_reward_fraction[valid_mask] = low_counts[valid_mask] / counts[valid_mask]

    return mean_rewards, low_reward_fraction, counts, x_edges, y_edges


def summarize_worst_pairwise_cells(
    df: pd.DataFrame,
    parameter_columns: list[str],
    reward_threshold: float,
    infeasible_mean_reward_threshold: float,
    n_bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    low_reward_mask = df["reward"].to_numpy(dtype=float) <= reward_threshold

    for x_name, y_name in combinations(parameter_columns, 2):
        x_values = df[x_name].to_numpy(dtype=float)
        y_values = df[y_name].to_numpy(dtype=float)
        reward_values = df["reward"].to_numpy(dtype=float)
        mean_rewards, low_reward_fraction, counts, x_edges, y_edges = compute_pairwise_bin_stats(
            x_values=x_values,
            y_values=y_values,
            reward_values=reward_values,
            low_reward_mask=low_reward_mask,
            n_bins=n_bins,
        )

        for y_idx in range(counts.shape[0]):
            for x_idx in range(counts.shape[1]):
                cell_count = int(counts[y_idx, x_idx])
                if cell_count == 0:
                    continue
                rows.append(
                    {
                        "x_param": x_name,
                        "y_param": y_name,
                        "x_bin_start": float(x_edges[x_idx]),
                        "x_bin_end": float(x_edges[x_idx + 1]),
                        "y_bin_start": float(y_edges[y_idx]),
                        "y_bin_end": float(y_edges[y_idx + 1]),
                        "count": cell_count,
                        "mean_reward": float(mean_rewards[y_idx, x_idx]),
                        "low_reward_fraction": float(low_reward_fraction[y_idx, x_idx]),
                        "infeasible_flag": bool(
                            cell_count >= INFEASIBLE_MIN_CELL_COUNT
                            and float(low_reward_fraction[y_idx, x_idx]) >= INFEASIBLE_LOW_REWARD_FRACTION
                            and float(mean_rewards[y_idx, x_idx]) <= infeasible_mean_reward_threshold
                        ),
                    }
                )

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        by=["low_reward_fraction", "mean_reward", "count"],
        ascending=[False, True, False],
    )
    return summary


def main() -> None:
    if not DEFAULT_CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {DEFAULT_CSV_PATH}")

    df = pd.read_csv(DEFAULT_CSV_PATH)

    required_columns = {"episode", "reward"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    parameter_columns = [col for col in df.columns if col.endswith("_norm")]
    if not parameter_columns:
        raise ValueError("No parameter columns found. Expected columns ending with '_norm'.")

    reward_values = df["reward"].to_numpy(dtype=float)
    reward_threshold = float(df["reward"].quantile(LOW_REWARD_QUANTILE))
    infeasible_mean_reward_threshold = float(df["reward"].quantile(INFEASIBLE_MEAN_REWARD_QUANTILE))
    low_reward_mask = reward_values <= reward_threshold

    print(
        f"Low-reward threshold set to the {int(LOW_REWARD_QUANTILE * 100)}th percentile: "
        f"{reward_threshold:.2f}"
    )
    print(
        f"Infeasibility mean-reward threshold set to the "
        f"{int(INFEASIBLE_MEAN_REWARD_QUANTILE * 100)}th percentile: "
        f"{infeasible_mean_reward_threshold:.2f}"
    )
    print(
        f"Episodes at or below threshold: {int(low_reward_mask.sum())}/{len(df)}"
    )

    plt.figure(figsize=(10, 4.5))
    plt.plot(df["episode"], reward_values, color="tab:blue", linewidth=2, label="reward")
    plt.axhline(float(df["reward"].mean()), color="tab:red", linestyle="--", linewidth=1.5, label="mean reward")
    plt.title("Reward Over Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_current_figure("reward_over_episode.png")

    plt.figure(figsize=(8, 4.5))
    correlations = df[parameter_columns].corrwith(df["reward"]).sort_values()
    plt.barh(
        [pretty_label(col) for col in correlations.index],
        correlations.values,
        color="tab:green",
        alpha=0.8,
    )
    plt.axvline(0.0, color="black", linewidth=1)
    plt.title("Reward Correlation By Parameter")
    plt.xlabel("Pearson Correlation")
    plt.grid(True, axis="x", alpha=0.3)
    save_current_figure("reward_parameter_correlations.png")

    if "regression_loss" in df.columns:
        x_values = df["regression_loss"].to_numpy(dtype=float)
        corr_value = df["regression_loss"].corr(df["reward"])

        plt.figure(figsize=(8, 5))
        plt.scatter(x_values, reward_values, alpha=0.7, s=35, color="tab:purple", label="episodes")
        add_trend_line(plt.gca(), x_values, reward_values)
        title = "Return vs Encoder Error"
        if pd.notna(corr_value):
            title += f" (r={corr_value:.2f})"
        plt.title(title)
        plt.xlabel("Encoder Error (episode mean MSE)")
        plt.ylabel("Episode Return")
        plt.grid(True, alpha=0.3)
        plt.legend()
        save_current_figure("return_vs_encoder_error.png")

    if "regression_uncertainty" in df.columns:
        x_values = df["regression_uncertainty"].to_numpy(dtype=float)
        corr_value = df["regression_uncertainty"].corr(df["reward"])

        plt.figure(figsize=(8, 5))
        plt.scatter(x_values, reward_values, alpha=0.7, s=35, color="tab:cyan", label="episodes")
        add_trend_line(plt.gca(), x_values, reward_values)
        title = "Reward vs Regression Uncertainty"
        if pd.notna(corr_value):
            title += f" (r={corr_value:.2f})"
        plt.title(title)
        plt.xlabel("Regression Uncertainty")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.3)
        plt.legend()
        save_current_figure("reward_vs_regression_uncertainty.png")

    for column_name in parameter_columns:
        x_values = df[column_name].to_numpy(dtype=float)
        corr_value = df[column_name].corr(df["reward"])

        plt.figure(figsize=(9, 8))

        ax1 = plt.subplot(2, 1, 1)
        ax1.scatter(x_values, reward_values, alpha=0.7, s=35, color="tab:purple", label="episodes")
        add_trend_line(ax1, x_values, reward_values)
        title = f"Reward vs {pretty_label(column_name)}"
        if pd.notna(corr_value):
            title += f" (r={corr_value:.2f})"
        ax1.set_title(title)
        ax1.set_xlabel(f"{pretty_label(column_name)} (normalized)")
        ax1.set_ylabel("Reward")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = plt.subplot(2, 1, 2)
        bin_centers, bin_means = compute_binned_means(x_values, reward_values, N_BINS)
        ax2.plot(bin_centers, bin_means, marker="o", linewidth=2, color="tab:blue")
        ax2.set_title(f"Binned Mean Reward vs {pretty_label(column_name)}")
        ax2.set_xlabel(f"{pretty_label(column_name)} (normalized)")
        ax2.set_ylabel("Mean Reward")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_current_figure(f"reward_vs_{column_name}.png")

    pairwise_summary = summarize_worst_pairwise_cells(
        df=df,
        parameter_columns=parameter_columns,
        reward_threshold=reward_threshold,
        infeasible_mean_reward_threshold=infeasible_mean_reward_threshold,
        n_bins=N_BINS,
    )

    if not pairwise_summary.empty:
        summary_path = DEFAULT_OUTPUT_DIR / "pairwise_worst_cells.csv"
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pairwise_summary.to_csv(summary_path, index=False)
        print(f"Saved pairwise cell summary to: {summary_path}")

        alert_cells = pairwise_summary[pairwise_summary["count"] >= MIN_CELL_COUNT_FOR_ALERT].head(10)
        if not alert_cells.empty:
            print("\nWorst pairwise cells with enough samples:")
            for row in alert_cells.itertuples(index=False):
                print(
                    f"- {row.x_param} in [{row.x_bin_start:.3f}, {row.x_bin_end:.3f}], "
                    f"{row.y_param} in [{row.y_bin_start:.3f}, {row.y_bin_end:.3f}] | "
                    f"count={row.count}, mean_reward={row.mean_reward:.2f}, "
                    f"low_reward_fraction={row.low_reward_fraction:.2f}, "
                    f"infeasible_flag={bool(row.infeasible_flag)}"
                )

        infeasible_cells = pairwise_summary[pairwise_summary["infeasible_flag"]]
        if not infeasible_cells.empty:
            print("\nFlagged pairwise cells as potentially infeasible:")
            for row in infeasible_cells.head(10).itertuples(index=False):
                print(
                    f"- {row.x_param} in [{row.x_bin_start:.3f}, {row.x_bin_end:.3f}], "
                    f"{row.y_param} in [{row.y_bin_start:.3f}, {row.y_bin_end:.3f}] | "
                    f"count={row.count}, mean_reward={row.mean_reward:.2f}, "
                    f"low_reward_fraction={row.low_reward_fraction:.2f}"
                )
        else:
            print("\nNo pairwise cells met the current infeasibility flag criteria.")

    for x_name, y_name in combinations(parameter_columns, 2):
        x_values = df[x_name].to_numpy(dtype=float)
        y_values = df[y_name].to_numpy(dtype=float)

        if float(np.std(x_values)) == 0.0 or float(np.std(y_values)) == 0.0:
            continue

        mean_rewards, low_reward_fraction, counts, x_edges, y_edges = compute_pairwise_bin_stats(
            x_values=x_values,
            y_values=y_values,
            reward_values=reward_values,
            low_reward_mask=low_reward_mask,
            n_bins=N_BINS,
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10))

        mean_im = ax1.imshow(
            mean_rewards,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            cmap="viridis",
        )
        ax1.set_title(f"Mean Reward: {pretty_label(x_name)} vs {pretty_label(y_name)}")
        ax1.set_xlabel(f"{pretty_label(x_name)} (normalized)")
        ax1.set_ylabel(f"{pretty_label(y_name)} (normalized)")
        fig.colorbar(mean_im, ax=ax1, label="Mean Reward")

        frac_im = ax2.imshow(
            low_reward_fraction,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        ax2.set_title(
            f"Low-Reward Fraction (reward <= {reward_threshold:.2f})"
        )
        ax2.set_xlabel(f"{pretty_label(x_name)} (normalized)")
        ax2.set_ylabel(f"{pretty_label(y_name)} (normalized)")
        fig.colorbar(frac_im, ax=ax2, label="Fraction of Low-Reward Episodes")

        for y_idx in range(counts.shape[0]):
            y_center = 0.5 * (y_edges[y_idx] + y_edges[y_idx + 1])
            for x_idx in range(counts.shape[1]):
                cell_count = int(counts[y_idx, x_idx])
                if cell_count == 0:
                    continue
                x_center = 0.5 * (x_edges[x_idx] + x_edges[x_idx + 1])
                cell_mean_reward = float(mean_rewards[y_idx, x_idx])
                cell_low_reward_fraction = float(low_reward_fraction[y_idx, x_idx])
                infeasible_flag = (
                    cell_count >= INFEASIBLE_MIN_CELL_COUNT
                    and cell_low_reward_fraction >= INFEASIBLE_LOW_REWARD_FRACTION
                    and cell_mean_reward <= infeasible_mean_reward_threshold
                )
                ax2.text(
                    x_center,
                    y_center,
                    str(cell_count),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
                )
                if infeasible_flag:
                    ax1.scatter(
                        [x_center],
                        [y_center],
                        marker="x",
                        s=90,
                        linewidths=2,
                        color="red",
                    )
                    ax2.scatter(
                        [x_center],
                        [y_center],
                        marker="x",
                        s=90,
                        linewidths=2,
                        color="cyan",
                    )

        plt.tight_layout()
        save_current_figure(f"interaction_{x_name}_vs_{y_name}.png")

    plt.show()


if __name__ == "__main__":
    main()
