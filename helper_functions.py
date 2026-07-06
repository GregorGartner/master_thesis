import pandas as pd
import matplotlib.pyplot as plt
import os, re
from callbacks import LivePlotCallback, PrintProgressCallback
import numpy as np



def _extract_experiment_title(path):
    """Walk up directory components to find one matching the timestamp pattern."""
    path = os.path.abspath(path)
    while True:
        base = os.path.basename(path)
        if not base:
            return ""
        m = re.match(r'^.*?\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(.*)$', base)
        if m and m.group(1).strip():
            return m.group(1).strip()
        parent = os.path.dirname(path)
        if parent == path:  # reached root
            return ""
        path = parent


def plot_reward_and_running_mean(log_dirs, window=20, save_plot=False, info_dict=None):
    """
    Plot episode returns and a rolling mean for all experiments found under the
    given log_dirs. Creates one subplot per experiment, arranged side-by-side.

    An "experiment" is any directory containing a "monitor.csv".

    If a corresponding SB3 `progress.csv` exists in the same directory and
    contains the column `train/regression_loss`, it is plotted in a second
    subplot below the return plot.
    """
    n = len(log_dirs)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 7), squeeze=False)

    for col, exp_dir in enumerate(log_dirs):
        ax_ret = axes[0, col]
        ax_reg = axes[1, col]

        df = pd.read_csv(os.path.join(exp_dir, "monitor.csv"), skiprows=1)  # first line is a comment
        rewards = df["r"].to_numpy()

        rolling = pd.Series(rewards).rolling(window, min_periods=1).mean()

        ax_ret.plot(rewards, label="episode return")
        ax_ret.plot(rolling, label=f"rolling mean ({window})")
        ax_ret.set_xlabel("episode")
        ax_ret.set_ylabel("return")
        ax_ret.grid(True)

        # Optional: plot regression loss from SB3 progress.csv
        progress_path = os.path.join(exp_dir, "progress.csv")
        plotted_reg = False
        if os.path.exists(progress_path):
            try:
                dfp = pd.read_csv(progress_path)
                loss_col = "train/regression_loss"
                if loss_col in dfp.columns and len(dfp[loss_col]) > 0:
                    x_prog = np.arange(len(dfp[loss_col]), dtype=np.int64)
                    ax_reg.plot(x_prog, dfp[loss_col].to_numpy(), label="regression loss")
                    unc_col = "train/regression_uncertainty"
                    if unc_col in dfp.columns and len(dfp[unc_col]) == len(dfp[loss_col]):
                        ax_reg.plot(x_prog, dfp[unc_col].to_numpy(), label="uncertainty (mean var)")
                    ax_reg.set_xlabel("update")
                    ax_reg.set_ylabel("regression loss")
                    ax_reg.grid(True)
                    ax_reg.legend(loc="best")
                    plotted_reg = True
            except Exception:
                # Keep plotting robust: ignore progress.csv issues silently.
                pass

        if not plotted_reg:
            ax_reg.set_axis_off()

        title = _extract_experiment_title(exp_dir)
        ax_ret.set_title(title)

        # ax.set_title(os.path.basename(exp_dir) or exp_dir)

    fig.tight_layout()

    # Add info_dict as text below the plots
    if info_dict:
        info_lines = [f"{k}: {v}" for k, v in info_dict.items()]
        info_text = "\n".join(info_lines)
        fig.subplots_adjust(bottom=0.18)
        fig.text(
            0.5, 0.02, info_text,
            ha="center", va="bottom", fontsize=8,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.9),
        )

    if save_plot and len(log_dirs) == 1:
        title = _extract_experiment_title(log_dirs[0])
        out_path = os.path.join(log_dirs[0], f"reward_plot_{title}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved reward plot to: {out_path}")

    plt.show()

