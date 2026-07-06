import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import os
import json


class PrintProgressCallback(BaseCallback):
    """
    Prints episode return and length to console every couple time step
    """
    def __init__(self, print_every=2000):
        super().__init__()
        self.print_every = print_every

    def _on_step(self) -> bool:
        # When an episode ends, SB3 puts info["episode"] into infos (Monitor)
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_r = info["episode"]["r"]
                ep_l = info["episode"]["l"]
                print(f"t={self.num_timesteps:6d} | ep_return={ep_r:7.1f} | ep_len={ep_l:4d}")
        return True



class LivePlotCallback(BaseCallback):
    def __init__(self, update_every_episodes: int = 5, running_mean_window: int = 20):
        super().__init__()
        self.update_every_episodes = int(update_every_episodes)
        self.running_mean_window = int(running_mean_window)
        self.periodic_save_path = None
        self.episode_csv_path = None
        self.regression_csv_path = None
        self.periodic_save_every_episodes = max(1, 10 * self.update_every_episodes)
        self.returns = []
        self.shaped_returns = []
        self.running_mean = []
        self.regression_losses = []
        self.regression_mses = []
        self.uncertainties = []
        self._last_seen_n_updates = None

        plt.ion()
        self.fig, (self.ax_ret, self.ax_reg) = plt.subplots(2, 1, sharex=False, figsize=(7, 7))

        (self.line_ret,) = self.ax_ret.plot([], [], label="episode return")
        (self.line_ret_shaped,) = self.ax_ret.plot([], [], label="episode return (shaped)")
        (self.line_rm,) = self.ax_ret.plot([], [], label=f"running mean ({self.running_mean_window})")
        self.ax_ret.set_xlabel("episode")
        self.ax_ret.set_ylabel("return")
        self.ax_ret.grid(True)
        self.ax_ret.legend(loc="best")

        (self.line_reg,) = self.ax_reg.plot([], [], label="regression loss (NLL/MSE)")
        (self.line_mse,) = self.ax_reg.plot([], [], label="regression MSE")
        (self.line_unc,) = self.ax_reg.plot([], [], label="uncertainty (mean var)")
        self.ax_reg.set_xlabel("update")
        self.ax_reg.set_ylabel("regression / uncertainty")
        self.ax_reg.grid(True)
        self.ax_reg.legend(loc="best")
        self.ax_reg.set_axis_off()

    def _append_regression_csv(self, reg: float, mse: float | None, unc: float | None) -> None:
        if not self.regression_csv_path:
            return
        if not os.path.exists(self.regression_csv_path):
            open(self.regression_csv_path, "w", encoding="utf-8").write("regression_loss,regression_mse,uncertainty\n")
        mse_str = "" if mse is None else f"{float(mse)}"
        unc_str = "" if unc is None else f"{float(unc)}"
        open(self.regression_csv_path, "a", encoding="utf-8").write(f"{float(reg)},{mse_str},{unc_str}\n")

    def _maybe_record_regression_loss(self) -> None:
        """Record latest regression loss if the model exposes it.

        For ContextPPO, we expose `model._last_regression_loss` and we can
        detect new updates via `model._n_updates`.
        """

        model = getattr(self, "model", None)
        if model is None:
            return

        reg = getattr(model, "_last_regression_loss", None)
        if reg is None:
            return
        mse = getattr(model, "_last_regression_mse", None)
        unc = getattr(model, "_last_regression_uncertainty", None)

        n_updates = getattr(model, "_n_updates", None)
        if n_updates is None:
            # Fallback: append every time we see a value
            self.regression_losses.append(float(reg))
            if mse is not None:
                self.regression_mses.append(float(mse))
            if unc is not None:
                self.uncertainties.append(float(unc))
            self._append_regression_csv(float(reg), None if mse is None else float(mse), None if unc is None else float(unc))
            return

        if self._last_seen_n_updates is None or n_updates != self._last_seen_n_updates:
            self._last_seen_n_updates = n_updates
            self.regression_losses.append(float(reg))
            if mse is not None:
                self.regression_mses.append(float(mse))
            if unc is not None:
                self.uncertainties.append(float(unc))
            self._append_regression_csv(float(reg), None if mse is None else float(mse), None if unc is None else float(unc))

    def _update_plots(self) -> None:
        # Returns plot
        x = np.arange(len(self.returns), dtype=np.int64)
        self.line_ret.set_data(x, self.returns)
        self.line_ret_shaped.set_data(x, self.shaped_returns)
        self.line_rm.set_data(x, self.running_mean)
        self.ax_ret.relim()
        self.ax_ret.autoscale_view()

        # Regression plot (if any)
        if len(self.regression_losses) > 0:
            x_reg = np.arange(len(self.regression_losses), dtype=np.int64)
            self.line_reg.set_data(x_reg, self.regression_losses)
            if len(self.regression_mses) == len(self.regression_losses):
                self.line_mse.set_data(x_reg, self.regression_mses)
            if len(self.uncertainties) == len(self.regression_losses):
                self.line_unc.set_data(x_reg, self.uncertainties)
            self.ax_reg.set_axis_on()
            self.ax_reg.relim()
            self.ax_reg.autoscale_view()
        else:
            self.ax_reg.set_axis_off()

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _on_step(self) -> bool:
        # Regression loss is produced during `train()`, which runs between rollouts.
        # We read it from the model (if available) without reading any files.
        self._maybe_record_regression_loss()

        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_return = float(info["episode"]["r"])
                ep_return_shaped = float(info["episode"].get("r_shaped", ep_return))
                self.returns.append(ep_return)
                self.shaped_returns.append(ep_return_shaped)
                if self.episode_csv_path:
                    if not os.path.exists(self.episode_csv_path):
                        open(self.episode_csv_path, "w", encoding="utf-8").write("r,r_shaped\n")
                    open(self.episode_csv_path, "a", encoding="utf-8").write(f"{ep_return},{ep_return_shaped}\n")

                if self.running_mean_window <= 1:
                    self.running_mean.append(float(self.returns[-1]))
                else:
                    w = min(len(self.returns), self.running_mean_window)
                    self.running_mean.append(float(np.mean(self.returns[-w:])))

                if len(self.returns) % self.update_every_episodes == 0:
                    self._update_plots()
                    if self.periodic_save_path and len(self.returns) % self.periodic_save_every_episodes == 0:
                        self.fig.savefig(self.periodic_save_path, dpi=120, bbox_inches="tight")
                    
        return True


class SaveModelCallback(BaseCallback):
    """Save latest and best model checkpoints during training."""

    def __init__(
        self,
        save_every_episodes: int = 200,
        save_dir: str | None = None,
        save_name: str = "weights",
        save_best_name: str = "weights_best",
        best_metric: str = "auto",   # "auto", "regression_mse", "regression_loss", "uncertainty_loss", "episode_reward"
        best_mode: str | None = None,  # "min" / "max"; None => inferred from metric
        save_on_training_end: bool = True,
    ):
        super().__init__()
        self.save_every_episodes = int(save_every_episodes)
        self.save_dir = save_dir
        self.save_latest_name = str(save_name)
        self.save_best_name = str(save_best_name)
        self.best_metric = str(best_metric)
        self.best_mode = best_mode
        self.save_on_training_end = bool(save_on_training_end)

        self._episode_count = 0
        self._best_value = None
        self._last_seen_n_updates = None

    def _save_model(self, save_name: str) -> None:
        # checking whether experiments directory specified
        if self.save_dir is None:
            raise ValueError("SaveModelCallback requires a non-empty save_dir.")
        
        # save the latest model
        os.makedirs(self.save_dir, exist_ok=True)
        self.model.save(os.path.join(self.save_dir, save_name))

    def _save_latest(self) -> None:
        self._save_model(self.save_latest_name)

    def _resolved_best_metric(self) -> str:
        # Use the user-specified metric if provided.
        if self.best_metric != "auto":
            return self.best_metric

        # Otherwise choose a sensible default based on context_mode.
        context_mode = str(getattr(self.model, "context_mode", ""))
        if context_mode.startswith("encoder"):
            return "regression_mse"

        return "episode_reward"

    def _resolved_best_mode(self, metric: str) -> str:
        # Use the user-specified mode if provided.
        if self.best_mode is not None:
            return self.best_mode

        # Otherwise infer it from the metric.
        if metric in {"regression_mse", "regression_loss", "uncertainty_loss"}:
            return "min"

        return "max"

    def _current_metric_value(self):
        metric = self._resolved_best_metric()

        if metric == "regression_mse":
            return getattr(self.model, "_last_regression_mse", None)
        if metric == "regression_loss":
            return getattr(self.model, "_last_regression_loss", None)
        if metric == "uncertainty_loss":
            return getattr(self.model, "_last_regression_uncertainty", None)
        if metric == "episode_reward":
            infos = getattr(self, "locals", {}).get("infos", [])
            values = [float(info["episode"]["r"]) for info in infos if "episode" in info]
            return values[-1] if values else None

        raise ValueError(f"Unsupported best_metric={metric!r}")

    def _maybe_save_best(self) -> None:
        metric = self._resolved_best_metric()
        mode = self._resolved_best_mode(metric)
        if self._best_value is None and self.save_dir is not None:
            best_metric_path = os.path.join(self.save_dir, f"{self.save_best_name}.metric")
            if os.path.exists(best_metric_path):
                try:
                    payload = json.load(open(best_metric_path, "r", encoding="utf-8"))
                    if isinstance(payload, dict) and payload.get("metric_name") == metric and payload.get("mode") == mode:
                        self._best_value = float(payload["best_value"])
                except Exception:
                    pass

        # Unlike episode_reward, regression metrics only change on training updates.
        if metric != "episode_reward":
            n_updates = getattr(self.model, "_n_updates", None)
            if n_updates is not None:
                if n_updates == self._last_seen_n_updates:
                    return
                self._last_seen_n_updates = n_updates

        value = self._current_metric_value()
        if value is None:
            return

        is_better = (
            self._best_value is None
            or (mode == "min" and value < self._best_value)
            or (mode == "max" and value > self._best_value)
        )

        if is_better:
            self._best_value = float(value)
            self._save_model(self.save_best_name)
            with open(os.path.join(self.save_dir, f"{self.save_best_name}.metric"), "w", encoding="utf-8") as f:
                json.dump({"best_value": self._best_value, "metric_name": metric, "mode": mode}, f)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # SB3 only adds "episode" to infos when an episode ends.
            if "episode" in info:
                self._episode_count += 1

                # Save periodically based on episode count.
                if self.save_every_episodes > 0 and self._episode_count % self.save_every_episodes == 0:
                    self._save_latest()

        # Always check whether a new best checkpoint should be saved.
        self._maybe_save_best()
        return True

    def _on_training_end(self) -> None:
        if self.save_on_training_end:
            self._save_latest()
            self._maybe_save_best()
