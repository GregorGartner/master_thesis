import math
from pathlib import Path
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th

sys.path.append(str(Path(__file__).resolve().parent.parent))
from unified_context_ppo import ContextActorCriticPolicy


th.manual_seed(0)

ENCODER_TYPE = "mlp" # "temporal_cnn"  # change to "mlp" to test the other exact encoder path
WINDOW_LENGTH = 4
TRAJ_DIM = 2
ENSEMBLE_SIZE = 10


def make_dataset(n_samples: int) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    theta = th.empty(n_samples, 1).uniform_(-0.9, 0.9)
    true_std = th.full((n_samples, 1), 0.6)

    # u_sequence = th.sin(th.linspace(0.0, 3.0 * math.pi, WINDOW_LENGTH))
    x = th.empty(n_samples, 1).uniform_(-0.9, 0.9)
    traj = th.zeros(n_samples, WINDOW_LENGTH, TRAJ_DIM)

    for t in range(WINDOW_LENGTH):
        # u_t = u_sequence[t].expand(n_samples, 1)
        u_t = 0.8 * th.tanh(0.4 + 0.5 * theta - 0.7 * x)
        traj[:, t, 0:1] = x
        traj[:, t, 1:2] = u_t
        x = theta * x + u_t + true_std * th.randn_like(x)

    true_mean = theta
    y = true_mean # + true_std * th.randn_like(true_mean)
    return traj, y, true_mean, true_std


def build_policy() -> ContextActorCriticPolicy:
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    policy = ContextActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        window_length=WINDOW_LENGTH,
        traj_dim=TRAJ_DIM,
        latent_dim=1,
        encoder_type=ENCODER_TYPE,
        encoder_net_arch=[64, 64],
        actor_net_arch=[64, 64],
        critic_net_arch=[64, 64],
        detach_context_for_rl=False,
        condition_on_uncertainty=False,
        device="cpu",
    )

    # Match UnifiedContextPPO(context_mode="encoder_nll") exactly:
    policy.nll_mode = True
    context_encoder = policy._build_context_encoder(policy.latent_dim * 2)
    context_encoder.apply(lambda m: policy._ortho_init_module(m, math.sqrt(2)))
    policy.context_encoder = context_encoder.to(policy.device)

    for name, param in policy.named_parameters():
        if not name.startswith("context_encoder."):
            param.requires_grad_(False)

    policy.optimizer = th.optim.Adam(policy.context_encoder.parameters(), lr=1e-3, eps=1e-5)
    return policy


def main() -> None:
    train_traj, train_y, _, _ = make_dataset(20000)
    val_traj, val_y, true_mean, true_std = make_dataset(2000)

    policies = [build_policy() for _ in range(ENSEMBLE_SIZE)]

    for step in range(20000):
        batch_idx = th.randint(0, train_traj.shape[0], (512,))
        traj_batch = train_traj[batch_idx]
        y_batch = train_y[batch_idx]
        losses = []

        for policy in policies:
            mu, logvar = policy.encode_context(traj_batch, return_logvar=True)
            logvar = policy.clamp_logvar(logvar)
            loss = 0.5 * (logvar + (y_batch - mu).pow(2) / logvar.exp()).mean()

            policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy.optimizer.step()
            losses.append(loss.item())

        if (step + 1) % 250 == 0:
            print(f"step {step + 1:4d} | train_nll {np.mean(losses):.4f}")

    pred_mu_list = []
    pred_std_list = []
    with th.no_grad():
        for policy in policies:
            pred_mu, pred_logvar = policy.encode_context(val_traj, return_logvar=True)
            pred_logvar = policy.clamp_logvar(pred_logvar)
            pred_mu_list.append(pred_mu)
            pred_std_list.append(th.exp(0.5 * pred_logvar))

    pred_mu_stack = th.stack(pred_mu_list, dim=0)
    pred_std_stack = th.stack(pred_std_list, dim=0)
    pred_mu = pred_mu_stack.mean(dim=0)
    aleatoric_var = (pred_std_stack ** 2).mean(dim=0)
    epistemic_var = pred_mu_stack.var(dim=0, unbiased=False)
    aleatoric_std = aleatoric_var.sqrt()
    epistemic_std = epistemic_var.sqrt()
    pred_std = (aleatoric_var + epistemic_var).sqrt()

    mean_rmse = (pred_mu - true_mean).pow(2).mean().sqrt().item()
    pred_logvar = 2.0 * th.log(pred_std)
    nll = 0.5 * (pred_logvar + (val_y - pred_mu).pow(2) / pred_logvar.exp()).mean().item()

    print(f"validation mean_rmse: {mean_rmse:.4f}")
    print(f"validation epistemic: {epistemic_std.mean().item():.4f}")
    print(f"validation nll:       {nll:.4f}")

    fig, axes = plt.subplots(1, 5, figsize=(26, 4))
    abs_error = (pred_mu - true_mean).abs().squeeze(1)

    axes[0].scatter(true_mean.numpy(), pred_mu.numpy(), s=10, alpha=0.25)
    lo = min(true_mean.min().item(), pred_mu.min().item())
    hi = max(true_mean.max().item(), pred_mu.max().item())
    axes[0].plot([lo, hi], [lo, hi], "k--")
    axes[0].set_title("Theta")
    axes[0].set_xlabel("True theta")
    axes[0].set_ylabel("Predicted theta")

    axes[1].scatter(pred_std.numpy(), abs_error.numpy(), s=10, alpha=0.25)
    sigma = pred_std.squeeze(1)
    sigma_max = sigma.max().item()
    sigma_grid = th.linspace(0.0, sigma_max, 200)
    axes[1].plot(
        sigma_grid.numpy(),
        (math.sqrt(2.0 / math.pi) * sigma_grid).numpy(),
        "k--",
    )
    axes[1].set_title("|mu_hat - theta| vs total std")
    axes[1].set_xlabel("Predicted total std")
    axes[1].set_ylabel("|Predicted theta - True theta|")

    axes[2].scatter(epistemic_std.numpy(), abs_error.numpy(), s=10, alpha=0.25)
    axes[2].set_title("|mu_hat - theta| vs epistemic std")
    axes[2].set_xlabel("Epistemic std")
    axes[2].set_ylabel("|Predicted theta - True theta|")

    sorted_idx = th.argsort(sigma)
    binned_sigma = []
    binned_error = []
    for chunk in th.chunk(sorted_idx, 10):
        if chunk.numel() == 0:
            continue
        binned_sigma.append(sigma[chunk].mean().item())
        binned_error.append(abs_error[chunk].mean().item())

    axes[3].plot(binned_sigma, binned_error, marker="o")
    axes[3].plot(
        sigma_grid.numpy(),
        (math.sqrt(2.0 / math.pi) * sigma_grid).numpy(),
        "k--",
    )
    axes[3].set_title("Binned calibration")
    axes[3].set_xlabel("Mean predicted total std")
    axes[3].set_ylabel("Mean |Predicted theta - True theta|")

    ks = th.linspace(0.1, 3.0, 30)
    coverage = []
    ideal_coverage = []
    for k in ks:
        inside = abs_error <= k * sigma
        coverage.append(inside.float().mean().item())
        ideal_coverage.append(math.erf(float(k) / math.sqrt(2.0)))

    axes[4].plot(ks.numpy(), coverage, marker="o", markersize=3)
    axes[4].plot(ks.numpy(), ideal_coverage, "k--")
    axes[4].set_title("Coverage")
    axes[4].set_xlabel("k in mu_hat ± k·total std")
    axes[4].set_ylabel("Empirical coverage")

    fig.suptitle(f"First-Order System ID Sanity Check ({ENCODER_TYPE}, exact PPO encoder path)")
    fig.tight_layout()

    output_path = Path(__file__).with_name("closed_loop_dynamics_parameter_estimation.png")
    fig.savefig(output_path, dpi=180)
    print(f"saved plot to {output_path}")

    uncertainty_map = {
        "total": pred_std.squeeze(1),
        "aleatoric": aleatoric_std.squeeze(1),
        "epistemic": epistemic_std.squeeze(1),
    }
    fig_diag, axes_diag = plt.subplots(3, 3, figsize=(16, 12))
    ks = th.linspace(0.1, 3.0, 30)

    for row_idx, (name, sigma_raw) in enumerate(uncertainty_map.items()):
        sigma_local = th.clamp(sigma_raw, min=0.0)
        sigma_max_local = max(float(sigma_local.max().item()), 1e-8)
        sigma_grid_local = th.linspace(0.0, sigma_max_local, 200)

        ax_err = axes_diag[row_idx, 0]
        ax_err.scatter(sigma_local.numpy(), abs_error.numpy(), s=10, alpha=0.25)
        ax_err.plot(
            sigma_grid_local.numpy(),
            (math.sqrt(2.0 / math.pi) * sigma_grid_local).numpy(),
            "k--",
        )
        ax_err.set_title(f"|mu_hat - theta| vs {name} std")
        ax_err.set_xlabel(f"Predicted {name} std")
        ax_err.set_ylabel("|Predicted theta - True theta|")

        sorted_idx_local = th.argsort(sigma_local)
        binned_sigma_local = []
        binned_error_local = []
        for chunk in th.chunk(sorted_idx_local, 10):
            if chunk.numel() == 0:
                continue
            binned_sigma_local.append(sigma_local[chunk].mean().item())
            binned_error_local.append(abs_error[chunk].mean().item())

        ax_bin = axes_diag[row_idx, 1]
        ax_bin.plot(binned_sigma_local, binned_error_local, marker="o")
        ax_bin.plot(
            sigma_grid_local.numpy(),
            (math.sqrt(2.0 / math.pi) * sigma_grid_local).numpy(),
            "k--",
        )
        ax_bin.set_title(f"Binned calibration ({name})")
        ax_bin.set_xlabel(f"Mean predicted {name} std")
        ax_bin.set_ylabel("Mean |Predicted theta - True theta|")

        coverage_local = []
        ideal_coverage_local = []
        for k in ks:
            inside = abs_error <= k * sigma_local
            coverage_local.append(inside.float().mean().item())
            ideal_coverage_local.append(math.erf(float(k) / math.sqrt(2.0)))

        ax_cov = axes_diag[row_idx, 2]
        ax_cov.plot(ks.numpy(), coverage_local, marker="o", markersize=3)
        ax_cov.plot(ks.numpy(), ideal_coverage_local, "k--")
        ax_cov.set_title(f"Coverage ({name})")
        ax_cov.set_xlabel(f"k in mu_hat ± k·{name} std")
        ax_cov.set_ylabel("Empirical coverage")

    fig_diag.suptitle(
        f"Uncertainty Diagnostics by Component ({ENCODER_TYPE}, exact PPO encoder path)"
    )
    fig_diag.tight_layout()
    diag_output_path = Path(__file__).with_name(
        "closed_loop_dynamics_parameter_estimation_uncertainty_diagnostics.png"
    )
    fig_diag.savefig(diag_output_path, dpi=180)
    print(f"saved plot to {diag_output_path}")


if __name__ == "__main__":
    main()
