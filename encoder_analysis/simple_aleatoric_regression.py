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

ENCODER_TYPE = "temporal_cnn" # "mlp"  # change to "temporal_cnn" to test the other exact encoder path
WINDOW_LENGTH = 50
TRAJ_DIM = 2


def make_dataset(n_samples: int) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    x = th.linspace(-6.0, 6.0, n_samples).unsqueeze(1)

    # Same scalar x copied into every timestep and every feature slot so the
    # encoder sees the exact PPO trajectory shape, but there is no dynamics.
    traj = x[:, None, :].repeat(1, WINDOW_LENGTH, TRAJ_DIM)

    true_mean = 1.5 * x + 0.3
    true_std = 0.15 + 1.35 * (th.sin(x) + 1.0) / 2.0
    y = true_mean + true_std * th.randn_like(true_mean)
    return traj, y, true_mean, true_std, x


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

    # Match UnifiedContextPPO(context_mode="encoder_nll") exactly.
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
    train_traj, train_y, _, _, _ = make_dataset(5000)
    val_traj, val_y, true_mean, true_std, x = make_dataset(2000)

    policy = build_policy()

    for step in range(2000):
        batch_idx = th.randint(0, train_traj.shape[0], (256,))
        traj_batch = train_traj[batch_idx]
        y_batch = train_y[batch_idx]

        mu, logvar = policy.encode_context(traj_batch, return_logvar=True)
        logvar = policy.clamp_logvar(logvar)
        loss = 0.5 * (logvar + (y_batch - mu).pow(2) / logvar.exp()).mean()

        policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        policy.optimizer.step()

        if (step + 1) % 250 == 0:
            print(f"step {step + 1:4d} | train_nll {loss.item():.4f}")

    with th.no_grad():
        pred_mu, pred_logvar = policy.encode_context(val_traj, return_logvar=True)
        pred_logvar = policy.clamp_logvar(pred_logvar)
        pred_std = th.exp(0.5 * pred_logvar)

    mean_rmse = (pred_mu - true_mean).pow(2).mean().sqrt().item()
    std_rmse = (pred_std - true_std).pow(2).mean().sqrt().item()
    nll = 0.5 * (pred_logvar + (val_y - pred_mu).pow(2) / pred_logvar.exp()).mean().item()

    print(f"validation mean_rmse: {mean_rmse:.4f}")
    print(f"validation std_rmse:  {std_rmse:.4f}")
    print(f"validation nll:       {nll:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    order = th.argsort(x.squeeze(1))
    x_sorted = x[order].squeeze(1)
    true_mean_sorted = true_mean[order].squeeze(1)
    true_std_sorted = true_std[order].squeeze(1)
    pred_mu_sorted = pred_mu[order].squeeze(1)
    pred_std_sorted = pred_std[order].squeeze(1)

    axes[0].scatter(x.numpy(), val_y.numpy(), s=8, alpha=0.15, label="noisy targets")
    axes[0].plot(x_sorted.numpy(), true_mean_sorted.numpy(), "k--", label="true mean")
    axes[0].plot(x_sorted.numpy(), pred_mu_sorted.numpy(), color="tab:blue", label="predicted mean")
    axes[0].set_title("Mean Fit")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()

    axes[1].plot(x_sorted.numpy(), true_std_sorted.numpy(), "k--", label="true std")
    axes[1].plot(x_sorted.numpy(), pred_std_sorted.numpy(), color="tab:orange", label="predicted std")
    axes[1].set_title("Uncertainty Envelope")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("std")
    axes[1].legend()

    fig.suptitle(f"Static Heteroscedastic Regression ({ENCODER_TYPE}, exact PPO encoder path)")
    fig.tight_layout()

    output_path = Path(__file__).with_name("simple_aleatoric_regression.png")
    fig.savefig(output_path, dpi=180)
    print(f"saved plot to {output_path}")


if __name__ == "__main__":
    main()
