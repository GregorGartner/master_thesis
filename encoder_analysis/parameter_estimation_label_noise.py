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
WINDOW_LENGTH = 50
TRAJ_DIM = 3


def make_dataset(n_samples: int) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    theta = th.empty(n_samples, 1).uniform_(-0.9, 0.9)
    high_noise = th.rand(n_samples, 1) > 0.5
    true_std = th.where(high_noise, th.full((n_samples, 1), 0.25), th.full((n_samples, 1), 0.05))

    u_sequence = th.sin(th.linspace(0.0, 3.0 * math.pi, WINDOW_LENGTH))
    # x = th.zeros(n_samples, 1)
    x = th.empty(n_samples, 1).uniform_(-1.0, 1.0)
    traj = th.zeros(n_samples, WINDOW_LENGTH, TRAJ_DIM)

    for t in range(WINDOW_LENGTH):
        u_t = u_sequence[t].expand(n_samples, 1)
        traj[:, t, 0:1] = x
        traj[:, t, 1:2] = u_t
        traj[:, t, 2:3] = theta * x + u_t + true_std * th.randn_like(x) - x
        x = x + traj[:, t, 2:3]

    true_mean = theta
    y = true_mean + true_std * th.randn_like(true_mean)
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
    train_traj, train_y, _, _ = make_dataset(100000)
    val_traj, val_y, true_mean, true_std = make_dataset(800)

    policy = build_policy()

    for step in range(5000):
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

    axes[0].scatter(true_mean.numpy(), pred_mu.numpy(), s=10, alpha=0.25)
    lo = min(true_mean.min().item(), pred_mu.min().item())
    hi = max(true_mean.max().item(), pred_mu.max().item())
    axes[0].plot([lo, hi], [lo, hi], "k--")
    axes[0].set_title("Theta")
    axes[0].set_xlabel("True theta")
    axes[0].set_ylabel("Predicted theta")

    axes[1].scatter(true_std.numpy(), pred_std.numpy(), s=10, alpha=0.25)
    lo = min(true_std.min().item(), pred_std.min().item())
    hi = max(true_std.max().item(), pred_std.max().item())
    axes[1].plot([lo, hi], [lo, hi], "k--")
    axes[1].set_title("Uncertainty")
    axes[1].set_xlabel("True std")
    axes[1].set_ylabel("Predicted std")

    fig.suptitle(f"First-Order System ID Sanity Check ({ENCODER_TYPE}, exact PPO encoder path)")
    fig.tight_layout()

    output_path = Path(__file__).with_name("parameter_estimation_label_noise.png")
    fig.savefig(output_path, dpi=180)
    print(f"saved plot to {output_path}")


if __name__ == "__main__":
    main()
