"""extension_timestep_weighting.py — Extension: Principled timestep weighting.

BACKGROUND:
  The standard L_simple loss uses UNIFORM weighting over timesteps:
      L_simple = E_t [ ||eps - eps_hat||^2 ]

  But the full VLB (variational lower bound) assigns different weights to
  each timestep based on the SNR:
      L_vlb = E_t [ w(t) * ||eps - eps_hat||^2 ]

  where the ELBO-derived weight is:
      w(t) = SNR(t-1) - SNR(t)   (drops in signal-to-noise ratio per step)
           = ᾱ_{t-1}/(1-ᾱ_{t-1})  -  ᾱ_t/(1-ᾱ_t)

  This gives HIGH weight to low-t (low noise) steps — where SNR is large —
  and LOW weight to high-t (high noise) steps — where prediction is random.

  Problem: this weighting is very unstable — SNR at t=0 can be ~10000
  while at t=999 it is ~0.00004. A 250,000x range makes training unstable.

  Min-SNR-gamma weighting (Hang et al., 2022) clips the weights:
      w_gamma(t) = min(SNR(t), gamma) / SNR(t)
      where gamma=5 is recommended.
  This down-weights easy (low-t) steps while up-weighting hard (high-t) steps.

PRE-ABLATION HYPOTHESIS:
  Min-SNR weighting (gamma=5) will produce better sample quality at equal
  training steps because:
  - L_simple under-trains high-t steps (hard, high noise) relative to low-t
  - Min-SNR increases the gradient signal from high-t steps
  - The model learns to denoise from pure noise more reliably

  We expect: similar or lower FID, sharper global structure at intermediate t.

  Counter-hypothesis: for FashionMNIST at 28x28, L_simple may already be
  sufficient — the task is simple enough that weighting makes no difference.

Usage:
    python extension_timestep_weighting.py
    python extension_timestep_weighting.py --steps 100000 --gamma 5.0
"""
import os, sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion.schedule  import NoiseSchedule
from diffusion.forward   import q_sample, sample_timesteps
from diffusion.ddpm      import ancestral_sample
from models.unet         import UNet


# ─────────────────────────────────────────────────────────────────────────────
# Weighting functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_min_snr_weights(schedule: NoiseSchedule, gamma: float) -> torch.Tensor:
    """Compute Min-SNR-gamma weights for all timesteps.

    w(t) = min(SNR(t), gamma) / SNR(t)
         = min(1, gamma / SNR(t))

    Properties:
      - At t near 0  (SNR >> gamma): w(t) = gamma/SNR(t) << 1  → down-weighted
      - At t near 999 (SNR << gamma): w(t) = 1                  → full weight
      - gamma controls the transition point

    Args:
        schedule : NoiseSchedule with precomputed SNR
        gamma    : clipping threshold (Hang et al. recommend 5)

    Returns:
        weights : tensor of shape (L,), values in (0, 1]
    """
    snr = schedule.snr  # (L,), values from ~10000 down to ~0.00004
    weights = torch.clamp(snr, max=gamma) / snr  # min(SNR, gamma) / SNR
    return weights


def compute_elbo_weights(schedule: NoiseSchedule) -> torch.Tensor:
    """Compute ELBO-derived (VLB) weights.

    w(t) = SNR(t-1) - SNR(t)   (change in SNR per step)

    Note: these can be very large at low t and tiny at high t.
    We normalize by mean so the total expected loss is comparable to L_simple.

    Returns:
        weights : tensor of shape (L,), normalized so mean = 1
    """
    snr      = schedule.snr            # (L,)
    snr_prev = torch.cat([
        torch.tensor([1e6], device=snr.device),  # SNR at t=-1 is infinite (no noise)
        snr[:-1]
    ])
    weights = (snr_prev - snr).clamp(min=0)  # must be non-negative
    weights = weights / weights.mean()        # normalize: mean weight = 1
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Weighted MSE loss
# ─────────────────────────────────────────────────────────────────────────────

def weighted_mse_loss(
    eps_hat: torch.Tensor,
    eps: torch.Tensor,
    t: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sample weighted MSE loss.

    L = E_t [ w(t) * ||eps - eps_hat||^2 ]

    Args:
        eps_hat : predicted noise, shape (B, C, H, W)
        eps     : true noise,      shape (B, C, H, W)
        t       : timesteps,       shape (B,)
        weights : per-timestep weights, shape (L,)

    Returns:
        scalar loss
    """
    # Per-pixel MSE: (B, C, H, W) → (B,) via mean over spatial dims
    per_sample_mse = (eps_hat - eps).pow(2).mean(dim=[1, 2, 3])  # (B,)

    # Per-sample weight from timestep
    w = weights[t]  # (B,)

    return (w * per_sample_mse).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_with_weighting(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = "outputs/extensions/timestep_weighting"
    os.makedirs(out_dir, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    ds     = torchvision.datasets.FashionMNIST("./data", train=True,
                                               download=True, transform=transform)
    loader = DataLoader(ds, batch_size=128, shuffle=True,
                        num_workers=0, drop_last=True)

    # ── Schedule ──────────────────────────────────────────────────────────────
    schedule = NoiseSchedule(L=1000, schedule_type="linear", device=str(device))

    # ── Precompute weights ────────────────────────────────────────────────────
    uniform_weights  = torch.ones(1000, device=device)
    minsnr_weights   = compute_min_snr_weights(schedule, gamma=args.gamma).to(device)
    elbo_weights     = compute_elbo_weights(schedule).to(device)

    print(f"\nWeight statistics:")
    print(f"  Uniform   : min={uniform_weights.min():.4f}  max={uniform_weights.max():.4f}  mean={uniform_weights.mean():.4f}")
    print(f"  Min-SNR-{args.gamma} : min={minsnr_weights.min():.4f}  max={minsnr_weights.max():.4f}  mean={minsnr_weights.mean():.4f}")
    print(f"  ELBO      : min={elbo_weights.min():.4f}  max={elbo_weights.max():.4f}  mean={elbo_weights.mean():.4f}")

    # ── Plot weight profiles ──────────────────────────────────────────────────
    ts = np.arange(1000)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(ts, uniform_weights.cpu().numpy(),  label="Uniform (L_simple)",  color="blue")
    axes[0].plot(ts, minsnr_weights.cpu().numpy(),   label=f"Min-SNR-{args.gamma}", color="orange")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("w(t)")
    axes[0].set_title("Timestep weights  (linear scale)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(ts, uniform_weights.cpu().numpy(),  label="Uniform",           color="blue")
    axes[1].semilogy(ts, minsnr_weights.cpu().numpy(),   label=f"Min-SNR-{args.gamma}", color="orange")
    axes[1].semilogy(ts, elbo_weights.cpu().numpy(),     label="ELBO",              color="green")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("w(t)  [log scale]")
    axes[1].set_title("Timestep weights  (log scale — shows ELBO range)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Timestep weighting schemes  (gamma={args.gamma})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "weight_profiles.png"), dpi=100)
    plt.close()
    print(f"  Weight profiles saved → {out_dir}/weight_profiles.png")

    # ── Train models ──────────────────────────────────────────────────────────
    results = {}
    configs = [
        ("uniform",         uniform_weights,  "blue"),
        (f"minsnr_g{args.gamma}", minsnr_weights, "orange"),
    ]

    for name, weights, color in configs:
        print(f"\n  Training with {name} weighting for {args.steps} steps...")
        torch.manual_seed(42)
        model = UNet(in_channels=1, base_channels=32,
                     channel_mult=(1, 2, 4), time_emb_dim=256).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=2e-4)

        loader_iter = iter(loader)
        losses = []

        for step in range(1, args.steps + 1):
            model.train()
            try:
                x0, _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x0, _ = next(loader_iter)

            x0  = x0.to(device)
            t   = sample_timesteps(x0.shape[0], 1000, device)
            eps = torch.randn_like(x0)
            x_t = q_sample(x0, t, eps, schedule)
            eps_hat = model(x_t, t)

            # Key difference: weighted vs unweighted MSE
            loss = weighted_mse_loss(eps_hat, eps, t, weights)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

            if step % 10000 == 0 or step == args.steps:
                avg = np.mean(losses[-500:])
                print(f"    step {step:6d}/{args.steps}  loss={avg:.4f}")

        # Generate samples
        model.eval()
        with torch.no_grad():
            result = ancestral_sample(model, schedule,
                                      shape=(64, 1, 28, 28), device=device)

        path = os.path.join(out_dir, f"samples_{name}.png")
        imgs = (result["x0"].clamp(-1, 1) + 1) / 2.0
        grid = torchvision.utils.make_grid(imgs, nrow=8, padding=2)
        np_grid = grid.permute(1, 2, 0).cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(np_grid[:, :, 0], cmap="gray")
        ax.axis("off")
        ax.set_title(f"Weighting: {name}  (step {args.steps})", fontsize=10)
        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"    Samples saved → {path}")

        results[name] = {"losses": losses, "color": color}

    # ── Training curves comparison ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, info in results.items():
        smooth = np.convolve(info["losses"], np.ones(200) / 200, mode="valid")
        ax.plot(smooth, label=name, color=info["color"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Weighted loss")
    ax.set_title(f"Timestep weighting comparison  ({args.steps} steps)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=100)
    plt.close()
    print(f"\n  Training curves saved → {out_dir}/training_curves.png")
    print("\nPost-ablation: compare samples_uniform.png vs samples_minsnr_*.png")
    print("If hypothesis holds: Min-SNR samples show better global structure.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPM with timestep weighting")
    parser.add_argument("--steps", type=int,   default=50_000)
    parser.add_argument("--gamma", type=float, default=5.0,
                        help="Min-SNR clipping threshold (Hang et al. recommend 5)")
    args = parser.parse_args()
    train_with_weighting(args)
