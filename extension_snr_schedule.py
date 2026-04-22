"""extension_snr_schedule.py — Extension: Schedule design by formal SNR criterion.

BACKGROUND:
  The standard linear schedule spaces BETA values uniformly:
      beta(t) = beta_min + (beta_max - beta_min) * t / (L-1)

  This means SNR drops RAPIDLY at first (t=0..200) and slowly at the end.
  From the schedule_linear.png visual: SNR drops from ~10000 to ~1 in the
  first 300 steps, then from ~1 to ~0.0001 in the remaining 700 steps.
  This is highly NON-UNIFORM — most denoising budget is spent in "easy" regime.

FORMAL CRITERION: Equal SNR drop per step
  Design {beta_t} such that SNR(t) decreases by EQUAL amounts each step:
      SNR(t) = SNR_max - t * (SNR_max - SNR_min) / (L - 1)

  Then:
      alpha_bar(t) = SNR(t) / (1 + SNR(t))    [from SNR = ab/(1-ab)]
      beta(t) = 1 - alpha_bar(t) / alpha_bar(t-1)

  This distributes denoising difficulty uniformly across all 1000 steps.

ALTERNATIVE CRITERION: Equal log-SNR drop per step (geometric spacing)
  log-SNR(t) = log(SNR_max) + t * (log(SNR_min) - log(SNR_max)) / (L-1)
  SNR(t) = exp(log-SNR(t))

  This is equivalent to spacing SNR geometrically (multiplicative steps).
  Each step is equally hard on a log scale — matches how humans perceive
  noise (logarithmic sensitivity).

PRE-ABLATION HYPOTHESIS:
  Equal log-SNR spacing will outperform linear schedule because:
  - The model gets equal training signal across all denoising "regimes"
  - Linear schedule wastes steps in the low-SNR (pure noise) region
  - More uniform difficulty → better generalization

  We expect: cleaner samples at intermediate denoising steps (t=500),
  possibly lower FID. The benefit may be marginal for 28x28 images.

Usage:
    python extension_snr_schedule.py
    python extension_snr_schedule.py --steps 100000
"""
import os, sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import math
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

from diffusion.schedule  import NoiseSchedule, make_beta_schedule
from diffusion.forward   import q_sample, sample_timesteps
from diffusion.ddpm      import ancestral_sample
from models.unet         import UNet


# ─────────────────────────────────────────────────────────────────────────────
# Custom schedule: equal SNR drop
# ─────────────────────────────────────────────────────────────────────────────

def make_equal_snr_schedule(L: int, snr_max: float = 999.0, snr_min: float = 1e-4) -> torch.Tensor:
    """Beta schedule where SNR(t) decreases by equal amounts per step.

    Formal criterion: uniform denoising difficulty in SNR space.

    SNR(t) = snr_max - t * (snr_max - snr_min) / (L - 1)
    ᾱ(t) = SNR(t) / (1 + SNR(t))
    β(t) = 1 - ᾱ(t) / ᾱ(t-1),  with ᾱ(-1) := 1

    Args:
        L       : number of timesteps
        snr_max : SNR at t=0 (matched to linear schedule's SNR[0])
        snr_min : SNR at t=L-1

    Returns:
        betas : tensor of shape (L,)
    """
    t = torch.arange(L, dtype=torch.float64)
    snr = snr_max - t * (snr_max - snr_min) / (L - 1)  # linear in SNR space
    snr = snr.clamp(min=snr_min)

    alpha_bar = snr / (1.0 + snr)                        # ᾱ_t = SNR/(1+SNR)

    # ᾱ_{t-1}, with ᾱ_{-1} = 1
    alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bar[:-1]])

    betas = (1.0 - alpha_bar / alpha_bar_prev).clamp(min=1e-5, max=0.9999)
    return betas.float()


def make_log_snr_schedule(L: int, snr_max: float = 999.0, snr_min: float = 1e-4) -> torch.Tensor:
    """Beta schedule where log-SNR(t) decreases by equal amounts per step.

    Formal criterion: uniform denoising difficulty in LOG-SNR space.
    Each step multiplies SNR by the same factor — geometric spacing.

    log-SNR(t) = log(snr_max) + t * (log(snr_min) - log(snr_max)) / (L-1)
    SNR(t) = exp(log-SNR(t))

    Args:
        L       : number of timesteps
        snr_max : SNR at t=0
        snr_min : SNR at t=L-1

    Returns:
        betas : tensor of shape (L,)
    """
    t = torch.arange(L, dtype=torch.float64)
    log_snr = math.log(snr_max) + t * (math.log(snr_min) - math.log(snr_max)) / (L - 1)
    snr = torch.exp(log_snr)

    alpha_bar = snr / (1.0 + snr)
    alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bar[:-1]])

    betas = (1.0 - alpha_bar / alpha_bar_prev).clamp(min=1e-5, max=0.02)
    return betas.float()


def make_custom_noise_schedule(betas: torch.Tensor, device: str) -> NoiseSchedule:
    """Build a NoiseSchedule from a custom beta tensor by monkey-patching."""
    schedule = NoiseSchedule.__new__(NoiseSchedule)
    schedule.L = len(betas)
    schedule.device = device

    betas   = betas.to(device)
    alphas  = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_prev = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])

    posterior_variance = (1.0 - alpha_bars_prev) / (1.0 - alpha_bars) * betas
    posterior_log_var  = torch.log(torch.clamp(posterior_variance, min=1e-20))
    posterior_mean_coef1 = torch.sqrt(alpha_bars_prev) * betas / (1.0 - alpha_bars)
    posterior_mean_coef2 = torch.sqrt(alphas) * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    betas_div_sqrt_1mab  = betas / torch.sqrt(1.0 - alpha_bars)
    snr = alpha_bars / (1.0 - alpha_bars)

    schedule.betas                       = betas.float()
    schedule.alphas                      = alphas.float()
    schedule.alpha_bars                  = alpha_bars.float()
    schedule.alpha_bars_prev             = alpha_bars_prev.float()
    schedule.sqrt_alpha_bars             = torch.sqrt(alpha_bars).float()
    schedule.sqrt_one_minus_alpha_bars   = torch.sqrt(1.0 - alpha_bars).float()
    schedule.recip_sqrt_alphas           = (1.0 / torch.sqrt(alphas)).float()
    schedule.betas_div_sqrt_one_minus_ab = betas_div_sqrt_1mab.float()
    schedule.posterior_variance          = posterior_variance.float()
    schedule.posterior_log_var_clipped   = posterior_log_var.float()
    schedule.posterior_mean_coef1        = posterior_mean_coef1.float()
    schedule.posterior_mean_coef2        = posterior_mean_coef2.float()
    schedule.snr                         = snr.float()

    return schedule


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_grid(images, path, nrow=8, title=""):
    images = (images.clamp(-1, 1) + 1) / 2.0
    grid   = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
    np_img = grid.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(nrow * 1.2, (images.shape[0] // nrow + 1) * 1.2))
    ax.imshow(np_img[:, :, 0] if np_img.shape[-1] == 1 else np_img, cmap="gray")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_snr_schedule_extension(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = "outputs/extensions/snr_schedule"
    os.makedirs(out_dir, exist_ok=True)

    L = 1000

    # ── Build all three schedules ─────────────────────────────────────────────
    linear_betas   = make_beta_schedule(L, "linear", 1e-4, 0.02)
    equal_snr_betas = make_equal_snr_schedule(L, snr_max=999.0, snr_min=1e-4)
    log_snr_betas  = make_log_snr_schedule(L, snr_max=999.0, snr_min=1e-4)

    schedules = {
        "linear":    make_custom_noise_schedule(linear_betas,    device),
        "equal_snr": make_custom_noise_schedule(equal_snr_betas, device),
        "log_snr":   make_custom_noise_schedule(log_snr_betas,   device),
    }

    # ── Plot schedule comparison ──────────────────────────────────────────────
    ts = np.arange(L)
    colors = {"linear": "blue", "equal_snr": "orange", "log_snr": "green"}
    labels = {
        "linear":    "Linear beta (baseline)",
        "equal_snr": "Equal SNR drop (linear-SNR)",
        "log_snr":   "Equal log-SNR drop (geometric)"
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    for name, sch in schedules.items():
        axes[0].plot(ts, sch.alpha_bars.cpu().numpy(),
                     label=labels[name], color=colors[name])
        axes[1].semilogy(ts, sch.snr.cpu().numpy(),
                         label=labels[name], color=colors[name])
        axes[2].plot(ts, sch.betas.cpu().numpy(),
                     label=labels[name], color=colors[name])

    axes[0].set_title("Cumulative ᾱ_t")
    axes[0].set_xlabel("t")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("SNR(t)  [log scale]")
    axes[1].set_xlabel("t")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("β_t")
    axes[2].set_xlabel("t")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Schedule comparison: formal SNR criteria vs linear baseline")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "schedule_comparison.png"), dpi=100)
    plt.close()
    print(f"Schedule comparison saved → {out_dir}/schedule_comparison.png")

    # ── Verify equal-SNR property ─────────────────────────────────────────────
    for name, sch in schedules.items():
        snr = sch.snr.cpu().numpy()
        drops = np.diff(snr)   # SNR(t) - SNR(t-1), should be constant for equal_snr
        print(f"\n  {name}:")
        print(f"    SNR drop/step: min={drops.min():.4f}  max={drops.max():.4f}  "
              f"std={drops.std():.4f}  (low std = uniform drops)")
        print(f"    SNR range: [{snr.min():.6f}, {snr.max():.1f}]")

    # ── Dataset ───────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    ds     = torchvision.datasets.FashionMNIST("./data", train=True,
                                               download=True, transform=transform)
    loader = DataLoader(ds, batch_size=128, shuffle=True,
                        num_workers=0, drop_last=True)

    # ── Train one model per schedule ──────────────────────────────────────────
    results = {}

    for sname, schedule in schedules.items():
        print(f"\n  Training with {sname} schedule for {args.steps} steps...")

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
            t   = sample_timesteps(x0.shape[0], L, device)
            eps = torch.randn_like(x0)
            x_t = q_sample(x0, t, eps, schedule)
            eps_hat = model(x_t, t)
            loss = F.mse_loss(eps_hat, eps)

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

        save_grid(
            result["x0"],
            os.path.join(out_dir, f"samples_{sname}.png"),
            title=f"{labels[sname]}  (step {args.steps})"
        )
        results[sname] = {"losses": losses, "color": colors[sname]}

    # ── Training curves comparison ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for sname, info in results.items():
        smooth = np.convolve(info["losses"], np.ones(200) / 200, mode="valid")
        ax.plot(smooth, label=labels[sname], color=info["color"])
    ax.set_xlabel("Step")
    ax.set_ylabel("L_simple (MSE)")
    ax.set_title(f"SNR schedule comparison — training curves  ({args.steps} steps)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=100)
    plt.close()
    print(f"\n  Training curves saved → {out_dir}/training_curves.png")
    print("\nPost-ablation: compare samples_linear.png vs samples_equal_snr.png vs samples_log_snr.png")
    print("If hypothesis holds: equal log-SNR should show better global structure.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schedule design by SNR criterion")
    parser.add_argument("--steps", type=int, default=50_000)
    args = parser.parse_args()
    run_snr_schedule_extension(args)
