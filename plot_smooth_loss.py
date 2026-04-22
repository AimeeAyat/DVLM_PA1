"""plot_smooth_loss.py — Simulate steady-state loss and plot smoothed curve.

Loads the final checkpoint, runs N forward passes on FashionMNIST val set
with random timesteps (same as training), records per-step loss, then plots:
  - raw loss (light, same oscillation pattern as training)
  - smoothed moving average (window=500)

NOTE: This shows the STEADY-STATE loss distribution, not the historical
training curve. The early drop (steps 0→5000) is not recoverable since
raw losses were not saved in checkpoints.

Usage:
    python plot_smooth_loss.py --checkpoint outputs/checkpoints/ckpt_step0100000.pt
    python plot_smooth_loss.py --checkpoint outputs/checkpoints/ckpt_step0100000.pt --n_steps 5000
"""
import argparse
import sys
import os

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diffusion.schedule import NoiseSchedule
from diffusion.forward  import q_sample, sample_timesteps
from models.unet        import UNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_steps",    type=int, default=3000,
                   help="Number of forward passes to simulate")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--L",          type=int, default=1000)
    p.add_argument("--schedule",   default="linear")
    p.add_argument("--beta_min",   type=float, default=1e-4)
    p.add_argument("--beta_max",   type=float, default=0.02)
    p.add_argument("--base_ch",    type=int,   default=32)
    p.add_argument("--ch_mult",    nargs="+",  type=int, default=[1, 2, 4])
    p.add_argument("--time_emb",   type=int,   default=256)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--data_dir",   default="./data")
    p.add_argument("--out",        default="outputs/loss_curve_smoothed.png")
    p.add_argument("--window",     type=int,   default=500,
                   help="Moving average window size")
    return p.parse_args()


def moving_average(values, window):
    """Simple centered moving average using numpy convolution."""
    kernel = np.ones(window) / window
    # 'same' mode; edges will have partial windows — replace with nan
    smoothed = np.convolve(values, kernel, mode='same')
    half = window // 2
    smoothed[:half] = np.nan
    smoothed[-half:] = np.nan
    return smoothed


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    print(f"[Load] {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Override args with saved training args if available
    saved_args = ckpt.get("args", {})
    for key in ("L", "schedule", "beta_min", "beta_max",
                "base_ch", "ch_mult", "time_emb", "dropout"):
        if key in saved_args:
            setattr(args, key, saved_args[key])

    # ── Build schedule + model ─────────────────────────────────────────────────
    schedule = NoiseSchedule(
        schedule=args.schedule,
        L=args.L,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        device=device,
    )

    model = UNet(
        in_ch=1,
        base_ch=args.base_ch,
        ch_mult=tuple(args.ch_mult),
        time_emb_dim=args.time_emb,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Model] Loaded from step {ckpt.get('step', '?')}")

    # ── FashionMNIST val set ───────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),   # → [-1, 1]
    ])
    val_set = torchvision.datasets.FashionMNIST(
        args.data_dir, train=False, download=True, transform=transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    val_iter = iter(val_loader)

    # ── Simulate N forward passes ──────────────────────────────────────────────
    losses = []
    print(f"[Simulate] Running {args.n_steps} forward passes...")
    with torch.no_grad():
        for step in range(args.n_steps):
            try:
                x0, _ = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                x0, _ = next(val_iter)

            x0 = x0.to(device)
            t   = sample_timesteps(x0.shape[0], args.L, device)
            eps = torch.randn_like(x0)
            x_t = q_sample(x0, t, eps, schedule)
            eps_hat = model(x_t, t)
            loss = F.mse_loss(eps_hat, eps)
            losses.append(loss.item())

            if (step + 1) % 500 == 0:
                print(f"  step {step+1}/{args.n_steps}  loss={loss.item():.4f}")

    losses = np.array(losses)
    smoothed = moving_average(losses, args.window)

    # ── Plot ───────────────────────────────────────────────────────────────────
    steps = np.arange(len(losses))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: raw + smoothed overlay
    ax = axes[0]
    ax.plot(steps, losses, color="#90b8d8", linewidth=0.6, alpha=0.7, label="Raw loss")
    ax.plot(steps, smoothed, color="#d62728", linewidth=2.0, label=f"Moving avg (w={args.window})")
    ax.set_xlabel("Simulated step")
    ax.set_ylabel("L_simple (MSE)")
    ax.set_title("Steady-state loss: raw vs smoothed")
    ax.legend()
    ax.grid(True, alpha=0.3)
    mean_val = np.nanmean(smoothed)
    ax.axhline(mean_val, color="black", linewidth=1.0, linestyle="--",
               label=f"Mean = {mean_val:.4f}")
    ax.legend()

    # Right: loss histogram to show distribution
    ax2 = axes[1]
    ax2.hist(losses, bins=60, color="#4878cf", edgecolor="white", linewidth=0.3)
    ax2.axvline(np.mean(losses), color="#d62728", linewidth=2.0,
                label=f"Mean = {np.mean(losses):.4f}")
    ax2.axvline(np.median(losses), color="orange", linewidth=2.0,
                linestyle="--", label=f"Median = {np.median(losses):.4f}")
    ax2.set_xlabel("L_simple (MSE)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Loss distribution (why oscillations exist)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"\n[Done] Saved → {args.out}")
    print(f"       Mean loss = {np.mean(losses):.4f}")
    print(f"       Std  loss = {np.std(losses):.4f}")
    print(f"       Range     = [{np.min(losses):.4f}, {np.max(losses):.4f}]")


if __name__ == "__main__":
    main()
