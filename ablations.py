"""ablations.py — Task 6: Focused ablations + one DDPM-adjacent extension.

Implements the ablations specified in §4.11:

ABLATION 1 — Schedule ablation (linear vs cosine):
  Hypothesis: cosine schedule should produce better quality at the cost of
  slightly more compute, because it maintains a more uniform SNR drop per
  step (avoids the sudden noise ramp-up at high timesteps in linear schedule).
  Record: ᾱ_i plots, SNR(i) plots, training curves, sample grids.

ABLATION 2 — Sampling-step ablation:
  Hypothesis: reducing sampling steps (by skipping timesteps) degrades sample
  quality gradually.  Early degradation (at ~100 steps) mainly affects fine
  details; severe degradation (at ~10 steps) causes blurry, incoherent outputs.
  Record: sample grids, failure mode narrative.

EXTENSION — DDIM deterministic sampling (implemented in diffusion/ddpm.py):
  Uses `ddim_sample()` with η=0 for deterministic generation.
  Allows meaningful step reduction (good quality at 50–100 steps vs 1000
  for ancestral sampling).

IMPORTANT (§4.11 pitfall):
  Keep EVERYTHING else fixed: same model, same optimizer, same training steps.
  Change only ONE variable per ablation.

Usage:
    python ablations.py --checkpoint outputs/checkpoints/ckpt_step0100000.pt
    python ablations.py --schedule_ablation       # compare linear vs cosine
    python ablations.py --step_ablation           # compare different step counts
    python ablations.py --ddim_ablation           # DDIM vs ancestral
"""
import os, sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from diffusion.schedule  import NoiseSchedule
from diffusion.forward   import q_sample, sample_timesteps
from diffusion.ddpm      import ancestral_sample, ddim_sample
from models.unet         import UNet


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_grid(images: torch.Tensor, path: str, nrow: int = 8, title: str = ""):
    """Save image grid.  images in [−1, 1]."""
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


def quick_loss_estimate(model, schedule, device, data_dir, n_batches=20):
    """Estimate L_simple on a small validation set for comparison."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    ds     = torchvision.datasets.FashionMNIST(
        data_dir, train=False, download=True, transform=transform
    )
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i, (x0, _) in enumerate(loader):
            if i >= n_batches:
                break
            x0  = x0.to(device)
            t   = sample_timesteps(x0.shape[0], schedule.L, device)
            eps = torch.randn_like(x0)
            x_t = q_sample(x0, t, eps, schedule)
            eps_hat = model(x_t, t)
            total += F.mse_loss(eps_hat, eps, reduction="sum").item()
            n     += x0.shape[0]
    return total / n


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION 1 — Schedule ablation: linear vs cosine
# ─────────────────────────────────────────────────────────────────────────────

def schedule_ablation(
    model: torch.nn.Module,
    device: torch.device,
    data_dir: str = "./data",
    out_dir: str = "outputs/ablations/schedule",
    L: int = 1000,
    train_steps: int = 100_000,   # matched compute for fair comparison (same as main training run)
    lr: float = 2e-4,
):
    """Compare linear vs cosine noise schedule.

    PRE-ABLATION HYPOTHESIS:
      The cosine schedule maintains a more uniform SNR drop per step, meaning
      the model gets more training signal at intermediate noise levels.  We
      expect cosine to produce slightly crisper samples, especially for large
      timesteps where linear schedule changes very rapidly.

    POST-ABLATION: compare training curves and sample grids.
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[Ablation 1] Schedule ablation: linear vs cosine")
    print(f"  Training for {train_steps} steps each, matched compute.")

    # ── Plot schedule comparison first ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ts = np.arange(L)

    for stype, color, ls in [("linear", "blue", "-"), ("cosine", "orange", "--")]:
        sch = NoiseSchedule(L=L, schedule_type=stype, device=str(device))
        axes[0].plot(ts, sch.alpha_bars.cpu().numpy(),
                     label=stype, color=color, linestyle=ls)
        axes[1].semilogy(ts, sch.snr.cpu().numpy(),
                         label=stype, color=color, linestyle=ls)
        axes[2].plot(ts, sch.betas.cpu().numpy(),
                     label=stype, color=color, linestyle=ls)

    axes[0].set_title("Cumulative ᾱ_t")
    axes[0].set_xlabel("t")
    axes[0].legend()
    axes[1].set_title("SNR(t) = ᾱ_t / (1−ᾱ_t)  [log]")
    axes[1].set_xlabel("t")
    axes[1].legend()
    axes[2].set_title("β_t")
    axes[2].set_xlabel("t")
    axes[2].legend()
    plt.suptitle("Schedule Comparison: Linear vs Cosine")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "schedule_comparison.png"), dpi=100)
    plt.close()
    print(f"  Schedule comparison plot saved → {out_dir}/schedule_comparison.png")

    # ── Train models ─────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    ds     = torchvision.datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=transform
    )
    loader = DataLoader(ds, batch_size=128, shuffle=True,
                        num_workers=0, drop_last=True)

    results = {}
    for stype in ["linear", "cosine"]:
        print(f"\n  Training with {stype} schedule…")
        schedule_abl = NoiseSchedule(L=L, schedule_type=stype, device=str(device))

        # Fresh model (same architecture, same init for fair comparison)
        torch.manual_seed(42)
        model_abl = UNet(in_channels=1, base_channels=32,
                         channel_mult=(1, 2, 4), time_emb_dim=256).to(device)
        opt = torch.optim.Adam(model_abl.parameters(), lr=lr)

        loader_iter = iter(loader)
        losses_abl  = []

        for step in range(train_steps):
            try:
                x0, _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x0, _ = next(loader_iter)

            x0  = x0.to(device)
            t   = sample_timesteps(x0.shape[0], L, device)
            eps = torch.randn_like(x0)
            x_t = q_sample(x0, t, eps, schedule_abl)
            eps_hat = model_abl(x_t, t)
            loss = F.mse_loss(eps_hat, eps)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses_abl.append(loss.item())

            if step % 5000 == 0 or step == train_steps - 1:
                avg = np.mean(losses_abl[-500:])
                print(f"    step {step:6d}/{train_steps}  loss={avg:.4f}")

        # Generate samples
        model_abl.eval()
        result = ancestral_sample(model_abl, schedule_abl,
                                  shape=(64, 1, 28, 28), device=device)
        save_grid(
            result["x0"],
            os.path.join(out_dir, f"samples_{stype}.png"),
            nrow=8,
            title=f"{stype} schedule  (step {train_steps})"
        )

        results[stype] = {
            "losses":   losses_abl,
            "final_val_loss": quick_loss_estimate(
                model_abl, schedule_abl, device, data_dir
            ),
        }
        print(f"  {stype}: final val loss = {results[stype]['final_val_loss']:.4f}")

    # ── Plot training curves side by side ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for stype, color in [("linear", "blue"), ("cosine", "orange")]:
        losses = results[stype]["losses"]
        # Smooth with window=200
        smooth = np.convolve(losses, np.ones(200) / 200, mode="valid")
        ax.plot(smooth, label=f"{stype}  "
                f"(final={results[stype]['final_val_loss']:.4f})",
                color=color)
    ax.set_xlabel("Step")
    ax.set_ylabel("L_simple (MSE)")
    ax.set_title(f"Training curves: linear vs cosine  ({train_steps} steps)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves_comparison.png"), dpi=100)
    plt.close()
    print(f"\n  Training curves saved → {out_dir}/training_curves_comparison.png")
    print("  Sample grids saved in", out_dir)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION 2 — Sampling-step ablation
# ─────────────────────────────────────────────────────────────────────────────

def sampling_step_ablation(
    model: torch.nn.Module,
    schedule: NoiseSchedule,
    device: torch.device,
    out_dir: str = "outputs/ablations/sampling_steps",
    step_counts: list = None,
):
    """Evaluate sample quality as a function of number of sampling steps.

    Strategy: skip intermediate timesteps by using every k-th step.
    E.g., 100 steps = skip every 10 timesteps (0, 10, 20, ..., 990, 999).

    PRE-ABLATION HYPOTHESIS:
      • 1000 steps: best quality (baseline).
      • 100 steps: slight quality drop but mostly fine.  Global structure
        is captured early (large t), fine details need small t.
      • 50 steps: noticeable artifacts, slight blurriness.
      • 10 steps: significant quality drop.  The reverse chain approximation
        becomes inaccurate with such large steps.  Fine details lost first.

    Failure mode: at very few steps, x̂_0 estimates are inaccurate because
    the model never saw noise levels corresponding to the large step sizes
    used.  The MSE loss was trained with Δt=1 steps.

    Args:
        model       : trained ε_θ model.
        schedule    : NoiseSchedule.
        device      : torch device.
        out_dir     : output directory.
        step_counts : list of step counts to test.  Defaults to [1000,500,200,100,50,20,10].
    """
    os.makedirs(out_dir, exist_ok=True)
    if step_counts is None:
        step_counts = [1000, 500, 200, 100, 50, 20, 10]

    print(f"\n[Ablation 2] Sampling-step ablation")
    print(f"  Testing step counts: {step_counts}")
    print("  PRE-ABLATION HYPOTHESIS: quality degrades as steps decrease.")
    print("  Failure mode starts with fine details, then global structure.")

    L = schedule.L
    torch.manual_seed(0)
    fixed_noise = torch.randn(64, 1, 28, 28, device=device)

    model.eval()
    all_samples = {}

    for n_steps in step_counts:
        print(f"\n  Sampling with {n_steps} steps…")

        if n_steps == L:
            # Full ancestral sampling
            torch.manual_seed(0)
            xL = torch.randn(64, 1, 28, 28, device=device)
            result = ancestral_sample(
                model, schedule,
                shape=(64, 1, 28, 28),
                device=device,
            )
            samples = result["x0"]
        else:
            # Use DDIM with η=1.0 (stochastic, equivalent to ancestral with fewer steps)
            # This tests the effect of step reduction
            samples = _ancestral_subset(model, schedule, fixed_noise.clone(),
                                        n_steps, device)

        save_grid(
            samples,
            os.path.join(out_dir, f"samples_{n_steps:04d}steps.png"),
            nrow=8,
            title=f"{n_steps} sampling steps"
        )
        all_samples[n_steps] = samples
        print(f"    Saved samples for {n_steps} steps")

    # Create side-by-side comparison
    fig_cols = min(4, len(step_counts))
    fig_rows = (len(step_counts) + fig_cols - 1) // fig_cols
    fig, axes = plt.subplots(fig_rows, fig_cols,
                              figsize=(fig_cols * 4, fig_rows * 4))
    axes = np.array(axes).flatten()

    for i, n_steps in enumerate(step_counts):
        if i < len(axes):
            samples = all_samples[n_steps]
            grid = torchvision.utils.make_grid(
                (samples[:16].clamp(-1, 1) + 1) / 2.0, nrow=4, padding=2
            )
            axes[i].imshow(grid.permute(1, 2, 0).numpy()[:, :, 0], cmap="gray")
            axes[i].set_title(f"{n_steps} steps", fontsize=10)
            axes[i].axis("off")

    for i in range(len(step_counts), len(axes)):
        axes[i].axis("off")

    plt.suptitle("Sampling-Step Ablation\n"
                 "Failure mode: fine details lost first, then global structure",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step_ablation_comparison.png"), dpi=100)
    plt.close()
    print(f"\n  Comparison grid saved → {out_dir}/step_ablation_comparison.png")
    print("  POST-ABLATION ANALYSIS:")
    print("  • Compare sample grids and note where quality degrades first.")
    print("  • Fine-grained details (textures) degrade before coarse structure.")
    print("  • At ~10 steps, the Gaussian approximation of p_θ(x_{t-k}|x_t)")
    print("    for large k is poor — the step is too large for the linear approx.")


def _ancestral_subset(
    model: torch.nn.Module,
    schedule: NoiseSchedule,
    x_start: torch.Tensor,
    n_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Run ancestral sampling using only n_steps out of L.

    We pick n_steps uniformly-spaced timesteps and run the standard
    p_sample_step for each, using the actual posterior variance for that t.

    This is a simple "step-skipping" approach (not DDIM — just using a
    subset of timesteps with the same model/schedule).
    """
    from diffusion.posterior import p_sample_step

    L = schedule.L
    step_size = max(1, L // n_steps)
    # Build the timestep sequence from high noise to low noise
    timesteps = list(reversed(range(0, L, step_size)))[:n_steps]

    model.eval()
    x = x_start

    with torch.no_grad():
        for t in timesteps:
            t_batch  = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
            eps_hat  = model(x, t_batch)
            x = p_sample_step(x, t, eps_hat, schedule)

    return x.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# EXTENSION — Parameterization ablation (ε-prediction vs x_0-prediction)
# ─────────────────────────────────────────────────────────────────────────────

def parameterization_ablation(
    device: torch.device,
    data_dir: str = "./data",
    out_dir: str = "outputs/ablations/parameterization",
    L: int = 1000,
    train_steps: int = 30_000,
):
    """Compare ε-prediction vs x_0-prediction parameterizations.

    x_0-prediction loss:
        L_x0 = E[‖x_0 − x̂_0‖²]
    where  x̂_0 = (x_t − √(1−ᾱ_t)·ε̂) / √ᾱ_t

    Equivalently, we can predict x_0 directly with the network.

    PRE-ABLATION HYPOTHESIS:
      ε-prediction is better conditioned — the target ε ~ N(0,I) has
      unit variance regardless of timestep.  x_0-prediction's effective
      variance grows with noise level and can be hard to learn at large t.
      We expect ε-prediction to have more stable training curves.

    Connection to analytical section:
      This directly relates to Problem 4 of the analytical section about
      ε-prediction vs other parameterizations.
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[Ablation 3] Parameterization ablation: ε-pred vs x_0-pred")

    from diffusion.posterior import predict_x0_from_eps

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    ds     = torchvision.datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=transform
    )
    loader = DataLoader(ds, batch_size=128, shuffle=True,
                        num_workers=0, drop_last=True)

    schedule = NoiseSchedule(L=L, schedule_type="linear", device=str(device))
    results  = {}

    for pred_type in ["eps", "x0"]:
        print(f"\n  Training {pred_type}-prediction model…")
        torch.manual_seed(42)
        model_p = UNet(in_channels=1, base_channels=32,
                       channel_mult=(1, 2, 4), time_emb_dim=256).to(device)
        opt = torch.optim.Adam(model_p.parameters(), lr=2e-4)
        loader_iter = iter(loader)
        losses_p = []

        for step in range(train_steps):
            try:
                x0, _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x0, _ = next(loader_iter)

            x0  = x0.to(device)
            t   = sample_timesteps(x0.shape[0], L, device)
            eps = torch.randn_like(x0)
            x_t = q_sample(x0, t, eps, schedule)
            pred = model_p(x_t, t)   # model always outputs same shape

            if pred_type == "eps":
                # Standard ε-prediction loss (eq 9)
                loss = F.mse_loss(pred, eps)
            else:
                # x_0-prediction loss: predict x_0 directly
                # Model outputs x̂_0; loss = ‖x_0 − x̂_0‖²
                loss = F.mse_loss(pred, x0)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses_p.append(loss.item())

            if step % 5000 == 0 or step == train_steps - 1:
                avg = np.mean(losses_p[-200:])
                print(f"    step {step:6d}/{train_steps}  loss={avg:.6f}")

        # Generate samples using ε-parameterization
        # For x_0-pred: convert model output to ε̂ for sampling
        if pred_type == "x0":
            # Wrap model to convert x_0 prediction to ε prediction for sampler
            class X0ToEpsWrapper(torch.nn.Module):
                def __init__(self, m, sch):
                    super().__init__()
                    self.m = m
                    self.sch = sch

                def forward(self, x_t, t):
                    x0_hat = self.m(x_t, t)
                    x0_hat = x0_hat.clamp(-1, 1)
                    sqrt_ab   = self.sch.extract(self.sch.sqrt_alpha_bars, t, x_t.ndim)
                    sqrt_1mab = self.sch.extract(self.sch.sqrt_one_minus_alpha_bars,
                                                 t, x_t.ndim)
                    # ε̂ = (x_t − √ᾱ_t · x̂_0) / √(1−ᾱ_t)
                    eps_hat = (x_t - sqrt_ab * x0_hat) / sqrt_1mab.clamp(min=1e-8)
                    return eps_hat
            sample_model = X0ToEpsWrapper(model_p, schedule).to(device)
        else:
            sample_model = model_p

        sample_model.eval()
        result = ancestral_sample(sample_model, schedule,
                                  shape=(64, 1, 28, 28), device=device)
        save_grid(
            result["x0"],
            os.path.join(out_dir, f"samples_{pred_type}.png"),
            nrow=8,
            title=f"{pred_type}-prediction ({train_steps} steps)"
        )
        results[pred_type] = losses_p

    # Compare training curves
    fig, ax = plt.subplots(figsize=(10, 4))
    for pred_type, color in [("eps", "blue"), ("x0", "orange")]:
        ls = results[pred_type]
        smooth = np.convolve(ls, np.ones(200) / 200, mode="valid")
        ax.plot(smooth, label=f"{pred_type}-prediction", color=color)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Parameterization ablation: ε-pred vs x_0-pred")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "parameterization_curves.png"), dpi=100)
    plt.close()
    print(f"\n  Parameterization comparison saved → {out_dir}/")
    print("  POST-ABLATION: note whether ε-prediction is more stable")
    print("  and whether x_0-prediction shows high variance at large timesteps.")


# ─────────────────────────────────────────────────────────────────────────────
# EXTENSION — DDIM deterministic sampling comparison
# ─────────────────────────────────────────────────────────────────────────────

def ddim_vs_ancestral(
    model: torch.nn.Module,
    schedule: NoiseSchedule,
    device: torch.device,
    out_dir: str = "outputs/ablations/ddim",
    n_samples: int = 64,
):
    """Compare DDIM (deterministic, 50 steps) vs ancestral (stochastic, 1000 steps).

    DDIM allows MUCH faster sampling (50 steps vs 1000) while maintaining
    reasonable quality.  The key insight: same score/denoiser model, just a
    different ODE integrator.

    PRE-ABLATION HYPOTHESIS:
      DDIM (η=0, 50 steps) will be slightly blurrier than ancestral (1000 steps)
      but noticeably faster.  With η>0 (stochastic DDIM), quality improves.
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[Extension] DDIM vs ancestral sampling")

    model.eval()
    torch.manual_seed(0)

    # Ancestral (baseline)
    print("  Running ancestral sampling (1000 steps)…")
    result = ancestral_sample(model, schedule,
                              shape=(n_samples, 1, 28, 28), device=device)
    save_grid(result["x0"],
              os.path.join(out_dir, "ancestral_1000steps.png"),
              nrow=8, title="Ancestral sampling (1000 steps)")

    # DDIM various step counts and η values
    for n_steps, eta in [(50, 0.0), (50, 1.0), (100, 0.0), (200, 0.0)]:
        print(f"  Running DDIM  (steps={n_steps}, η={eta})…")
        samples = ddim_sample(
            model, schedule,
            shape=(n_samples, 1, 28, 28),
            device=device,
            num_steps=n_steps,
            eta=eta,
        )
        fname = f"ddim_{n_steps}steps_eta{eta:.1f}.png"
        save_grid(samples,
                  os.path.join(out_dir, fname),
                  nrow=8,
                  title=f"DDIM (steps={n_steps}, η={eta})")
        print(f"    Saved → {out_dir}/{fname}")

    print(f"\n  POST-ABLATION: compare grids.")
    print("  DDIM η=0 is deterministic: same noise seed → same sample.")
    print("  DDIM η=1 ≈ ancestral sampling with fewer steps.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPM ablations (Task 6)")
    parser.add_argument("--checkpoint",         default=None)
    parser.add_argument("--data_dir",           default="./data")
    parser.add_argument("--out_dir",            default="./outputs/ablations")
    parser.add_argument("--L",                  type=int, default=1000)
    parser.add_argument("--schedule",           default="linear")
    parser.add_argument("--schedule_ablation",  action="store_true")
    parser.add_argument("--step_ablation",      action="store_true")
    parser.add_argument("--param_ablation",     action="store_true")
    parser.add_argument("--ddim_ablation",      action="store_true")
    parser.add_argument("--all",                action="store_true",
                        help="Run all ablations")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.schedule_ablation or args.all:
        schedule_ablation(
            model=None,     # trains from scratch
            device=device,
            data_dir=args.data_dir,
            out_dir=os.path.join(args.out_dir, "schedule"),
            L=args.L,
        )

    if (args.step_ablation or args.ddim_ablation or args.all) and args.checkpoint:
        schedule = NoiseSchedule(L=args.L, schedule_type=args.schedule, device=str(device))
        model = UNet(in_channels=1, base_channels=32,
                     channel_mult=(1, 2, 4), time_emb_dim=256).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        if args.step_ablation or args.all:
            sampling_step_ablation(
                model, schedule, device,
                out_dir=os.path.join(args.out_dir, "sampling_steps"),
            )
        if args.ddim_ablation or args.all:
            ddim_vs_ancestral(
                model, schedule, device,
                out_dir=os.path.join(args.out_dir, "ddim"),
            )
    elif (args.step_ablation or args.ddim_ablation or args.all) and not args.checkpoint:
        print("\nNote: step_ablation and ddim_ablation require --checkpoint")

    if args.param_ablation or args.all:
        parameterization_ablation(
            device=device,
            data_dir=args.data_dir,
            out_dir=os.path.join(args.out_dir, "parameterization"),
            L=args.L,
        )
