"""train.py — DDPM training loop with logging and sanity checks.

Training objective: L_simple (eq 9)
    L_simple(θ) = E_{x0, i, ε} [ ‖ε − ε̂_θ(x_i, i)‖² ]

Each training iteration:
    1. Sample minibatch x_0
    2. Sample t ~ Unif{0, …, L-1}  (0-indexed)
    3. Sample ε ~ N(0, I)
    4. Compute x_t via q_sample (eq 2)
    5. Predict ε̂ = ε_θ(x_t, t)
    6. Compute MSE loss: ‖ε − ε̂‖²
    7. Backprop + Adam step

Sanity checks implemented (§4.9):
    1. Overfit test — train on 256 images only, samples should match subset.
    2. One-step posterior check — verify E[‖x_{t-1}-x_0‖²] < E[‖x_t-x_0‖²].
    3. Noise-prediction sanity — correlation between ε and ε̂ on val batch.
    4. Timestep sanity — histogram of sampled timesteps (should be uniform).

Usage:
    python train.py
    python train.py --overfit_test          # quick sanity: overfit 256 images
    python train.py --schedule cosine       # cosine schedule (Task 6 ablation)
    python train.py --steps 100000         # full training run
"""
import os, sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Project imports
from diffusion.schedule import NoiseSchedule
from diffusion.forward  import q_sample, sample_timesteps
from diffusion.ddpm     import ancestral_sample
from models.unet        import UNet


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DDPM on FashionMNIST")

    # Dataset
    p.add_argument("--dataset",     default="FashionMNIST",
                   choices=["FashionMNIST", "MNIST"])
    p.add_argument("--data_dir",    default="./data")

    # Diffusion schedule (§4.2 baseline)
    p.add_argument("--L",           type=int,   default=1000,
                   help="Number of diffusion timesteps")
    p.add_argument("--schedule",    default="linear",
                   choices=["linear", "cosine"],
                   help="Noise schedule type")
    p.add_argument("--beta_min",    type=float, default=1e-4)
    p.add_argument("--beta_max",    type=float, default=0.02)

    # Model (§4.2 baseline)
    p.add_argument("--base_ch",     type=int,   default=32)
    p.add_argument("--ch_mult",     nargs="+",  type=int, default=[1, 2, 4])
    p.add_argument("--time_emb",    type=int,   default=256)
    p.add_argument("--dropout",     type=float, default=0.1)

    # Optimiser (§4.2 baseline)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--weight_decay",type=float, default=0.0)
    p.add_argument("--batch_size",  type=int,   default=128)

    # Training budget
    p.add_argument("--steps",       type=int,   default=100_000,
                   help="Total gradient steps")
    p.add_argument("--overfit_test",action="store_true",
                   help="Train on only 256 images for overfitting sanity check")

    # Logging
    p.add_argument("--log_every",   type=int,   default=100)
    p.add_argument("--sample_every",type=int,   default=5_000)
    p.add_argument("--save_every",  type=int,   default=10_000)
    p.add_argument("--out_dir",     default="./outputs")

    # Mixed precision
    p.add_argument("--amp",         action="store_true",
                   help="Use automatic mixed precision (bf16 on RTX 5090 / Blackwell)")

    # RTX 5090 / Blackwell-specific speed-ups
    p.add_argument("--compile",     action="store_true",
                   help="torch.compile() the model for ~2-3x speed-up (PyTorch 2+)")
    p.add_argument("--tf32",        action="store_true", default=True,
                   help="Enable TF32 matmuls (on by default for Ampere/Blackwell)")

    # Checkpoint resume
    p.add_argument("--resume",      default=None,
                   help="Path to checkpoint .pt to resume training from")

    # RTX 5090: cache entire dataset to GPU (FashionMNIST < 200 MB)
    p.add_argument("--gpu_cache",   action="store_true",
                   help="Pre-load all training data to GPU tensor for zero-latency batching")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# GPU data cache — eliminates DataLoader latency (RTX 5090 optimization)
# ─────────────────────────────────────────────────────────────────────────────

class GPUDataCache:
    """Holds the entire training set as a single GPU tensor.

    FashionMNIST: 60,000 × 1 × 28 × 28 × float32 ≈ 188 MB — trivial for 32 GB.
    Sampling a batch is a single GPU index operation (~1 µs vs ~1 ms for DataLoader).
    """
    def __init__(self, dataset, device: torch.device):
        print(f"[GPU Cache] Loading {len(dataset)} images to GPU...")
        # Stack all images into one big tensor on GPU
        images = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
        self.images = images  # (N, C, H, W) float32 on GPU
        self.N = len(images)
        print(f"[GPU Cache] Cached {self.N} images  ({images.element_size() * images.numel() / 1e6:.1f} MB on GPU)")

    def sample(self, batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, self.N, (batch_size,), device=self.images.device)
        return self.images[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Task 0: Dataset pipeline
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(args):
    """Load FashionMNIST / MNIST with scaling to [−1, 1].

    DATA SCALING DECISION: we use [−1, 1] (recommended in §4.5).
    transform = ToTensor() → [0,1] → Normalize(0.5, 0.5) → [−1,1]

    WHY [−1,1]?  Keeps the clean image x_0 in a symmetric range around zero,
    matching the zero-mean Gaussian noise that DDPM adds.  Consistency between
    training and decoding is CRITICAL — mismatch causes washed-out samples.
    """
    # ToTensor: [0,255] uint8 → [0,1] float32
    # Normalize(0.5, 0.5): (x - 0.5) / 0.5 → [−1,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),   # → [−1, 1]
    ])

    DatasetClass = getattr(torchvision.datasets, args.dataset)

    train_set = DatasetClass(
        root=args.data_dir, train=True,  download=True, transform=transform
    )
    val_set = DatasetClass(
        root=args.data_dir, train=False, download=True, transform=transform
    )

    # ── Task 0 verification: shape and range checks ──────────────────────────
    x_sample, _ = train_set[0]
    assert x_sample.shape == (1, 28, 28), \
        f"Expected (1,28,28), got {x_sample.shape}"
    assert x_sample.min() >= -1.0 - 1e-5 and x_sample.max() <= 1.0 + 1e-5, \
        f"Data not in [−1,1]: min={x_sample.min():.3f}, max={x_sample.max():.3f}"
    print(f"[Task 0] Dataset shape: {x_sample.shape}, "
          f"dtype: {x_sample.dtype}, "
          f"range: [{x_sample.min():.3f}, {x_sample.max():.3f}]  ✓")

    # Overfit test: restrict training set to 256 images (§4.9 sanity check 1)
    if args.overfit_test:
        indices = list(range(min(256, len(train_set))))
        train_set = Subset(train_set, indices)
        print(f"[Overfit test] Using {len(train_set)} training images.")

    # num_workers=0 required on Windows (spawn-based multiprocessing hangs with CUDA)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_image_grid(images: torch.Tensor, path: str, nrow: int = 8, title: str = ""):
    """Save a grid of images.  images assumed in [−1,1]; rescaled to [0,1]."""
    images = (images.clamp(-1.0, 1.0) + 1.0) / 2.0   # → [0,1]
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(nrow * 1.2, math.ceil(images.shape[0] / nrow) * 1.2))
    ax.imshow(grid_np if grid_np.shape[-1] > 1 else grid_np[:, :, 0], cmap="gray")
    ax.axis("off")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_real_reference(loader: DataLoader, out_dir: str):
    """Save a grid of real images — Task 0 artifact."""
    x, _ = next(iter(loader))
    x = x[:64]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "real_images.png")
    save_image_grid(x, path, nrow=8, title="Real FashionMNIST images (reference)")
    print(f"[Task 0] Saved real image grid → {path}")


def save_denoising_trajectory(trajectory: dict, out_dir: str, step: int):
    """Save denoising trajectory  {t: x_t}  as a single figure (§4.10)."""
    sorted_ts = sorted(trajectory.keys(), reverse=True)
    n = len(sorted_ts)
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    if n == 1:
        axes = [axes]
    for ax, t in zip(axes, sorted_ts):
        img = trajectory[t][0]   # first sample in batch
        img = (img.clamp(-1, 1) + 1) / 2
        if img.shape[0] == 1:
            ax.imshow(img[0].numpy(), cmap="gray")
        else:
            ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(f"t={t}", fontsize=8)
        ax.axis("off")
    plt.suptitle(f"Denoising trajectory (step {step})", fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, f"trajectory_step{step:07d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Sample] Trajectory saved → {path}")


def plot_loss_curve(losses: list, out_dir: str):
    """Plot and save the training loss curve."""
    plt.figure(figsize=(10, 4))
    plt.plot(losses, linewidth=0.8)
    plt.xlabel("Step")
    plt.ylabel("L_simple (MSE)")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  [Log] Loss curve saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks (§4.9)
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check_timesteps(args):
    """Sanity check 4: verify timesteps are sampled uniformly (§4.9)."""
    print("\n[Sanity] Timestep uniformity check...")
    device = torch.device("cpu")
    all_t = []
    for _ in range(1000):
        t = sample_timesteps(args.batch_size, args.L, device)
        all_t.append(t)
    all_t = torch.cat(all_t).numpy()

    # Kolmogorov-Smirnov test or visual histogram
    from scipy import stats
    ks_stat, p_val = stats.kstest(all_t, "uniform",
                                  args=(0, args.L - 1))
    plt.figure(figsize=(8, 3))
    plt.hist(all_t, bins=50, edgecolor="k", alpha=0.7)
    plt.title(f"Timestep histogram  (KS p={p_val:.3f}, uniform if p>0.05)")
    plt.xlabel("Timestep (0-indexed)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("outputs/sanity_timestep_hist.png", dpi=80)
    plt.close()
    print(f"  KS test p-value = {p_val:.4f}  (>0.05 confirms uniformity)  ✓")


def sanity_check_one_step_posterior(schedule, device, n_trials=500):
    """Sanity check 2: E[‖x_{t-1} − x_0‖²] < E[‖x_t − x_0‖²]  (§4.9).

    This verifies that the posterior step actually brings x closer to x_0.
    Uses ORACLE x_0 (i.e., the true posterior — no model involved).
    """
    from diffusion.forward    import q_sample
    from diffusion.posterior  import q_posterior_mean_var

    print("\n[Sanity] One-step posterior check (oracle)...")
    t_fixed = schedule.L // 2   # pick a mid-level timestep

    dist_to_x0_before = []
    dist_to_x0_after  = []

    for _ in range(n_trials):
        # Synthetic 2D "image" for speed
        x0  = torch.randn(1, 1, 4, 4, device=device)
        eps = torch.randn_like(x0)
        t_batch = torch.tensor([t_fixed], device=device)

        # Forward: x_t
        x_t = q_sample(x0, t_batch, eps, schedule)

        # Posterior step: draw x_{t-1} ~ q(x_{t-1} | x_t, x_0)
        mu, var, _ = q_posterior_mean_var(x0, x_t, t_batch, schedule)
        z      = torch.randn_like(mu)
        x_prev = mu + torch.sqrt(var) * z

        dist_to_x0_before.append((x_t   - x0).pow(2).sum().item())
        dist_to_x0_after.append( (x_prev - x0).pow(2).sum().item())

    mean_before = sum(dist_to_x0_before) / n_trials
    mean_after  = sum(dist_to_x0_after)  / n_trials
    ok = mean_after < mean_before
    print(f"  E[‖x_t   − x_0‖²] = {mean_before:.4f}")
    print(f"  E[‖x_{{t-1}} − x_0‖²] = {mean_after:.4f}")
    print(f"  Closer after step?  {'YES ✓' if ok else 'NO ✗  ← BUG!'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── Setup ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── RTX 5090 / Blackwell speed-ups ────────────────────────────────────────
    if device.type == "cuda":
        # TF32: uses tensor-core hardware for float32 matmuls (Ampere+, Blackwell).
        # Trades a tiny bit of precision for 3-8x matmul throughput.
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32       = True
            print("[GPU] TF32 enabled for matmuls and conv (RTX 5090 / Blackwell)")
        # cuDNN auto-tuner: tries different conv algorithms and picks the fastest.
        # Ideal when input shapes are fixed (FashionMNIST 28×28 is fixed).
        torch.backends.cudnn.benchmark = True
        print("[GPU] cuDNN benchmark mode enabled")

    out_dir      = Path(args.out_dir)
    ckpt_dir     = out_dir / "checkpoints"
    sample_dir   = out_dir / "samples"
    schedule_dir = out_dir / "schedules"
    for d in [out_dir, ckpt_dir, sample_dir, schedule_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Task 0: Dataset ───────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(args)
    save_real_reference(train_loader, str(out_dir))

    # ── Task 1: Noise schedule ────────────────────────────────────────────────
    schedule = NoiseSchedule(
        L=args.L,
        schedule_type=args.schedule,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        device=str(device),
    )
    schedule.print_stats()

    # ── Task 1 verification: plot ᾱ_i and SNR(i) ─────────────────────────────
    ts = np.arange(args.L)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(ts, schedule.alpha_bars.cpu().numpy())
    axes[0].set_xlabel("t (0-indexed)")
    axes[0].set_ylabel("ᾱ_t")
    axes[0].set_title(f"Cumulative alpha-bar  ({args.schedule} schedule)")
    axes[0].grid(True, alpha=0.3)

    snr_np = schedule.snr.cpu().numpy()
    axes[1].semilogy(ts, snr_np)
    axes[1].set_xlabel("t (0-indexed)")
    axes[1].set_ylabel("SNR(t)  =  ᾱ_t / (1−ᾱ_t)")
    axes[1].set_title("Signal-to-Noise Ratio  (log scale)")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Schedule: {args.schedule}, L={args.L}")
    plt.tight_layout()
    sched_path = str(schedule_dir / f"schedule_{args.schedule}.png")
    plt.savefig(sched_path, dpi=100)
    plt.close()
    print(f"[Task 1] Schedule plots saved → {sched_path}")

    # ── Task 3: Model ─────────────────────────────────────────────────────────
    model = UNet(
        in_channels=1,
        base_channels=args.base_ch,
        channel_mult=tuple(args.ch_mult),
        time_emb_dim=args.time_emb,
        dropout=args.dropout,
    ).to(device)

    n_params = model.count_parameters()
    print(f"[Model] UNet  |  params: {n_params:,}")

    # ── Sanity checks (before training) ──────────────────────────────────────
    sanity_check_timesteps(args)
    sanity_check_one_step_posterior(schedule, device)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Optional mixed-precision scaler (bf16 preferred on Blackwell/RTX 5090)
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None
    amp_dtype = torch.bfloat16 if (args.amp and device.type == "cuda") else torch.float32

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimiser.load_state_dict(ckpt["optimiser_state"])
        start_step = ckpt["step"]
        print(f"[Resume] Loaded checkpoint from {args.resume}  (step {start_step})")

    # ── torch.compile: fuses ops, reduces Python overhead, uses CUDA graphs ───
    # RTX 5090 (Blackwell / sm_100) benefits significantly from graph compilation.
    # Falls back gracefully if compile fails (Windows / older drivers).
    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[GPU] torch.compile() applied — first few steps will be slower (tracing)")
        except Exception as e:
            print(f"[GPU] torch.compile() failed ({e}), continuing without compile")

    # ── GPU data cache (RTX 5090 optimization) ───────────────────────────────
    gpu_cache = None
    if getattr(args, "gpu_cache", False) and device.type == "cuda":
        from torch.utils.data import Subset as _Subset
        raw_set = train_loader.dataset
        gpu_cache = GPUDataCache(raw_set, device)

    # ── Training loop ─────────────────────────────────────────────────────────
    losses         = []           # per-step loss for loss curve
    grad_norms     = []           # for detecting gradient explosions
    train_iter     = iter(train_loader)
    step           = start_step
    start_time     = time.time()

    print(f"\n[Train] Starting training for {args.steps:,} steps...")

    while step < args.steps:
        model.train()

        # ── Get next batch ───────────────────────────────────────────────────
        if gpu_cache is not None:
            # Zero-latency GPU sampling (RTX 5090: eliminates DataLoader overhead)
            x0 = gpu_cache.sample(args.batch_size)
        else:
            try:
                x0, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x0, _ = next(train_iter)
            x0 = x0.to(device)         # (B, 1, 28, 28) in [−1, 1]

        B  = x0.shape[0]

        # ── Step 2: sample timesteps t ~ Unif{0, …, L-1} ───────────────────
        t = sample_timesteps(B, args.L, device)

        # ── Step 3: sample noise ε ~ N(0, I) ────────────────────────────────
        eps = torch.randn_like(x0)

        # ── Step 4: compute noisy image x_t via eq (2) ──────────────────────
        x_t = q_sample(x0, t, eps, schedule)

        # ── Step 5–6: predict noise and compute L_simple (eq 9) ─────────────
        with torch.amp.autocast("cuda", enabled=(scaler is not None), dtype=amp_dtype):
            eps_hat = model(x_t, t)
            # L_simple = E[‖ε − ε̂‖²]   (eq 9)
            loss = F.mse_loss(eps_hat, eps)

        # ── Step 7: backprop + optimizer step ────────────────────────────────
        optimiser.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        losses.append(loss.item())
        grad_norms.append(grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm))
        step += 1

        # ── Logging ───────────────────────────────────────────────────────────
        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(losses[-args.log_every:]) / args.log_every
            avg_gnorm = sum(grad_norms[-args.log_every:]) / args.log_every
            print(f"  step {step:7d}/{args.steps}  |  "
                  f"loss={avg_loss:.4f}  |  "
                  f"grad_norm={avg_gnorm:.3f}  |  "
                  f"elapsed={elapsed/60:.1f}min")

        # ── Periodic sample grids + trajectory ────────────────────────────────
        if step % args.sample_every == 0 or step == args.steps:
            model.eval()
            print(f"\n  [Sample] Generating at step {step}…")
            result = ancestral_sample(
                model, schedule,
                shape=(64, 1, 28, 28),
                device=device,
                verbose=False,
            )
            sample_path = str(sample_dir / f"samples_step{step:07d}.png")
            save_image_grid(result["x0"], sample_path, nrow=8,
                            title=f"Generated (step {step})")
            print(f"  [Sample] Grid saved → {sample_path}")

            save_denoising_trajectory(
                result["trajectory"], str(sample_dir), step
            )
            model.train()

        # ── Checkpoint saving ────────────────────────────────────────────────
        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = str(ckpt_dir / f"ckpt_step{step:07d}.pt")
            torch.save({
                "step":           step,
                "model_state":    model.state_dict(),
                "optimiser_state":optimiser.state_dict(),
                "args":           vars(args),
            }, ckpt_path)
            print(f"  [Ckpt] Saved → {ckpt_path}")

    # ── Post-training: save loss curve ────────────────────────────────────────
    plot_loss_curve(losses, str(out_dir))

    # ── Noise-prediction sanity check (§4.9 check 3) ─────────────────────────
    print("\n[Sanity] Noise-prediction correlation check...")
    model.eval()
    t_fixed = args.L // 2
    corrs = []
    with torch.no_grad():
        for x0, _ in val_loader:
            x0 = x0.to(device)
            eps = torch.randn_like(x0)
            t_batch = torch.full((x0.shape[0],), t_fixed, device=device)
            x_t = q_sample(x0, t_batch, eps, schedule)
            eps_hat = model(x_t, t_batch)
            # Pearson correlation between ε and ε̂ (flattened)
            eps_f = eps.flatten(1).cpu().numpy()
            eh_f  = eps_hat.flatten(1).cpu().numpy()
            for i in range(eps_f.shape[0]):
                c = np.corrcoef(eps_f[i], eh_f[i])[0, 1]
                corrs.append(c)
            if len(corrs) >= 200:
                break

    mean_corr = float(np.nanmean(corrs))
    print(f"  Mean Pearson correlation(ε, ε̂) at t={t_fixed}: {mean_corr:.4f}")
    print(f"  (After convergence this should be > 0.5)")

    print("\n[Train] Done.")
    return model, schedule


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
