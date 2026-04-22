"""diffusion/schedule.py — Noise schedule for DDPM.

INDEXING CONVENTION (enforced throughout this entire codebase):
────────────────────────────────────────────────────────────────
  • All schedule arrays are 0-indexed: array index t  ↔  paper timestep (t+1).
  • So  betas[0]   = β_1 (paper),  betas[L-1]  = β_L (paper).
  • Timestep variables t are integers in [0, L-1].
  • sample_timesteps() returns integers in [0, L-1].
  • Forward noising:   t = 0  →  lightest noise,  t = L-1  →  heaviest noise.
  • Ancestral sampling: iterate t from L-1 → 0.

WHY 0-INDEXED?  Maps directly to Python array indexing and avoids off-by-one
errors.
────────────────────────────────────────────────────────────────

Key objects:
  make_beta_schedule(L, type, beta_min, beta_max) → Tensor (L,)
  NoiseSchedule — stores ALL precomputed diffusion scalars as tensors.
"""
import os, sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


import math
import torch
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Beta schedule factory
# ─────────────────────────────────────────────────────────────────────────────

def make_beta_schedule(
    L: int,
    schedule_type: str = "linear",
    beta_min: float = 1e-4,
    beta_max: float = 0.02,
) -> torch.Tensor:
    """Return a length-L tensor of β_i values (0-indexed).

    Supported schedules:
      "linear"  – linearly spaced from beta_min to beta_max  (Ho et al., 2020)
      "cosine"  – cosine schedule from Nichol & Dhariwal, 2021 (for ablation)

    Args:
        L           : number of diffusion steps (e.g. 1000).
        schedule_type: "linear" or "cosine".
        beta_min    : β_1  (first/smallest beta); used only for linear.
        beta_max    : β_L  (last/largest  beta); used only for linear.

    Returns:
        betas : float32 tensor of shape (L,), all values strictly in (0, 1).
    """
    if schedule_type == "linear":
        # DDPM baseline: β_1=1e-4, β_L=0.02, linearly spaced.
        betas = torch.linspace(beta_min, beta_max, L, dtype=torch.float64)

    elif schedule_type == "cosine":
        # Cosine schedule (Nichol & Dhariwal, "Improved DDPM", 2021).
        # ᾱ(t) = cos²(((t/T)+s)/(1+s) · π/2) / cos²(s/(1+s) · π/2)
        # This avoids the sudden noise increase at the end of linear schedules.
        s = 0.008  # small offset to prevent β_1 from being too small
        steps = L + 1
        t = torch.linspace(0, L, steps, dtype=torch.float64)
        # Compute cumulative alpha-bars using cosine formula
        alpha_bar_raw = torch.cos(((t / L) + s) / (1.0 + s) * math.pi / 2.0) ** 2
        alpha_bar = alpha_bar_raw / alpha_bar_raw[0]          # normalize so ᾱ_0 = 1
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])        # β_t = 1 - ᾱ_t/ᾱ_{t-1}
        betas = torch.clamp(betas, min=1e-5, max=0.02)         # clip per Nichol & Dhariwal (2021): prevents 1/√α_t explosion at t→999

    else:
        raise ValueError(
            f"Unknown schedule_type='{schedule_type}'. "
            "Choose 'linear' or 'cosine'."
        )

    return betas.float()


# ─────────────────────────────────────────────────────────────────────────────
# NoiseSchedule — container for ALL precomputed diffusion scalars
# ─────────────────────────────────────────────────────────────────────────────

class NoiseSchedule:
    """Stores all precomputed DDPM diffusion scalars as device tensors.

    Every quantity  needed in forward noising, posterior computation, or
    sampling is precomputed here ONCE so training/sampling code only does
    cheap indexing.

    INDEXING: all tensors have shape (L,) and are 0-indexed (see module doc).

    Key tensors (all shape (L,)):
    ──────────────────────────────────────────────────────────────
    betas                         β_i                    (0-indexed)
    alphas                        α_i = 1 - β_i
    alpha_bars                    ᾱ_i = ∏_{j=0}^{i} α_j     (cumulative product)
    alpha_bars_prev               ᾱ_{i-1}, with ᾱ_{-1} := 1
    sqrt_alpha_bars               √ᾱ_i
    sqrt_one_minus_alpha_bars     √(1 - ᾱ_i)
    recip_sqrt_alphas             1/√α_i
    betas_div_sqrt_one_minus_ab   β_i / √(1-ᾱ_i)            (used in eq 8)
    posterior_variance            β̃_i  from eq (6)
    posterior_log_var_clipped     log(max(β̃_i, 1e-20))
    posterior_mean_coef1          coef1 from eq (7): √ᾱ_{i-1}·β_i / (1-ᾱ_i)
    posterior_mean_coef2          coef2 from eq (7): √α_i·(1-ᾱ_{i-1}) / (1-ᾱ_i)
    snr                           SNR(i) = ᾱ_i / (1-ᾱ_i)
    ──────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        L: int = 1000,
        schedule_type: str = "linear",
        beta_min: float = 1e-4,
        beta_max: float = 0.02,
        device: str = "cpu",
    ):
        self.L = L
        self.device = device

        # ── β_i ────────────────────────────────────────────────────────────
        betas = make_beta_schedule(L, schedule_type, beta_min, beta_max)  # (L,)

        # ── α_i = 1 - β_i ──────────────────────────────────────────────────
        alphas = 1.0 - betas                                              # (L,)

        # ── ᾱ_i = ∏_{j=1}^{i} α_j  (cumulative product, 0-indexed) ───────
        # ᾱ_0 = α_0 = 1 - β_0  (lightest noise level)
        alpha_bars = torch.cumprod(alphas, dim=0)                         # (L,)

        # ── ᾱ_{i-1}, with ᾱ_{-1} := 1  (needed for posterior) ─────────────
        # alpha_bars_prev[0] = 1  (before any diffusion, ᾱ_0 = 1 by convention)
        # alpha_bars_prev[t] = alpha_bars[t-1]  for t >= 1
        alpha_bars_prev = torch.cat(
            [torch.ones(1, dtype=torch.float64), alpha_bars[:-1].double()]
        ).float()                                                          # (L,)

        # ── Posterior variance β̃_i  (eq 6) ─────────────────────────────────
        # β̃_i = (1 - ᾱ_{i-1}) / (1 - ᾱ_i) · β_i
        # IMPORTANT: β̃_0 = 0  because ᾱ_{-1} = 1, so (1-ᾱ_{-1}) = 0.
        # This means at t=0 (last denoising step) we add NO noise. ✓
        posterior_variance = (
            (1.0 - alpha_bars_prev) / (1.0 - alpha_bars) * betas
        )                                                                  # (L,)

        # log(β̃_i) is used in some variance computations; clamp to avoid -inf at t=0
        posterior_log_var_clipped = torch.log(
            torch.clamp(posterior_variance, min=1e-20)
        )                                                                  # (L,)

        # ── Posterior mean coefficients  (eq 7) ────────────────────────────
        # μ̃_i(x_i, x_0) = coef1 · x_0 + coef2 · x_i
        # coef1 = √ᾱ_{i-1} · β_i / (1 - ᾱ_i)
        # coef2 = √α_i · (1 - ᾱ_{i-1}) / (1 - ᾱ_i)
        posterior_mean_coef1 = (
            torch.sqrt(alpha_bars_prev) * betas / (1.0 - alpha_bars)
        )                                                                  # (L,)
        posterior_mean_coef2 = (
            torch.sqrt(alphas) * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        )                                                                  # (L,)

        # ── ε-parameterized mean term  (eq 8): β_i / √(1-ᾱ_i) ─────────────
        # μ_θ(x_i, i) = 1/√α_i · (x_i - β_i/√(1-ᾱ_i) · ε_θ(x_i,i))
        betas_div_sqrt_one_minus_ab = betas / torch.sqrt(1.0 - alpha_bars) # (L,)

        # ── SNR(i) = ᾱ_i / (1-ᾱ_i)  (defined in the assignment) ──────────
        snr = alpha_bars / (1.0 - alpha_bars)                             # (L,)

        # ── Store all tensors on the requested device ───────────────────────
        def _to(x):
            return x.float().to(device)

        self.betas                       = _to(betas)
        self.alphas                      = _to(alphas)
        self.alpha_bars                  = _to(alpha_bars)
        self.alpha_bars_prev             = _to(alpha_bars_prev)
        self.sqrt_alpha_bars             = _to(torch.sqrt(alpha_bars))
        self.sqrt_one_minus_alpha_bars   = _to(torch.sqrt(1.0 - alpha_bars))
        self.recip_sqrt_alphas           = _to(1.0 / torch.sqrt(alphas))
        self.betas_div_sqrt_one_minus_ab = _to(betas_div_sqrt_one_minus_ab)
        self.posterior_variance          = _to(posterior_variance)
        self.posterior_log_var_clipped   = _to(posterior_log_var_clipped)
        self.posterior_mean_coef1        = _to(posterior_mean_coef1)
        self.posterior_mean_coef2        = _to(posterior_mean_coef2)
        self.snr                         = _to(snr)

    # ─────────────────────────────────────────────────────────────────────────
    # Utility: broadcast a schedule tensor to image shape
    # ─────────────────────────────────────────────────────────────────────────

    def extract(self, arr: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        """Extract arr[t] and reshape for broadcasting over spatial dims.

        Args:
            arr  : 1-D schedule tensor of length L, already on correct device.
            t    : integer tensor of shape (B,), values in [0, L-1].
            ndim : number of dimensions of the target tensor (e.g. 4 for BCHW).

        Returns:
            shape (B, 1, 1, ...) with (ndim-1) trailing singleton dims.

        Example:
            sqrt_ab_t = schedule.extract(schedule.sqrt_alpha_bars, t, x.ndim)
            # → shape (B, 1, 1, 1)  →  broadcasts with x of shape (B, C, H, W)
        """
        values = arr[t]                           # (B,)
        return values.reshape(t.shape[0], *([1] * (ndim - 1)))

    def to(self, device: str) -> "NoiseSchedule":
        """Move all tensors to a new device in place."""
        self.device = device
        attrs = [
            "betas", "alphas", "alpha_bars", "alpha_bars_prev",
            "sqrt_alpha_bars", "sqrt_one_minus_alpha_bars",
            "recip_sqrt_alphas", "betas_div_sqrt_one_minus_ab",
            "posterior_variance", "posterior_log_var_clipped",
            "posterior_mean_coef1", "posterior_mean_coef2", "snr",
        ]
        for attr in attrs:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Quick diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    def print_stats(self) -> None:
        """Print min/max of key schedule quantities — Step 1 of debugging checklist."""
        print("── Schedule statistics (0-indexed, L={}) ──".format(self.L))
        for name in ["betas", "alphas", "alpha_bars", "posterior_variance"]:
            arr = getattr(self, name)
            print(f"  {name:35s}: min={arr.min().item():.6f}  max={arr.max().item():.6f}")
        print(f"  {'SNR[0] (lightest noise)':<35s}: {self.snr[0].item():.4f}")
        print(f"  {'SNR[-1] (heaviest noise)':<35s}: {self.snr[-1].item():.6f}")
        print(f"  {'alpha_bars[-1]':<35s}: {self.alpha_bars[-1].item():.6f}  "
              f"(should be close to 0)")
