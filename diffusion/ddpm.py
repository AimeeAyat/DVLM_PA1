"""diffusion/ddpm.py — Ancestral DDPM sampler.

Implements the full reverse chain:

    x_L ~ N(0, I)
    for t = L-1, L-2, ..., 0:
        ε̂ = ε_θ(x_t, t)
        x_{t-1} = p_sample_step(x_t, t, ε̂)

Uses β̃_t as sampling variance (this assignment's variance convention).
Saves intermediate x_t at specified timesteps for trajectory visualization.

INDEXING CONVENTION: 0-indexed (see diffusion/schedule.py).
"""
from __future__ import annotations
import os, sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import torch
import torch.nn as nn
from typing import Optional, List, Dict

from diffusion.schedule import NoiseSchedule
from diffusion.posterior import p_sample_step


# ─────────────────────────────────────────────────────────────────────────────
# Full ancestral sampler
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ancestral_sample(
    model: nn.Module,
    schedule: NoiseSchedule,
    shape: tuple,
    device: torch.device,
    save_at: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    """Generate samples using ancestral (DDPM) sampling.

    Full reverse-chain algorithm (§4.10 of the assignment):
      1. Start from x_L ~ N(0, I)
      2. For t = L-1 down to 0:
           ε̂ = model(x_t, t)
           x_{t-1} = p_sample_step(x_t, t, ε̂)
      3. Return x_0

    Args:
        model    : trained ε_θ(x_t, t) network.
        schedule : NoiseSchedule (must be on same device).
        shape    : output shape, e.g. (64, 1, 28, 28).
        device   : torch device.
        save_at  : list of 0-indexed timesteps at which to save x_t for
                   trajectory visualization.  E.g. [999, 749, 499, 249, 0].
                   Defaults to [L-1, 3L/4, L/2, L/4, 0].
        verbose  : if True, print progress every 100 steps.

    Returns:
        dict with keys:
          "x0"           : final sample,  shape = `shape`.
          "trajectory"   : dict {t: x_t_tensor} for each t in save_at.
          "norms"        : list of ‖x_t‖ values across all timesteps (for
                           blow-up detection, §4.10 diagnostic).
    """
    model.eval()
    L = schedule.L

    # Default trajectory checkpoints: {L-1, 3L/4, L/2, L/4, 0}
    if save_at is None:
        save_at = sorted(set([
            L - 1,
            int(3 * L / 4),
            L // 2,
            L // 4,
            0,
        ]), reverse=True)

    # ── 1. Sample initial noise x_L ~ N(0, I) ──────────────────────────────
    x = torch.randn(shape, device=device)

    trajectory = {}
    norms = []

    # ── 2. Iterate from t = L-1 down to t = 0 ──────────────────────────────
    for t in reversed(range(L)):

        # Build integer-timestep batch tensor for the network
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # Predict noise ε̂ = ε_θ(x_t, t)
        eps_hat = model(x, t_batch)

        # One reverse step: x_t → x_{t-1}
        x = p_sample_step(x, t, eps_hat, schedule)

        # Track ‖x_t‖ for blow-up detection
        norms.append(x.norm().item())

        # Save intermediate state if requested
        if t in save_at:
            trajectory[t] = x.detach().cpu()

        if verbose and t % 100 == 0:
            print(f"  t={t:4d}  ‖x‖={norms[-1]:.3f}")

    return {
        "x0":         x.cpu(),         # final generated images
        "trajectory": trajectory,      # {timestep: tensor}
        "norms":      norms,           # list[float], length L
    }


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic DDIM-style sampler (Task 6 extension — optional)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    schedule: NoiseSchedule,
    shape: tuple,
    device: torch.device,
    num_steps: int = 50,
    eta: float = 0.0,
) -> torch.Tensor:
    """Deterministic (DDIM-style) sampling using a subset of timesteps.

    Reference: Song et al., "Denoising Diffusion Implicit Models" (2021).

    The key idea: instead of running all L=1000 steps, we pick a
    subsequence of S ≤ L timesteps and use a deterministic update rule.
    When η=0, the sampler is fully deterministic (same noise seed → same
    output).  When η=1, it reduces to the standard DDPM ancestral sampler.

    Deterministic update rule:
        x̂_0 = (x_t − √(1−ᾱ_t)·ε̂) / √ᾱ_t
        x_{t-1} = √ᾱ_{t-1} · x̂_0
                + √(1−ᾱ_{t-1} − σ²) · ε̂
                + σ · z,         z ~ N(0,I)
        σ = η · √[(1−ᾱ_{t-1})/(1−ᾱ_t)] · √[1 − ᾱ_t/ᾱ_{t-1}]

    When η=0:  σ=0,  fully deterministic.
    When η=1:  σ=β̃_t^{1/2},  standard DDPM.

    Args:
        model     : trained ε_θ network.
        schedule  : NoiseSchedule.
        shape     : output shape, e.g. (64, 1, 28, 28).
        device    : torch device.
        num_steps : number of sampling steps S ≤ L.
        eta       : stochasticity coefficient (0 = deterministic DDIM).

    Returns:
        x0 : generated samples, shape = `shape`, on CPU.
    """
    model.eval()
    L = schedule.L

    # Build a uniformly-spaced subsequence of timesteps in [0, L-1]
    # e.g. L=1000, S=50 → [0, 20, 40, ..., 980] (reversed during sampling)
    step_size = L // num_steps
    timesteps = list(reversed(range(0, L, step_size)))[:num_steps]

    x = torch.randn(shape, device=device)

    for i, t in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1

        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # Predict noise
        eps_hat = model(x, t_batch)

        # Extract schedule quantities
        ab_t    = schedule.alpha_bars[t]
        ab_prev = schedule.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Recover x̂_0
        x0_hat = (x - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt()
        x0_hat = x0_hat.clamp(-1.0, 1.0)

        # Compute σ for this step
        sigma = (
            eta
            * ((1 - ab_prev) / (1 - ab_t)).sqrt()
            * (1 - ab_t / ab_prev).sqrt()
        )

        # DDIM update
        mean_component = ab_prev.sqrt() * x0_hat
        dir_component  = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps_hat

        if t_prev >= 0:
            z = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = mean_component + dir_component + sigma * z
        else:
            x = mean_component  # last step: t_prev = -1 → ᾱ_prev = 1

    return x.cpu()
