"""diffusion/forward.py — Forward (noising) process for DDPM.

Implements the closed-form marginal  q(x_i | x_0)  from equation (2):

    x_i = √ᾱ_i · x_0  +  √(1−ᾱ_i) · ε,    ε ~ N(0, I)          (eq 2)

This is the KEY formula that lets us jump directly to ANY noise level i in
one shot — no need to simulate the chain step-by-step during training.

Why is this possible?
  The forward chain is a product of Gaussians (eq 1).
  The composition of Gaussians is Gaussian  →  closed form for any i.
 
INDEXING CONVENTION: 0-indexed (see diffusion/schedule.py for details).
"""
import os, sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


import torch
from diffusion.schedule import NoiseSchedule


# ─────────────────────────────────────────────────────────────────────────────
# q_sample — closed-form marginal  (eq 2)
# ─────────────────────────────────────────────────────────────────────────────

def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor,
    schedule: NoiseSchedule,
) -> torch.Tensor:
    """Apply closed-form forward noising to x0 at timestep t.

    Implements eq (2):
        x_t = √ᾱ_t · x_0  +  √(1−ᾱ_t) · ε

    This does NOT simulate the chain step-by-step.  It uses the reparameterized
    form that gives us x at any timestep t directly from x_0.

    Args:
        x0       : clean data tensor,     shape (B, C, H, W), in [−1, 1].
        t        : integer timestep indices, shape (B,), values in [0, L−1].
        eps      : noise tensor,           shape (B, C, H, W) ~ N(0, I).
        schedule : NoiseSchedule instance (must be on same device as x0).

    Returns:
        x_t : noisy data at timestep t,   shape (B, C, H, W).

    Note on pitfall:
        Data must be consistently in [−1, 1] here and at decoding time.
        Training in [−1, 1] but displaying in [0, 1] without re-scaling causes
        "washed-out" / grey samples.
    """
    # √ᾱ_t, shape → (B, 1, 1, 1) for broadcasting
    sqrt_ab   = schedule.extract(schedule.sqrt_alpha_bars,           t, x0.ndim)
    # √(1−ᾱ_t), shape → (B, 1, 1, 1)
    sqrt_1mab = schedule.extract(schedule.sqrt_one_minus_alpha_bars, t, x0.ndim)

    # x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε       ← eq (2)
    return sqrt_ab * x0 + sqrt_1mab * eps


# ─────────────────────────────────────────────────────────────────────────────
# sample_timesteps — uniform timestep sampling for training
# ─────────────────────────────────────────────────────────────────────────────

def sample_timesteps(batch_size: int, L: int, device: torch.device) -> torch.Tensor:
    """Sample a batch of timestep indices uniformly from {0, …, L−1}.

    This corresponds to  i ~ Unif{1, …, L}  in the paper (eq 9),
    translated to 0-indexed [0, L−1] per our convention.

    Args:
        batch_size : number of timesteps to sample (one per training image).
        L          : total diffusion steps (e.g. 1000).
        device     : torch device.

    Returns:
        t : int64 tensor of shape (batch_size,), values in [0, L−1].
    """
    return torch.randint(0, L, (batch_size,), device=device)
