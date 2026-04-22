"""diffusion/posterior.py — True posterior and reverse-step mechanics.

Key formulas implemented here:

  True forward posterior  q(x_{i-1} | x_i, x_0)  =  N(μ̃_i, β̃_i · I)
                                                        ↑ eq (5)
  β̃_i  =  (1 − ᾱ_{i-1}) / (1 − ᾱ_i) · β_i                       (eq 6)

  μ̃_i(x_i, x_0) = [√ᾱ_{i-1}·β_i / (1−ᾱ_i)] · x_0
                 + [√α_i·(1−ᾱ_{i-1}) / (1−ᾱ_i)] · x_i            (eq 7)

  ε-parameterized reverse mean (eq 8):
  μ_θ(x_i, i) = 1/√α_i · (x_i − β_i/√(1−ᾱ_i) · ε̂_θ(x_i, i))

  x̂_0 from predicted noise (needed to compute eq 7 during sampling):
  x̂_0 = 1/√ᾱ_i · (x_i − √(1−ᾱ_i) · ε̂)

CRITICAL PITFALL (from assignment §4.7):
  Using β_i instead of β̃_i as the sampling noise variance is a COMMON BUG
  that produces poor sample quality.  This file exclusively uses β̃_i
  (posterior_variance from schedule) for sampling.

INDEXING CONVENTION: 0-indexed (see diffusion/schedule.py).
"""
import os, sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


import torch
from diffusion.schedule import NoiseSchedule


# ─────────────────────────────────────────────────────────────────────────────
# 1. True posterior statistics  (eqs 6–7)
# ─────────────────────────────────────────────────────────────────────────────

def q_posterior_mean_var(
    x0: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
):
    """Compute the true posterior mean and variance  q(x_{t-1} | x_t, x_0).

    This is the "teacher" distribution — the distribution the reverse model
    pθ(x_{t-1} | x_t) should match.  It is tractable because we condition on
    x_0 (i.e., we are using the training-time oracle).

    Formulas:
        μ̃_t = coef1 · x_0  +  coef2 · x_t          (eq 7)
        β̃_t = (1 − ᾱ_{t-1})/(1 − ᾱ_t) · β_t       (eq 6)

    Args:
        x0       : clean image,     shape (B, C, H, W).
        x_t      : noisy image at t, shape (B, C, H, W).
        t        : 0-indexed timesteps, shape (B,).
        schedule : NoiseSchedule.

    Returns:
        (posterior_mean, posterior_variance, posterior_log_var_clipped)
        all of shape (B, C, H, W).

    Why β̃ < β?
        β̃_t = (1−ᾱ_{t-1})/(1−ᾱ_t) · β_t
        Since ᾱ_{t-1} > ᾱ_t  (cumulative product is decreasing),
        we have  (1−ᾱ_{t-1}) < (1−ᾱ_t), so  β̃_t < β_t.
        Interpretation: the posterior is *more certain* than the forward step
        because x_0 is observed — it anchors the estimate.
    """
    ndim = x_t.ndim

    # Extract per-sample coefficients, shape (B, 1, 1, 1)
    coef1 = schedule.extract(schedule.posterior_mean_coef1, t, ndim)
    coef2 = schedule.extract(schedule.posterior_mean_coef2, t, ndim)
    pvar  = schedule.extract(schedule.posterior_variance,   t, ndim)
    plvar = schedule.extract(schedule.posterior_log_var_clipped, t, ndim)

    # μ̃_t(x_t, x_0)  =  coef1 · x_0  +  coef2 · x_t        (eq 7)
    posterior_mean = coef1 * x0 + coef2 * x_t

    return posterior_mean, pvar, plvar


# ─────────────────────────────────────────────────────────────────────────────
# 2. Recover x̂_0 from predicted noise ε̂
# ─────────────────────────────────────────────────────────────────────────────

def predict_x0_from_eps(
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps_hat: torch.Tensor,
    schedule: NoiseSchedule,
) -> torch.Tensor:
    """Recover the clean image estimate x̂_0 from the predicted noise.

    Derived by rearranging eq (2):
        x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε
        →  x̂_0 = (x_t − √(1−ᾱ_t) · ε̂) / √ᾱ_t
               = √(1/ᾱ_t) · x_t  −  √(1/ᾱ_t − 1) · ε̂

    This estimate can be used both:
      (a) during sampling to compute μ̃ via eq (7), and
      (b) in x_0-prediction ablations (Task 6).

    Args:
        x_t     : noisy image at t,     shape (B, C, H, W).
        t       : 0-indexed timesteps,  shape (B,).
        eps_hat : predicted noise ε̂,   shape (B, C, H, W).
        schedule: NoiseSchedule.

    Returns:
        x0_hat  : estimated clean image, shape (B, C, H, W).
    """
    # 1/√ᾱ_t,  shape (B, 1, 1, 1)
    recip_sqrt_ab = schedule.extract(
        1.0 / schedule.sqrt_alpha_bars, t, x_t.ndim
    )
    # √(1−ᾱ_t),  shape (B, 1, 1, 1)
    sqrt_1mab = schedule.extract(
        schedule.sqrt_one_minus_alpha_bars, t, x_t.ndim
    )

    # x̂_0 = (x_t − √(1−ᾱ_t) · ε̂) / √ᾱ_t
    x0_hat = recip_sqrt_ab * (x_t - sqrt_1mab * eps_hat)
    return x0_hat


# ─────────────────────────────────────────────────────────────────────────────
# 3. ε-parameterized reverse mean  (eq 8)
# ─────────────────────────────────────────────────────────────────────────────

def p_mean_from_eps(
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps_hat: torch.Tensor,
    schedule: NoiseSchedule,
) -> torch.Tensor:
    """Compute the learned reverse mean μ_θ from the predicted noise ε̂.

    Directly implements eq (8):
        μ_θ(x_t, t) = 1/√α_t · (x_t − β_t/√(1−ᾱ_t) · ε̂_θ(x_t, t))

    This is equivalent to plugging the denoiser's estimate x̂_0 = predict_x0_from_eps()
    into the posterior mean formula eq (7) — but the combined form is cleaner.

    Args:
        x_t     : noisy image at t,    shape (B, C, H, W).
        t       : 0-indexed timesteps, shape (B,).
        eps_hat : predicted noise ε̂,  shape (B, C, H, W).
        schedule: NoiseSchedule.

    Returns:
        mu : reverse mean, shape (B, C, H, W).
    """
    ndim = x_t.ndim

    # 1/√α_t,  shape (B, 1, 1, 1)
    recip_sqrt_alpha = schedule.extract(schedule.recip_sqrt_alphas, t, ndim)
    # β_t / √(1−ᾱ_t),  shape (B, 1, 1, 1)
    beta_div_sqrt_1mab = schedule.extract(
        schedule.betas_div_sqrt_one_minus_ab, t, ndim
    )

    # μ_θ = 1/√α_t · (x_t − β_t/√(1−ᾱ_t) · ε̂)     (eq 8)
    mu = recip_sqrt_alpha * (x_t - beta_div_sqrt_1mab * eps_hat)
    return mu


# ─────────────────────────────────────────────────────────────────────────────
# 4. One full reverse step  p(x_{t-1} | x_t)
# ─────────────────────────────────────────────────────────────────────────────

def p_sample_step(
    x_t: torch.Tensor,
    t: int,
    eps_hat: torch.Tensor,
    schedule: NoiseSchedule,
) -> torch.Tensor:
    """Perform one ancestral sampling step: x_t → x_{t-1}.

    Algorithm:
      1. Compute reverse mean  μ = p_mean_from_eps(x_t, t, ε̂)
      2. If t > 0: add noise  z ~ N(0,I) with variance β̃_t   (eq 5)
         If t = 0: return μ directly  (NO noise at last step!)

    CRITICAL: The variance used here is β̃_t (posterior_variance),
    NOT β_t.  Using β_t is a common bug that degrades sample quality.
    This assignment's variance convention: Σ_θ(t) = β̃_t · I  (Appendix §5.1.2)

    Args:
        x_t     : noisy image at t,      shape (B, C, H, W).
        t       : scalar 0-indexed timestep (integer, not tensor).
        eps_hat : predicted noise ε̂,    shape (B, C, H, W).
        schedule: NoiseSchedule.

    Returns:
        x_{t-1} : less noisy image, shape (B, C, H, W).
    """
    B = x_t.shape[0]
    device = x_t.device

    # Build a batch of the same timestep for extract()
    t_batch = torch.full((B,), t, device=device, dtype=torch.long)

    # Step 1 — compute reverse mean
    mu = p_mean_from_eps(x_t, t_batch, eps_hat, schedule)

    # Step 2 — add noise (or not)
    if t > 0:
        # β̃_t, shape (B, 1, 1, 1)
        beta_tilde = schedule.extract(schedule.posterior_variance, t_batch, x_t.ndim)
        z = torch.randn_like(x_t)
        # x_{t-1} = μ + √β̃_t · z                 (ancestral sampling rule)
        x_prev = mu + torch.sqrt(beta_tilde) * z
    else:
        # t = 0: last step — return mean, NO noise added
        # Why? At t=0, β̃_0 = 0 (see schedule.py notes), so σ=0.
        x_prev = mu

    return x_prev
