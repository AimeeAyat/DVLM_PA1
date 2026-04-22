"""models/unet.py — U-Net noise predictor  ε_θ(x_i, i).

Architecture (FashionMNIST 28×28):

    Input x_t: (B, 1, 28, 28)    +    timestep t: (B,)
    ↓
    SinusoidalTimestepEmbedding(dim=128) → MLP(128→256) → t_emb: (B, 256)
    ↓
    stem conv 1→32, 3×3
    ↓ ─────────────────────────── ENCODER ────────────────────────────
    ResBlock(32, 32,  t=256)  @ 28×28    → skip_0
    Downsample(32)            → 14×14
    ResBlock(32, 64,  t=256)  @ 14×14    → skip_1
    Downsample(64)            → 7×7
    ↓ ─────────────────────────── BOTTLENECK ─────────────────────────
    ResBlock(64, 128, t=256)  @ 7×7
    ↓ ─────────────────────────── DECODER ────────────────────────────
    Upsample(128)             → 14×14
    cat(skip_1: 64ch)         → 192 ch
    ResBlock(192, 64, t=256)  @ 14×14
    Upsample(64)              → 28×28
    cat(skip_0: 32ch)         → 96 ch
    ResBlock(96,  32, t=256)  @ 28×28
    ↓
    GroupNorm + SiLU + Conv(32→1) → output ε̂: (B, 1, 28, 28)

Design decisions (following assignment §4.4):
  • Sinusoidal embedding → MLP projection → added as BIAS after 1st conv
    of every ResBlock (not as a separate layer).  This is the standard DDPM
    injection method.
  • GroupNorm + SiLU everywhere (no BatchNorm, no ReLU).
  • No attention initially (assignment says: "Keep attention off initially").
  • GroupNorm groups = max(1, channels//4) so all channel sizes divide evenly.

CRITICAL PITFALL (§4.8):
  If the timestep embedding is computed but never used inside the network
  (e.g., forgotten in forward()), training loss will still decrease but
  sampling quality will plateau — the model learns an unconditional denoiser.
  Check by running the timestep sanity test in sanity_checks.py.
"""
import os, sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Helper: number of GroupNorm groups
# ─────────────────────────────────────────────────────────────────────────────

def _num_groups(channels: int) -> int:
    """Return the largest divisor of `channels` that is ≤ 32."""
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal Timestep Embedding  (fixed, not learned)
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalTimestepEmbedding(nn.Module):
    """Fixed sinusoidal positional encoding for timestep t.

    Converts integer t ∈ [0, L-1] into a continuous vector.
    Uses the same formula as the original Transformer positional encoding:

        e_k(t) = sin(t / 10000^{2k/d})   for k = 0, 1, …, d/2-1
        e_{k+d/2}(t) = cos(t / 10000^{2k/d})

    The resulting dim-dimensional vector is then projected through a small MLP
    to produce a time embedding that is *added as bias* inside each ResBlock.

    Why sinusoidal (not learned)?
      • Generalizes to unseen timesteps.
      • Well-studied in attention-based models (Vaswani et al., 2017).
      • The DDPM paper uses this and notes it works well in practice.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t : integer tensor, shape (B,), values in [0, L-1].
        Returns:
            embeddings : float tensor, shape (B, dim).
        """
        device = t.device
        half_dim = self.dim // 2

        # Frequency exponents: log(10000) / (half_dim - 1) * k  for k=0..half_dim-1
        exponent = torch.arange(half_dim, device=device).float()
        exponent = exponent * -(math.log(10000.0) / (half_dim - 1))
        freqs = torch.exp(exponent)                         # (half_dim,)

        # t · freqs,  shape (B, half_dim)
        args = t.float()[:, None] * freqs[None, :]

        # Concat sin and cos → (B, dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


# ─────────────────────────────────────────────────────────────────────────────
# Residual Block with timestep conditioning
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block with GroupNorm + SiLU and timestep bias injection.

    Architecture:
        x  →  GN → SiLU → Conv1(in→out) → [+t_bias] → GN → SiLU → Dropout → Conv2(out→out)
        +
        shortcut (1×1 conv if in≠out, else Identity)
        ↓
        output

    Timestep injection:
        t_emb → Linear(t_dim, out_channels) → reshape (B, out_ch, 1, 1)
        This is added as a BIAS to the activations AFTER the first conv.
        This is exactly the method described in §4.4 of the assignment.

    Args:
        in_channels  : input channel count.
        out_channels : output channel count.
        time_emb_dim : dimension of the time embedding vector (e.g. 256).
        dropout      : dropout probability (default 0.1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── First norm + conv ────────────────────────────────────────────────
        self.norm1 = nn.GroupNorm(_num_groups(in_channels),  in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # ── Timestep bias projection ─────────────────────────────────────────
        # SiLU before the linear is the standard DDPM approach.
        # The Linear maps t_emb → out_channels bias, added spatially.
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        # ── Second norm + conv ───────────────────────────────────────────────
        self.norm2   = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2   = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # ── Residual connection ──────────────────────────────────────────────
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x     : (B, in_channels, H, W)
            t_emb : (B, time_emb_dim)
        Returns:
            out   : (B, out_channels, H, W)
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # ⚠️ Inject timestep embedding HERE — after first conv, as spatial bias.
        # Shape: (B, out_channels) → (B, out_channels, 1, 1) → broadcasts with h.
        t_bias = self.time_proj(t_emb)[:, :, None, None]
        h = h + t_bias                  # this is the timestep conditioning

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


# ─────────────────────────────────────────────────────────────────────────────
# Downsampling and Upsampling
# ─────────────────────────────────────────────────────────────────────────────

class Downsample(nn.Module):
    """Stride-2 conv for spatial downsampling  H×W → H/2 × W/2."""

    def __init__(self, channels: int):
        super().__init__()
        # Strided convolution (not maxpool) preserves learned features
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Bilinear upsample 2× followed by conv to clean up grid artifacts."""

    def __init__(self, channels: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


# ─────────────────────────────────────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """Small U-Net for ε_θ(x_t, t) on FashionMNIST 28×28.

    Channel configuration:  base_channels=32, channel_mult=(1, 2, 4)
      → channel widths: (32, 64, 128)
      → spatial resolutions: 28×28, 14×14, 7×7

    This exactly matches the baseline configuration in §4.2 of the assignment:
      "small U-Net, channel multipliers (32, 64, 128), 2–3 resolution levels,
       1 residual block per level, GroupNorm + SiLU, no attention."

    Args:
        in_channels   : number of input image channels (1 for MNIST/FashionMNIST).
        base_channels : width of the first encoder level (default 32).
        channel_mult  : tuple of multipliers for each level (default (1, 2, 4)).
        time_emb_dim  : dimension of the projected time embedding (default 256).
        dropout       : dropout probability in ResBlocks (default 0.1).
    """

    def __init__(
        self,
        in_channels:   int   = 1,
        base_channels: int   = 32,
        channel_mult:  tuple = (1, 2, 4),
        time_emb_dim:  int   = 256,
        dropout:       float = 0.1,
    ):
        super().__init__()

        # Channel widths at each resolution level
        chs = [base_channels * m for m in channel_mult]  # e.g. [32, 64, 128]

        # ── Timestep embedding pipeline ──────────────────────────────────────
        # sinusoidal(128) → Linear(128→256) → SiLU → Linear(256→256)
        # The output t_emb of shape (B, time_emb_dim) is passed to every ResBlock.
        sin_dim = base_channels * 4  # 128
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(sin_dim),
            nn.Linear(sin_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ── Stem: initial conv to lift input channels ────────────────────────
        self.stem = nn.Conv2d(in_channels, chs[0], kernel_size=3, padding=1)

        # ── Encoder ─────────────────────────────────────────────────────────
        # Level 0: 28×28, chs[0] → chs[0]  (32 → 32)
        self.enc0 = ResBlock(chs[0],  chs[0],  time_emb_dim, dropout)
        self.down0 = Downsample(chs[0])         # 28→14

        # Level 1: 14×14, chs[0] → chs[1]  (32 → 64)
        self.enc1 = ResBlock(chs[0],  chs[1],  time_emb_dim, dropout)
        self.down1 = Downsample(chs[1])         # 14→7

        # ── Bottleneck ────────────────────────────────────────────────────────
        # Level 2: 7×7, chs[1] → chs[2]  (64 → 128)
        self.mid = ResBlock(chs[1], chs[2], time_emb_dim, dropout)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Up-block 1: 7→14, concat skip from enc1 (chs[1]=64)
        self.up1   = Upsample(chs[2])            # 7→14,  chs[2]=128
        # After concat: chs[2] + chs[1] = 192
        self.dec1  = ResBlock(chs[2] + chs[1], chs[1], time_emb_dim, dropout)

        # Up-block 0: 14→28, concat skip from enc0 (chs[0]=32)
        self.up0   = Upsample(chs[1])            # 14→28, chs[1]=64
        # After concat: chs[1] + chs[0] = 96
        self.dec0  = ResBlock(chs[1] + chs[0], chs[0], time_emb_dim, dropout)

        # ── Output projection ────────────────────────────────────────────────
        # Final: GroupNorm + SiLU + 1×1 conv → predict ε̂ same shape as input
        self.out_norm = nn.GroupNorm(_num_groups(chs[0]), chs[0])
        self.out_conv = nn.Conv2d(chs[0], in_channels, kernel_size=3, padding=1)

        # Initialize last conv to zero for stable training at step 0
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise ε̂ from noisy image x_t and timestep t.

        Args:
            x_t : noisy image, shape (B, in_channels, H, W).
            t   : 0-indexed timestep tensor, shape (B,), values in [0, L-1].

        Returns:
            eps_hat : predicted noise, shape (B, in_channels, H, W).
        """
        # ── Timestep embedding ──────────────────────────────────────────────
        # This t_emb is passed into every ResBlock — it MUST reach the blocks
        # or the model is unconditional!
        t_emb = self.time_embed(t)                  # (B, time_emb_dim)

        # ── Stem ─────────────────────────────────────────────────────────────
        h = self.stem(x_t)                           # (B, 32, 28, 28)

        # ── Encoder ──────────────────────────────────────────────────────────
        h = self.enc0(h, t_emb)                      # (B, 32, 28, 28)
        skip0 = h                                    # save for skip connection

        h = self.down0(h)                            # (B, 32, 14, 14)
        h = self.enc1(h, t_emb)                      # (B, 64, 14, 14)
        skip1 = h                                    # save for skip connection

        h = self.down1(h)                            # (B, 64,  7,  7)

        # ── Bottleneck ────────────────────────────────────────────────────────
        h = self.mid(h, t_emb)                       # (B, 128, 7,  7)

        # ── Decoder ──────────────────────────────────────────────────────────
        h = self.up1(h)                              # (B, 128, 14, 14)
        h = torch.cat([h, skip1], dim=1)             # (B, 192, 14, 14)
        h = self.dec1(h, t_emb)                      # (B,  64, 14, 14)

        h = self.up0(h)                              # (B,  64, 28, 28)
        h = torch.cat([h, skip0], dim=1)             # (B,  96, 28, 28)
        h = self.dec0(h, t_emb)                      # (B,  32, 28, 28)

        # ── Output ───────────────────────────────────────────────────────────
        h = self.out_norm(h)
        h = F.silu(h)
        eps_hat = self.out_conv(h)                   # (B, 1,  28, 28)

        return eps_hat

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
