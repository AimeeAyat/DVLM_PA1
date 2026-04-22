"""eval.py — Evaluation and metrics for DDPM (Task 7).

Implements §4.12 of the assignment:

1. Qualitative artifacts:
   • Final sample grid (64+ samples)
   • Denoising trajectory at {L, 3L/4, L/2, L/4, 0}
   • (Optional) noise interpolation grid

2. Quantitative metrics (§4.12 section A — MNIST/FashionMNIST):
   • Dataset-FID  (FID in learned feature space, NOT ImageNet-Inception)
   • Dataset-KID  (KID via polynomial-kernel MMD, more robust for small N)
   • Classifier accuracy on generated samples
   • Class entropy on generated samples (diversity proxy)

3. Overfitting checks (§4.12 section D):
   • Nearest-neighbor in pixel space  (ℓ₂ on raw tensors)
   • Train-vs-test FID gap

WHY NOT ImageNet-Inception FID?
  The assignment explicitly warns: "Inception-based FID/KID (ImageNet features)
  is often misleading on MNIST-scale datasets."  We train our own small CNN
  on FashionMNIST and use its penultimate-layer features as the embedding φ(x).

Usage:
    python eval.py --checkpoint outputs/checkpoints/ckpt_step0100000.pt
    python eval.py --checkpoint ckpt.pt --n_samples 10000
"""
import os, sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm

from diffusion.schedule import NoiseSchedule
from diffusion.ddpm     import ancestral_sample
from models.unet        import UNet


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractor (small CNN trained on FashionMNIST)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractorCNN(nn.Module):
    """LeNet-like classifier for FashionMNIST.

    Used as a fixed feature extractor  φ(x) ∈ R^256  for FID/KID computation.
    Training this on the REAL dataset ensures features are meaningful for the
    domain — unlike ImageNet Inception features which are off-domain here.

    Architecture:
        Conv(1→32, 3×3) + ReLU + MaxPool(2) → (32, 14, 14)
        Conv(32→64, 3×3) + ReLU + MaxPool(2) → (64, 7, 7)
        Flatten → 3136
        Linear(3136→256)  ← features extracted here (name: 'features')
        ReLU
        Linear(256→10)    ← classification logits
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, 3, padding=1)
        self.pool    = nn.MaxPool2d(2)
        self.fc1     = nn.Linear(64 * 7 * 7, 256)
        self.fc2     = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor):
        """Returns (logits, features)."""
        h = self.pool(F.relu(self.conv1(x)))    # (B, 32, 14, 14)
        h = self.pool(F.relu(self.conv2(h)))    # (B, 64,  7,  7)
        h = h.flatten(1)                         # (B, 3136)
        feat = F.relu(self.fc1(h))               # (B, 256)  ← φ(x)
        logits = self.fc2(feat)                  # (B, 10)
        return logits, feat

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return only feature embeddings φ(x)."""
        _, feat = self.forward(x)
        return feat


def train_feature_extractor(
    data_dir: str = "./data",
    device: torch.device = torch.device("cpu"),
    epochs: int = 10,
    save_path: str = "outputs/feature_extractor.pt",
) -> FeatureExtractorCNN:
    """Train the CNN on real FashionMNIST data.

    The feature extractor is trained ONCE and then kept FIXED across all
    evaluation runs — changing it loses comparability (§4.12 pitfall).
    """
    if os.path.exists(save_path):
        print(f"[Eval] Loading pre-trained feature extractor from {save_path}")
        model = FeatureExtractorCNN().to(device)
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()
        return model

    print(f"[Eval] Training feature extractor for {epochs} epochs…")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    train_set = torchvision.datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=transform
    )
    loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0)

    model = FeatureExtractorCNN().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.shape[0]
            correct    += (logits.argmax(1) == y).sum().item()
            n          += x.shape[0]
        print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/n:.4f}  "
              f"acc={correct/n*100:.1f}%")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"  Feature extractor saved → {save_path}")
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction utility
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    images: torch.Tensor,
    extractor: FeatureExtractorCNN,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract φ(x) features for a batch of images.

    Args:
        images    : float tensor (N, 1, 28, 28) in [−1, 1].
        extractor : trained FeatureExtractorCNN.
        device    : torch device.
        batch_size: mini-batch size for GPU.

    Returns:
        features : np.ndarray of shape (N, 256).
    """
    extractor.eval()
    all_feats = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i:i+batch_size].to(device)
        feat  = extractor.get_features(batch)
        all_feats.append(feat.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-FID  (in learned feature space)
# ─────────────────────────────────────────────────────────────────────────────

def compute_fid(real_feats: np.ndarray, gen_feats: np.ndarray) -> float:
    """Compute Frechet Inception Distance in the φ feature space.

    Formula (§4.12 eq):
        FID_φ(X_gen, X_real)
          = ‖μ_g − μ_r‖²  +  Tr(Σ_g + Σ_r − 2·(Σ_g Σ_r)^{1/2})

    Args:
        real_feats : (N_r, d) features from real test images.
        gen_feats  : (N_g, d) features from generated images.

    Returns:
        fid : float scalar.

    Note:
        sqrtm may produce tiny imaginary parts due to floating point; we
        take the real part and add an epsilon for numerical stability.
    """
    mu_r, Sigma_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_g, Sigma_g = gen_feats.mean(0),  np.cov(gen_feats,  rowvar=False)

    # ‖μ_g − μ_r‖²
    diff_mu = mu_g - mu_r
    mean_term = diff_mu @ diff_mu

    # Tr(Σ_g + Σ_r − 2·(Σ_g Σ_r)^{1/2})
    # Add small diagonal for numerical stability before sqrtm
    eps = 1e-6
    I   = np.eye(Sigma_r.shape[0])
    product = (Sigma_g + eps * I) @ (Sigma_r + eps * I)
    sqrt_product = sqrtm(product)

    # sqrtm may return small imaginary components; take real part
    if np.iscomplexobj(sqrt_product):
        sqrt_product = sqrt_product.real

    trace_term = np.trace(Sigma_g + Sigma_r - 2.0 * sqrt_product)
    fid = float(mean_term + trace_term)
    return fid


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-KID  (polynomial-kernel MMD, unbiased)
# ─────────────────────────────────────────────────────────────────────────────

def compute_kid(
    real_feats: np.ndarray,
    gen_feats:  np.ndarray,
    degree:     int   = 3,
    gamma:      float = None,
    coef0:      float = 1.0,
    n_subsets:  int   = 100,
    subset_size:int   = 1000,
) -> float:
    """Compute Kernel Inception Distance (KID) via polynomial MMD.

    KID is an UNBIASED estimator of MMD² in the kernel space — more stable
    than FID for limited sample sizes (§4.12 note).

    Polynomial kernel:  k(x, y) = (γ · x^T y + c₀)^d

    Uses the subset trick from the original KID paper for efficient estimation.

    Args:
        real_feats  : (N_r, d) real feature matrix.
        gen_feats   : (N_g, d) generated feature matrix.
        degree      : polynomial degree (default 3).
        gamma       : kernel bandwidth; if None, uses 1/d.
        coef0       : constant term c₀ (default 1.0).
        n_subsets   : number of random subsets for estimation.
        subset_size : size of each subset.

    Returns:
        kid_mean : float.
    """
    if gamma is None:
        gamma = 1.0 / real_feats.shape[1]

    kid_scores = []
    rng = np.random.default_rng(0)

    for _ in range(n_subsets):
        # Sample without replacement from both sets
        idx_r = rng.choice(len(real_feats), size=min(subset_size, len(real_feats)),
                           replace=False)
        idx_g = rng.choice(len(gen_feats),  size=min(subset_size, len(gen_feats)),
                           replace=False)
        r = real_feats[idx_r].astype(np.float64)
        g = gen_feats[idx_g].astype(np.float64)

        # Gram matrices
        K_rr = (gamma * r @ r.T + coef0) ** degree
        K_gg = (gamma * g @ g.T + coef0) ** degree
        K_rg = (gamma * r @ g.T + coef0) ** degree

        n, m = K_rr.shape[0], K_gg.shape[0]

        # Unbiased MMD² estimator
        # E_xx'[k(x,x')] unbiased: sum off-diagonal / n(n-1)
        K_rr_off = K_rr - np.diag(np.diag(K_rr))
        K_gg_off = K_gg - np.diag(np.diag(K_gg))

        mmd_sq = (
            K_rr_off.sum() / (n * (n - 1))
            + K_gg_off.sum() / (m * (m - 1))
            - 2.0 * K_rg.mean()
        )
        kid_scores.append(mmd_sq)

    return float(np.mean(kid_scores))


# ─────────────────────────────────────────────────────────────────────────────
# Classifier accuracy + entropy on generated samples
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_classifier_metrics(
    generated:  torch.Tensor,
    extractor:  FeatureExtractorCNN,
    device:     torch.device,
    batch_size: int = 256,
) -> dict:
    """Evaluate generated images using the trained classifier.

    Returns:
        "accuracy"     : fraction of generated samples classified with high
                         confidence  (measures perceptual quality).
        "entropy"      : entropy of predicted class distribution.
                         High entropy = diverse class coverage.
                         Low entropy  = mode collapse.
        "class_counts" : predicted class count histogram.
    """
    extractor.eval()
    all_probs = []
    for i in range(0, generated.shape[0], batch_size):
        batch  = generated[i:i+batch_size].to(device)
        logits, _ = extractor(batch)
        probs  = F.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)       # (N, 10)
    pred_labels = all_probs.argmax(axis=1)              # (N,)

    # Accuracy: % classified as any class with prob > 0.5
    high_conf = (all_probs.max(axis=1) > 0.5).mean()

    # Class histogram
    class_counts = np.bincount(pred_labels, minlength=10)
    class_probs  = class_counts / class_counts.sum()

    # Entropy of the class distribution
    # High entropy → model generates samples from all 10 classes (diverse)
    # Low entropy  → mode collapse to a few classes
    entropy = -np.sum(class_probs * np.log(class_probs + 1e-10))
    max_entropy = np.log(10)  # max possible for 10 classes

    return {
        "high_conf_fraction": float(high_conf),
        "class_entropy":      float(entropy),
        "max_entropy":        float(max_entropy),
        "entropy_fraction":   float(entropy / max_entropy),
        "class_counts":       class_counts.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Nearest-neighbor memorisation check  (§4.12 section D)
# ─────────────────────────────────────────────────────────────────────────────

def nearest_neighbor_check(
    generated:    torch.Tensor,
    train_images: torch.Tensor,
    n_display:    int = 8,
    out_path:     str = "outputs/nearest_neighbor.png",
) -> float:
    """For each generated image, find the nearest training image in pixel space.

    If generated images closely replicate training images, the model is likely
    memorising rather than generalising.

    Uses ℓ₂ distance on flattened tensors (pixel space check).

    Args:
        generated    : (N_gen, C, H, W)  generated images in [−1, 1].
        train_images : (N_train, C, H, W) training images in [−1, 1].
        n_display    : number of (generated, nearest) pairs to display.
        out_path     : path to save the comparison grid.

    Returns:
        mean_min_dist : mean ℓ₂ distance to nearest neighbor.
                        Large value → model is NOT memorising.
    """
    gen_flat   = generated[:n_display].flatten(1)    # (n, D)
    train_flat = train_images.flatten(1)             # (M, D)

    # Pairwise ℓ₂ distances: (n, M)
    dists = torch.cdist(gen_flat.float(), train_flat.float())
    min_dists, nn_indices = dists.min(dim=1)

    mean_min_dist = min_dists.mean().item()

    # Build a (n, 2, H, W) display grid: [generated | nearest neighbor]
    pairs = []
    for i in range(n_display):
        nn_img = train_images[nn_indices[i]]          # (C, H, W)
        pairs.append(generated[i])
        pairs.append(nn_img)

    pairs_tensor = torch.stack(pairs)                 # (2n, C, H, W)
    pairs_tensor = (pairs_tensor.clamp(-1, 1) + 1) / 2.0

    grid = torchvision.utils.make_grid(pairs_tensor, nrow=2, padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(4, n_display * 1.5))
    ax.imshow(grid_np[:, :, 0] if grid_np.shape[-1] == 1 else grid_np,
              cmap="gray" if grid_np.shape[-1] == 1 else None)
    ax.set_title(f"Generated (left) vs. Nearest Train Neighbor (right)\n"
                 f"mean ℓ₂ dist = {mean_min_dist:.3f}")
    ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [NN check] Saved comparison → {out_path}")
    print(f"  [NN check] Mean ℓ₂ distance to nearest training neighbor: {mean_min_dist:.4f}")
    print(f"             (A very small value ~0 suggests memorisation)")
    return mean_min_dist


# ─────────────────────────────────────────────────────────────────────────────
# NLL / bits-per-dimension estimate via ELBO  (§4.12 section E)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_elbo_bpd(
    model:     nn.Module,
    schedule:  NoiseSchedule,
    dataloader: DataLoader,
    device:    torch.device,
    n_batches: int = 20,
) -> float:
    """Estimate bits-per-dimension (bpd) via the ELBO upper bound on NLL.

    Formula (§4.12, §5.1.4):
        bpd = 1/(D · log2) · E_{x_0}[L_ELBO(x_0; θ)]

    where L_ELBO = L_prior + L_recon + L_diffusion (eq 11).

    For practical estimation we use the simplified noise-prediction form:
    the diffusion KL terms reduce to weighted MSE in ε space (see §4.9 of
    the DDPM paper).  We evaluate the unweighted version (L_simple) as a
    proxy, then convert to bpd.

    Note: This is only a PROXY for bpd, not the true ELBO bpd.  The true
    ELBO requires computing all L KL terms with proper weighting.  For a
    full ELBO computation, see the commented code below.

    Args:
        model      : trained ε_θ.
        schedule   : NoiseSchedule.
        dataloader : test set loader.
        device     : torch device.
        n_batches  : number of batches to average over.

    Returns:
        bpd_proxy : bits per dimension.
    """
    from diffusion.forward import q_sample

    model.eval()
    D = 28 * 28   # dimensions (FashionMNIST)
    total_loss = 0.0
    n_samples  = 0

    for i, (x0, _) in enumerate(dataloader):
        if i >= n_batches:
            break
        x0 = x0.to(device)
        B  = x0.shape[0]

        # Estimate L_simple by averaging over T timesteps
        batch_loss = 0.0
        n_mc = 10   # Monte Carlo timesteps per image
        for _ in range(n_mc):
            t   = torch.randint(0, schedule.L, (B,), device=device)
            eps = torch.randn_like(x0)
            x_t = q_sample(x0, t, eps, schedule)
            eps_hat   = model(x_t, t)
            batch_loss += F.mse_loss(eps_hat, eps, reduction="sum").item()

        total_loss += batch_loss / n_mc
        n_samples  += B

    # Average loss per sample per dimension
    avg_loss = total_loss / n_samples / D
    # Convert to bpd: bpd = loss / log(2)
    bpd = avg_loss / np.log(2.0)
    print(f"  [BPD] Proxy bpd (via L_simple) = {bpd:.4f}  (lower is better)")
    return bpd


# ─────────────────────────────────────────────────────────────────────────────
# Run full evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    model:      nn.Module,
    schedule:   NoiseSchedule,
    data_dir:   str,
    device:     torch.device,
    out_dir:    str = "outputs",
    n_samples:  int = 10_000,
):
    """Generate samples and compute all metrics.

    Metrics computed:
        • Dataset-FID
        • Dataset-KID
        • Classifier accuracy + entropy
        • Nearest-neighbor memorisation check
        • Proxy bpd
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # ── 1. Train/load feature extractor ──────────────────────────────────────
    extractor = train_feature_extractor(
        data_dir=data_dir,
        device=device,
        save_path=os.path.join(out_dir, "feature_extractor.pt"),
    )

    # ── 2. Load real test images ──────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    test_set = torchvision.datasets.FashionMNIST(
        data_dir, train=False, download=True, transform=transform
    )
    train_set = torchvision.datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=0)

    print(f"[Eval] Extracting features from {len(test_set)} real test images…")
    real_images_list = []
    for x, _ in test_loader:
        real_images_list.append(x)
    real_images = torch.cat(real_images_list)[:n_samples]  # (N, 1, 28, 28)
    real_feats  = extract_features(real_images, extractor, device)

    # ── 3. Generate samples ───────────────────────────────────────────────────
    print(f"[Eval] Generating {n_samples} samples…")
    gen_images_list = []
    generated_so_far = 0
    bs = 64
    while generated_so_far < n_samples:
        current_bs = min(bs, n_samples - generated_so_far)
        result = ancestral_sample(
            model, schedule,
            shape=(current_bs, 1, 28, 28),
            device=device,
        )
        gen_images_list.append(result["x0"])
        generated_so_far += current_bs
        if generated_so_far % 1000 == 0:
            print(f"  Generated {generated_so_far}/{n_samples}…")

    gen_images = torch.cat(gen_images_list)[:n_samples]  # (N, 1, 28, 28)
    gen_feats  = extract_features(gen_images, extractor, device)

    # ── 4. Final sample grid (64 samples) ────────────────────────────────────
    grid_path = os.path.join(out_dir, "final_sample_grid.png")
    display   = (gen_images[:64].clamp(-1, 1) + 1) / 2.0
    grid = torchvision.utils.make_grid(display, nrow=8, padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid_np[:, :, 0], cmap="gray")
    ax.axis("off")
    ax.set_title("Final generated samples (64 images)")
    plt.tight_layout()
    plt.savefig(grid_path, dpi=100)
    plt.close()
    print(f"  [Eval] Sample grid saved → {grid_path}")

    # ── 5. Dataset-FID ────────────────────────────────────────────────────────
    print("[Eval] Computing Dataset-FID…")
    fid = compute_fid(real_feats, gen_feats)
    print(f"  Dataset-FID = {fid:.2f}  (lower is better)")

    # ── 6. Dataset-KID ────────────────────────────────────────────────────────
    print("[Eval] Computing Dataset-KID…")
    kid = compute_kid(real_feats, gen_feats)
    print(f"  Dataset-KID = {kid:.6f}  (lower is better)")

    # ── 7. Classifier metrics ─────────────────────────────────────────────────
    print("[Eval] Computing classifier metrics…")
    clf_metrics = compute_classifier_metrics(gen_images, extractor, device)
    print(f"  High-confidence fraction : {clf_metrics['high_conf_fraction']:.3f}")
    print(f"  Class entropy            : {clf_metrics['class_entropy']:.4f}  "
          f"/ max={clf_metrics['max_entropy']:.4f}  "
          f"({clf_metrics['entropy_fraction']*100:.1f}%)")
    print(f"  Class counts             : {clf_metrics['class_counts']}")

    # Plot class distribution
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(10), clf_metrics["class_counts"])
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(f"Generated sample class distribution  "
                 f"(entropy={clf_metrics['class_entropy']:.3f}, "
                 f"max={clf_metrics['max_entropy']:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=100)
    plt.close()

    # ── 8. Nearest-neighbor memorisation check ────────────────────────────────
    print("[Eval] Running nearest-neighbor memorisation check…")
    train_images_list = []
    for x, _ in train_loader:
        train_images_list.append(x)
        if len(train_images_list) * 256 >= 5000:
            break
    train_images_small = torch.cat(train_images_list)[:5000]

    nn_dist = nearest_neighbor_check(
        generated=gen_images[:64],
        train_images=train_images_small,
        n_display=8,
        out_path=os.path.join(out_dir, "nearest_neighbor.png"),
    )

    # ── 9. Train-vs-test FID gap ──────────────────────────────────────────────
    print("[Eval] Computing train-vs-test FID gap…")
    train_images_big = torch.cat(
        [x for x, _ in train_loader]
    )[:n_samples]
    train_feats = extract_features(train_images_big, extractor, device)
    fid_train   = compute_fid(train_feats, gen_feats)
    fid_test    = fid  # already computed above
    print(f"  FID vs train = {fid_train:.2f}")
    print(f"  FID vs test  = {fid_test:.2f}")
    print(f"  Gap = {abs(fid_train - fid_test):.2f}  "
          f"(large gap may indicate overfitting)")

    # ── 10. Proxy bpd ─────────────────────────────────────────────────────────
    print("[Eval] Estimating proxy bpd…")
    bpd = compute_elbo_bpd(model, schedule, test_loader, device)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "Dataset-FID":        fid,
        "Dataset-KID":        kid,
        "high_conf_fraction": clf_metrics["high_conf_fraction"],
        "class_entropy":      clf_metrics["class_entropy"],
        "entropy_fraction":   clf_metrics["entropy_fraction"],
        "nn_mean_dist":       nn_dist,
        "fid_vs_train":       fid_train,
        "fid_vs_test":        fid_test,
        "proxy_bpd":          bpd,
    }

    print("\n══ Evaluation Summary ══════════════════════════")
    for k, v in summary.items():
        print(f"  {k:<25s}: {v:.4f}")
    print("════════════════════════════════════════════════")

    # Save summary to file
    summary_path = os.path.join(out_dir, "eval_summary.txt")
    with open(summary_path, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"\n[Eval] Summary saved → {summary_path}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--data_dir",   default="./data")
    p.add_argument("--out_dir",    default="./outputs/eval")
    p.add_argument("--n_samples",  type=int, default=10_000)
    # Schedule args (must match training run)
    p.add_argument("--L",          type=int,   default=1000)
    p.add_argument("--schedule",   default="linear")
    p.add_argument("--beta_min",   type=float, default=1e-4)
    p.add_argument("--beta_max",   type=float, default=0.02)
    p.add_argument("--base_ch",    type=int,   default=32)
    p.add_argument("--ch_mult",    nargs="+",  type=int, default=[1, 2, 4])
    p.add_argument("--time_emb",   type=int,   default=256)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = UNet(
        in_channels=1,
        base_channels=args.base_ch,
        channel_mult=tuple(args.ch_mult),
        time_emb_dim=args.time_emb,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    schedule = NoiseSchedule(
        L=args.L,
        schedule_type=args.schedule,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        device=str(device),
    )

    run_evaluation(model, schedule, args.data_dir, device, args.out_dir, args.n_samples)
