DDPM — pseudocode
Task 0: Data Loading

Load FashionMNIST images (28×28 grayscale, values 0–255)
Scale each pixel to [0,1] via divide by 255
Shift to [-1,1] by subtracting 0.5 and dividing by 0.5
Return batches of 128 images shuffled each epoch



Task 1: Noise Schedule

BUILD SCHEDULE (linear):
  Space beta values evenly from beta_min to beta_max over L steps
  
BUILD SCHEDULE (cosine):
  Compute cumulative alpha_bar using cosine curve
  Derive betas from consecutive alpha_bar ratios
  Clip betas to max 0.02 to prevent explosion
  
PRECOMPUTE from betas:
  alpha = 1 - beta  (signal kept per step)
  alpha_bar = running product of all alphas up to t  (total signal kept)
  sqrt_alpha_bar, sqrt_one_minus_alpha_bar  (for forward process)
  beta_tilde = posterior variance  (for reverse process)
  SNR = alpha_bar / (1 - alpha_bar)




Task 2: Forward Process

FORWARD (q_sample):
  Given clean image x0 and timestep t:
  Look up sqrt_alpha_bar[t] and sqrt_one_minus_alpha_bar[t]
  Sample random Gaussian noise eps
  Return  sqrt_alpha_bar[t] * x0  +  sqrt_one_minus_alpha_bar[t] * eps
  (This jumps directly to any noise level in one shot)



Task 3: Posterior (Reverse Step Math)

POSTERIOR MEAN (given x0 known — "teacher"):
  Use eq(6): weighted combination of x_t and x0
  coef1 * x0  +  coef2 * x_t
  (where coefs come from precomputed schedule tensors)

PREDICT x0 FROM eps (model's estimate):
  x0_hat = (x_t  -  sqrt_one_minus_alpha_bar[t] * eps_hat)  /  sqrt_alpha_bar[t]

REVERSE MEAN FROM eps (eq 8 — what model uses):
  mu = (1/sqrt_alpha[t])  *  (x_t  -  beta[t]/sqrt_one_minus_alpha_bar[t] * eps_hat)

ONE REVERSE STEP (p_sample_step):
  Compute mu from eps_hat using eq(8)
  If t > 0:  add Gaussian noise scaled by sqrt(beta_tilde[t])
  If t = 0:  return mu directly (no noise on final step)



Task 4: U-Net Architecture

TIMESTEP EMBEDDING:
  Take integer t (e.g. 500)
  Compute sin and cos at many frequencies  →  128-dim vector
  Pass through two linear layers  →  256-dim embedding

RESBLOCK:
  Input: feature map + timestep embedding
  Apply GroupNorm → SiLU → Conv
  Project timestep embedding to same channel count, add as bias
  Apply GroupNorm → SiLU → Conv
  Add skip connection (with 1x1 conv if channels changed)

UNET:
  Encoder: ResBlock at 28x28 → downsample to 14x14 → downsample to 7x7
  Bottleneck: two ResBlocks at 7x7
  Decoder: upsample to 14x14 → upsample to 28x28  (with skip connections from encoder)
  Final 1x1 conv → predicted noise (same shape as input)





Task 5: Training Loop

TRAIN:
  For each training step:
    Sample batch of real images x0
    For each image, sample a random timestep t from {0, ..., 999}
    Sample Gaussian noise eps
    Corrupt image:  x_t = forward_process(x0, t, eps)
    Predict noise:  eps_hat = UNet(x_t, t)
    Compute loss:   MSE between eps and eps_hat
    Backpropagate, clip gradients, update weights




Task 6: Ancestral Sampling (Generation)

GENERATE:
  Start from pure Gaussian noise x at shape (batch, 1, 28, 28)
  For t = 999 down to 0:
    Predict noise:  eps_hat = UNet(x, t)
    Compute reverse mean mu from eps_hat
    If t > 0: x = mu + small Gaussian noise
    If t = 0: x = mu  (final clean output)
  Return x  (generated images in [-1,1])
  
  To display: rescale from [-1,1] to [0,1]




Task 6b: DDIM Sampling

DDIM GENERATE (faster, fewer steps):
  Pick S timesteps evenly spaced (e.g. 50 out of 1000)
  Start from pure Gaussian noise x
  For each consecutive pair (t_current, t_prev) in reverse:
    Predict noise:  eps_hat = UNet(x, t_current)
    Estimate clean image:  x0_hat = (x - noise_component) / signal_component
    Deterministic update: blend x0_hat and eps_hat using next schedule values
    (No stochastic noise added if eta=0)
  Return x 



  
Task 7: Evaluation

FEATURE EXTRACTOR:
  Train small CNN classifier on real FashionMNIST once
  Save weights — reuse same frozen CNN for all evaluations

FID:
  Extract 256-dim features from 10,000 real test images
  Extract 256-dim features from 10,000 generated images
  Compute mean and covariance of each set
  FID = distance between the two Gaussian distributions

KID:
  Same features, but use kernel trick (polynomial MMD)
  More reliable than FID at small sample counts

CLASSIFIER METRICS:
  Run generated images through trained CNN
  high_conf_fraction: what fraction get a confident class prediction
  class_entropy: how uniformly spread across all 10 classes (diversity check)

NEAREST NEIGHBOR (memorization check):
  For each generated image, find its closest real training image by pixel distance
  If mean distance is near zero → model is copying training data (bad)
  If mean distance is large → model is generalizing (good)

PROXY BPD:
  Run test images through the model at random timesteps
  Average MSE loss → convert to bits-per-pixel
  Lower = model assigns higher probability to real data