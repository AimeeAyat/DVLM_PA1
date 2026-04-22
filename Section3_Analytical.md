# Section 3: Analytical — Diffusion Models from First Principles

> **AI 623 — Programming Assignment 1**
> Based on course book Chapters 2, 3, and 4.

---

## Table of Contents

1. [§3.1 The Variational Perspective: DDPM](#31-the-variational-perspective-ddpm)
   - [3.1.1 From VAEs to Diffusion: The Hierarchical ELBO](#311-from-vaes-to-diffusion-the-hierarchical-elbo)
   - [3.1.2 The Forward Process and Its Closed-Form Marginal](#312-the-forward-process-and-its-closed-form-marginal)
   - [3.1.3 The Gaussian Posterior](#313-the-gaussian-posterior)
   - [3.1.4 The Conditioning Trick (Theorem 2.2.1)](#314-the-conditioning-trick-theorem-221)
   - [3.1.5 The ε-Prediction Reparameterisation](#315-the-ε-prediction-reparameterisation)
   - [3.1.6 ELBO Decomposition and the Simple Loss](#316-elbo-decomposition-and-the-simple-loss)
   - [3.1.7 Ancestral Sampling Algorithm](#317-ancestral-sampling-algorithm)
2. [§3.2 The Score-Based Perspective: NCSN](#32-the-score-based-perspective-ncsn)
   - [3.2.1 Energy-Based Models and the Partition Problem](#321-energy-based-models-and-the-partition-problem)
   - [3.2.2 Score Functions and Score Matching](#322-score-functions-and-score-matching)
   - [3.2.3 Denoising Score Matching (DSM)](#323-denoising-score-matching-dsm)
   - [3.2.4 Tweedie's Formula](#324-tweedies-formula)
   - [3.2.5 Noise-Conditional Score Networks (NCSN)](#325-noise-conditional-score-networks-ncsn)
   - [3.2.6 Score–DDPM Equivalence](#326-scoreddpm-equivalence)
3. [§3.3 The Score SDE Framework](#33-the-score-sde-framework)
   - [3.3.1 The Forward SDE](#331-the-forward-sde)
   - [3.3.2 VP-SDE: DDPM as a Continuous Limit](#332-vp-sde-ddpm-as-a-continuous-limit)
   - [3.3.3 Anderson's Reverse-Time SDE](#333-andersons-reverse-time-sde)
   - [3.3.4 The Probability Flow ODE](#334-the-probability-flow-ode)
   - [3.3.5 Continuous Denoising Score Matching](#335-continuous-denoising-score-matching)
   - [3.3.6 Generalised Perturbation Kernel (Lemma 4.4.1)](#336-generalised-perturbation-kernel-lemma-441)
4. [§3.4 Unified Comparison](#34-unified-comparison)

---

## §3.1 The Variational Perspective: DDPM

### 3.1.1 From VAEs to Diffusion: The Hierarchical ELBO

A **Variational Autoencoder** (VAE) introduces a latent variable $\mathbf{z}$ and maximises a lower bound on $\log p(\mathbf{x})$:

$$
\log p(\mathbf{x}) \;\geq\; \mathcal{L}_{\text{VAE}} \;=\; \underbrace{\mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\bigl[\log p(\mathbf{x}|\mathbf{z})\bigr]}_{\text{reconstruction}} \;-\; \underbrace{D_{\mathrm{KL}}\!\bigl(q(\mathbf{z}|\mathbf{x})\,\|\,p(\mathbf{z})\bigr)}_{\text{regularisation}}
$$

The bound comes from Jensen's inequality applied to the concave logarithm:

$$
\log p(\mathbf{x}) = \log \int p(\mathbf{x},\mathbf{z})\,d\mathbf{z} = \log \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\!\left[\frac{p(\mathbf{x},\mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\right] \geq \mathbb{E}_{q}\!\left[\log\frac{p(\mathbf{x},\mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\right]
$$

A **DDPM** is a hierarchical VAE with $L$ latent variables $\mathbf{x}_1, \ldots, \mathbf{x}_L$ of the **same dimension** as $\mathbf{x}_0$. The joint model factorises as:

$$
p_\theta(\mathbf{x}_{0:L}) = p(\mathbf{x}_L)\prod_{i=1}^{L} p_\theta(\mathbf{x}_{i-1}|\mathbf{x}_i), \qquad p(\mathbf{x}_L) = \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

The approximate posterior (encoder) is the **fixed** Markov chain:

$$
q(\mathbf{x}_{1:L}|\mathbf{x}_0) = \prod_{i=1}^{L} q(\mathbf{x}_i|\mathbf{x}_{i-1})
$$

The **ELBO** for this hierarchical model is:

$$
\log p_\theta(\mathbf{x}_0) \;\geq\; \mathbb{E}_{q(\mathbf{x}_{1:L}|\mathbf{x}_0)}\!\left[\log\frac{p_\theta(\mathbf{x}_{0:L})}{q(\mathbf{x}_{1:L}|\mathbf{x}_0)}\right] =: \mathcal{L}_{\text{ELBO}}
$$

---

### 3.1.2 The Forward Process and Its Closed-Form Marginal

Each forward step adds Gaussian noise parameterised by $\beta_i \in (0,1)$:

$$
q(\mathbf{x}_i \mid \mathbf{x}_{i-1}) = \mathcal{N}\!\left(\mathbf{x}_i;\; \sqrt{1-\beta_i}\,\mathbf{x}_{i-1},\; \beta_i\,\mathbf{I}\right)
$$

**Defining** $\alpha_i = 1 - \beta_i$ and $\bar{\alpha}_i = \prod_{j=1}^{i}\alpha_j$, we can derive the **closed-form marginal** $q(\mathbf{x}_i|\mathbf{x}_0)$ in one shot.

**Proof by induction.** Base case ($i=1$):

$$
\mathbf{x}_1 = \sqrt{\alpha_1}\,\mathbf{x}_0 + \sqrt{1-\alpha_1}\,\boldsymbol{\epsilon}_1 \sim \mathcal{N}(\sqrt{\alpha_1}\,\mathbf{x}_0,\,(1-\alpha_1)\mathbf{I}) = \mathcal{N}(\sqrt{\bar\alpha_1}\,\mathbf{x}_0,\,(1-\bar\alpha_1)\mathbf{I})
$$

**Inductive step.** Assume $q(\mathbf{x}_{i-1}|\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha_{i-1}}\,\mathbf{x}_0,\,(1-\bar\alpha_{i-1})\mathbf{I})$. Then:

$$
\mathbf{x}_i = \sqrt{\alpha_i}\,\mathbf{x}_{i-1} + \sqrt{1-\alpha_i}\,\boldsymbol{\epsilon}_i
$$

Since both $\mathbf{x}_{i-1}$ and $\boldsymbol{\epsilon}_i$ are Gaussian (and independent), $\mathbf{x}_i|\mathbf{x}_0$ is also Gaussian with:

$$
\mathbb{E}[\mathbf{x}_i|\mathbf{x}_0] = \sqrt{\alpha_i}\cdot\sqrt{\bar\alpha_{i-1}}\,\mathbf{x}_0 = \sqrt{\alpha_i\bar\alpha_{i-1}}\,\mathbf{x}_0 = \sqrt{\bar\alpha_i}\,\mathbf{x}_0
$$

$$
\mathrm{Var}[\mathbf{x}_i|\mathbf{x}_0] = \alpha_i(1-\bar\alpha_{i-1}) + (1-\alpha_i) = \alpha_i - \alpha_i\bar\alpha_{i-1} + 1 - \alpha_i = 1 - \alpha_i\bar\alpha_{i-1} = 1-\bar\alpha_i
$$

Therefore:

$$
\boxed{q(\mathbf{x}_i \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_i;\;\sqrt{\bar\alpha_i}\,\mathbf{x}_0,\;(1-\bar\alpha_i)\,\mathbf{I}\right)}
$$

The **reparameterised sample** is:

$$
\mathbf{x}_i = \sqrt{\bar\alpha_i}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_i}\,\boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})
$$

Note that as $i\to L$, $\bar\alpha_i\to 0$, so $\mathbf{x}_L \approx \mathcal{N}(\mathbf{0},\mathbf{I})$ — the data is destroyed.

---

### 3.1.3 The Gaussian Posterior

By Bayes' theorem:

$$
q(\mathbf{x}_{i-1}\mid\mathbf{x}_i,\mathbf{x}_0) = \frac{q(\mathbf{x}_i\mid\mathbf{x}_{i-1})\,q(\mathbf{x}_{i-1}\mid\mathbf{x}_0)}{q(\mathbf{x}_i\mid\mathbf{x}_0)}
$$

All three factors are Gaussian, so the posterior is also Gaussian. Computing its parameters by completing the square in the exponent:

$$
\log q(\mathbf{x}_{i-1}|\mathbf{x}_i,\mathbf{x}_0) = -\frac{1}{2}\left[\frac{(\mathbf{x}_i - \sqrt{\alpha_i}\,\mathbf{x}_{i-1})^2}{\beta_i} + \frac{(\mathbf{x}_{i-1} - \sqrt{\bar\alpha_{i-1}}\,\mathbf{x}_0)^2}{1-\bar\alpha_{i-1}} - \frac{(\mathbf{x}_i - \sqrt{\bar\alpha_i}\,\mathbf{x}_0)^2}{1-\bar\alpha_i}\right] + \text{const}
$$

Collecting terms in $\mathbf{x}_{i-1}^2$ and $\mathbf{x}_{i-1}$:

**Coefficient of $\mathbf{x}_{i-1}^2$:**

$$
-\frac{1}{2}\left(\frac{\alpha_i}{\beta_i} + \frac{1}{1-\bar\alpha_{i-1}}\right) = -\frac{1}{2\tilde\beta_i}, \quad \text{where } \tilde\beta_i = \frac{\beta_i(1-\bar\alpha_{i-1})}{1-\bar\alpha_i}
$$

**Coefficient of $\mathbf{x}_{i-1}$:**

$$
\frac{\sqrt{\alpha_i}}{\beta_i}\,\mathbf{x}_i + \frac{\sqrt{\bar\alpha_{i-1}}}{1-\bar\alpha_{i-1}}\,\mathbf{x}_0 = \frac{\tilde{\mu}_i(\mathbf{x}_i,\mathbf{x}_0)}{\tilde\beta_i}
$$

This gives the **posterior mean**:

$$
\boxed{\tilde{\mu}_i(\mathbf{x}_i,\mathbf{x}_0) = \frac{\sqrt{\alpha_i}(1-\bar\alpha_{i-1})}{1-\bar\alpha_i}\,\mathbf{x}_i + \frac{\sqrt{\bar\alpha_{i-1}}\,\beta_i}{1-\bar\alpha_i}\,\mathbf{x}_0}
$$

And the **posterior variance**:

$$
\tilde\beta_i = \frac{1-\bar\alpha_{i-1}}{1-\bar\alpha_i}\cdot\beta_i
$$

So:

$$
q(\mathbf{x}_{i-1}\mid\mathbf{x}_i,\mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_{i-1};\;\tilde\mu_i(\mathbf{x}_i,\mathbf{x}_0),\;\tilde\beta_i\,\mathbf{I}\right)
$$

---

### 3.1.4 The Conditioning Trick (Theorem 2.2.1)

The ELBO involves expectations over the full path $q(\mathbf{x}_{1:L}|\mathbf{x}_0)$, which is computationally demanding. The **conditioning trick** shows that optimising the full ELBO is equivalent to optimising a per-step conditional KL.

**Theorem 2.2.1 (Simplified ELBO Decomposition).** The negative ELBO decomposes as:

$$
-\mathcal{L}_{\text{ELBO}} = \underbrace{D_{\mathrm{KL}}(q(\mathbf{x}_L|\mathbf{x}_0)\,\|\,p(\mathbf{x}_L))}_{\mathcal{L}_L\ (\text{prior matching})} + \sum_{i=2}^{L}\underbrace{\mathbb{E}_{q(\mathbf{x}_i|\mathbf{x}_0)}\!\left[D_{\mathrm{KL}}(q(\mathbf{x}_{i-1}|\mathbf{x}_i,\mathbf{x}_0)\,\|\,p_\theta(\mathbf{x}_{i-1}|\mathbf{x}_i))\right]}_{\mathcal{L}_{i-1}\ (\text{denoising})} + \underbrace{\mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)}\!\left[-\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)\right]}_{\mathcal{L}_0\ (\text{reconstruction})}
$$

**Proof sketch.** Starting from the ELBO:

$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_q\!\left[\log\frac{p_\theta(\mathbf{x}_{0:L})}{q(\mathbf{x}_{1:L}|\mathbf{x}_0)}\right] = \mathbb{E}_q\!\left[\log p(\mathbf{x}_L) + \sum_{i=1}^L\log p_\theta(\mathbf{x}_{i-1}|\mathbf{x}_i) - \sum_{i=1}^L\log q(\mathbf{x}_i|\mathbf{x}_{i-1})\right]
$$

Using the Markov property of the forward process, $q(\mathbf{x}_i|\mathbf{x}_{i-1}) = q(\mathbf{x}_i|\mathbf{x}_{i-1},\mathbf{x}_0)$, and Bayes' rule:

$$
q(\mathbf{x}_i|\mathbf{x}_{i-1},\mathbf{x}_0) = \frac{q(\mathbf{x}_{i-1}|\mathbf{x}_i,\mathbf{x}_0)\,q(\mathbf{x}_i|\mathbf{x}_0)}{q(\mathbf{x}_{i-1}|\mathbf{x}_0)}
$$

Substituting and telescoping the marginals $q(\mathbf{x}_i|\mathbf{x}_0)/q(\mathbf{x}_{i-1}|\mathbf{x}_0)$ yields a sum of log-ratios. Recognising these as KL divergences and taking expectations with respect to only $q(\mathbf{x}_i|\mathbf{x}_0)$ (not the full path) gives the three-term decomposition above.

**Key insight:** Each denoising term $\mathcal{L}_{i-1}$ requires only the **two-dimensional marginal** $q(\mathbf{x}_i|\mathbf{x}_0)$, which we can sample in $O(1)$ time via the closed-form marginal. No full path simulation is needed.

---

### 3.1.5 The ε-Prediction Reparameterisation

The denoising KL terms compare $q(\mathbf{x}_{i-1}|\mathbf{x}_i,\mathbf{x}_0)=\mathcal{N}(\tilde\mu_i,\tilde\beta_i\mathbf{I})$ to a learnable $p_\theta(\mathbf{x}_{i-1}|\mathbf{x}_i)=\mathcal{N}(\mu_\theta(\mathbf{x}_i,i),\sigma_i^2\mathbf{I})$. The KL between two Gaussians with fixed variance is:

$$
D_{\mathrm{KL}}\bigl(\mathcal{N}(\tilde\mu,\sigma^2\mathbf{I})\,\|\,\mathcal{N}(\mu_\theta,\sigma^2\mathbf{I})\bigr) = \frac{1}{2\sigma^2}\|\tilde\mu - \mu_\theta\|^2
$$

So we learn $\mu_\theta(\mathbf{x}_i,i)$ to match $\tilde\mu_i(\mathbf{x}_i,\mathbf{x}_0)$.

The posterior mean can be rewritten by substituting $\mathbf{x}_0 = \frac{\mathbf{x}_i - \sqrt{1-\bar\alpha_i}\,\boldsymbol{\epsilon}}{\sqrt{\bar\alpha_i}}$:

$$
\tilde\mu_i = \frac{1}{\sqrt{\alpha_i}}\!\left(\mathbf{x}_i - \frac{\beta_i}{\sqrt{1-\bar\alpha_i}}\,\boldsymbol{\epsilon}\right)
$$

This suggests parameterising the network to **predict the noise**:

$$
\mu_\theta(\mathbf{x}_i,i) = \frac{1}{\sqrt{\alpha_i}}\!\left(\mathbf{x}_i - \frac{\beta_i}{\sqrt{1-\bar\alpha_i}}\,\hat\epsilon_\theta(\mathbf{x}_i,i)\right)
$$

The denoising loss becomes:

$$
\mathcal{L}_{i-1} \propto \mathbb{E}_{\mathbf{x}_0,\boldsymbol{\epsilon}}\!\left[\frac{\beta_i^2}{2\sigma_i^2\alpha_i(1-\bar\alpha_i)}\,\|\boldsymbol{\epsilon} - \hat\epsilon_\theta(\mathbf{x}_i,i)\|^2\right]
$$

Dropping the time-dependent weighting (Ho et al., 2020 empirical improvement) gives the **simple loss**:

$$
\boxed{\mathcal{L}_{\text{simple}} = \mathbb{E}_{i\sim\mathcal{U}[1,L],\;\mathbf{x}_0\sim p_{\text{data}},\;\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})}\!\left[\|\boldsymbol{\epsilon} - \hat\epsilon_\theta(\sqrt{\bar\alpha_i}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_i}\,\boldsymbol{\epsilon},\;i)\|^2\right]}
$$

This is the training objective implemented in `train.py`.

---

### 3.1.6 ELBO Decomposition and the Simple Loss

The three terms in the ELBO have distinct roles:

| Term | Expression | Role |
|------|-----------|------|
| $\mathcal{L}_L$ (prior) | $D_{\mathrm{KL}}(q(\mathbf{x}_L|\mathbf{x}_0)\,\|\,\mathcal{N}(\mathbf{0},\mathbf{I}))$ | Ensures $\mathbf{x}_L\approx\mathcal{N}(\mathbf{0},\mathbf{I})$; **constant** for fixed schedule |
| $\mathcal{L}_{i-1}$ (denoising) | $\mathbb{E}_{q(\mathbf{x}_i|\mathbf{x}_0)}[D_{\mathrm{KL}}(q(\mathbf{x}_{i-1}|\mathbf{x}_i,\mathbf{x}_0)\,\|\,p_\theta(\mathbf{x}_{i-1}|\mathbf{x}_i))]$ | Main learning signal: match reverse steps |
| $\mathcal{L}_0$ (reconstruction) | $\mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)}[-\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)]$ | Pixel-level reconstruction from mildly noisy input |

The **$\mathcal{L}_{\text{simple}}$** loss approximates $\sum_{i=1}^L\mathcal{L}_{i-1}$ with uniform timestep sampling and unit weighting. This drops the $1/(2\sigma_i^2\alpha_i(1-\bar\alpha_i))$ weight, which up-weights small-$i$ (low noise) steps in the true ELBO. Empirically, flat weighting gives better sample quality because it forces the network to also learn fine-grained denoising.

---

### 3.1.7 Ancestral Sampling Algorithm

Given a trained $\hat\epsilon_\theta$, generate samples by **reversing** the Markov chain:

```
Algorithm: DDPM Ancestral Sampling
  Input: trained noise predictor ε̂_θ, schedule {ᾱ_i, β_i, β̃_i}

  1. Sample x_L ~ N(0, I)
  2. For i = L, L-1, ..., 1:
       a. Predict noise:   ε̂ = ε̂_θ(x_i, i)
       b. Predict x_0:     x̂_0 = (x_i - √(1-ᾱ_i) ε̂) / √ᾱ_i
       c. Posterior mean:  μ̃_i = √α_i(1-ᾱ_{i-1})/(1-ᾱ_i) · x_i
                                + √ᾱ_{i-1} β_i/(1-ᾱ_i) · x̂_0
       d. if i > 1: x_{i-1} = μ̃_i + √β̃_i · z,  z ~ N(0,I)
          if i = 1: x_0 = μ̃_1  (no noise at final step)
  3. Return x_0
```

This is exactly `p_sample_step` in `diffusion/posterior.py` and `ancestral_sample` in `diffusion/ddpm.py`.

---

## §3.2 The Score-Based Perspective: NCSN

### 3.2.1 Energy-Based Models and the Partition Problem

An **Energy-Based Model** (EBM) defines a distribution via an unnormalised density:

$$
p_\theta(\mathbf{x}) = \frac{\exp(-E_\theta(\mathbf{x}))}{Z(\theta)}, \qquad Z(\theta) = \int \exp(-E_\theta(\mathbf{x}))\,d\mathbf{x}
$$

where $E_\theta:\mathbb{R}^d\to\mathbb{R}$ is the energy function. The **partition function** $Z(\theta)$ is intractable for high-dimensional $\mathbf{x}$ (e.g., images), as it requires integration over all of $\mathbb{R}^d$.

Directly maximising $\log p_\theta(\mathbf{x}) = -E_\theta(\mathbf{x}) - \log Z(\theta)$ requires computing $\nabla_\theta\log Z(\theta)$, which is itself an expensive expectation under $p_\theta$.

---

### 3.2.2 Score Functions and Score Matching

The **score function** of a distribution $p(\mathbf{x})$ is:

$$
s(\mathbf{x}) = \nabla_\mathbf{x}\log p(\mathbf{x})
$$

**Key property:** The score is **independent of the partition function**:

$$
\nabla_\mathbf{x}\log p_\theta(\mathbf{x}) = \nabla_\mathbf{x}\log\exp(-E_\theta(\mathbf{x})) - \nabla_\mathbf{x}\log Z(\theta) = -\nabla_\mathbf{x}E_\theta(\mathbf{x})
$$

since $Z(\theta)$ does not depend on $\mathbf{x}$. Thus we can train a **score network** $s_\theta(\mathbf{x})\approx\nabla_\mathbf{x}\log p_{\text{data}}(\mathbf{x})$ without ever computing $Z$.

The **Explicit Score Matching** (SM) objective minimises:

$$
\mathcal{L}_{\text{SM}} = \mathbb{E}_{p_{\text{data}}}\!\left[\tfrac{1}{2}\|s_\theta(\mathbf{x}) - \nabla_\mathbf{x}\log p_{\text{data}}(\mathbf{x})\|^2\right]
$$

This requires knowing $\nabla_\mathbf{x}\log p_{\text{data}}$, which is unknown. **Hyvärinen (2005)** showed, via integration by parts, that:

$$
\mathcal{L}_{\text{SM}} = \mathbb{E}_{p_{\text{data}}}\!\left[\mathrm{tr}(\nabla_\mathbf{x}s_\theta(\mathbf{x})) + \tfrac{1}{2}\|s_\theta(\mathbf{x})\|^2\right] + C
$$

where the constant $C$ does not depend on $\theta$. The trace term $\mathrm{tr}(\nabla_\mathbf{x}s_\theta)$ requires computing the full Jacobian diagonal — $O(d)$ backward passes for $d$-dimensional data — which is prohibitive for images.

---

### 3.2.3 Denoising Score Matching (DSM)

**Vincent (2011)** proposed a tractable alternative. Instead of fitting the clean data score, add noise $\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\sigma^2\mathbf{I})$ to obtain $\tilde\mathbf{x} = \mathbf{x} + \boldsymbol{\epsilon}$, and fit the score of the **noisy distribution** $p_\sigma(\tilde\mathbf{x}) = \int p_{\text{data}}(\mathbf{x})\,p_\sigma(\tilde\mathbf{x}|\mathbf{x})\,d\mathbf{x}$.

The **Denoising Score Matching** objective is:

$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}_{p_{\text{data}}(\mathbf{x}),\,p_\sigma(\tilde\mathbf{x}|\mathbf{x})}\!\left[\tfrac{1}{2}\|s_\theta(\tilde\mathbf{x}) - \nabla_{\tilde\mathbf{x}}\log p_\sigma(\tilde\mathbf{x}|\mathbf{x})\|^2\right]
$$

For Gaussian corruption $p_\sigma(\tilde\mathbf{x}|\mathbf{x}) = \mathcal{N}(\tilde\mathbf{x};\mathbf{x},\sigma^2\mathbf{I})$:

$$
\nabla_{\tilde\mathbf{x}}\log p_\sigma(\tilde\mathbf{x}|\mathbf{x}) = -\frac{\tilde\mathbf{x}-\mathbf{x}}{\sigma^2} = -\frac{\boldsymbol{\epsilon}}{\sigma^2}
$$

So:

$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}\!\left[\tfrac{1}{2}\left\|s_\theta(\tilde\mathbf{x}) + \frac{\boldsymbol{\epsilon}}{\sigma^2}\right\|^2\right]
$$

**Theorem (Vincent 2011).** $\mathcal{L}_{\text{SM}} = \mathcal{L}_{\text{DSM}} + C$ for a constant $C$ independent of $\theta$.

**Proof.** Expanding $\mathcal{L}_{\text{SM}}$ using Bayes:

$$
\nabla_{\tilde\mathbf{x}}\log p_\sigma(\tilde\mathbf{x}) = \mathbb{E}_{p(\mathbf{x}|\tilde\mathbf{x})}\!\left[\nabla_{\tilde\mathbf{x}}\log p_\sigma(\tilde\mathbf{x}|\mathbf{x})\right]
$$

Substituting into $\mathcal{L}_{\text{SM}}$ and expanding the squared norm:

$$
\mathcal{L}_{\text{SM}} = \mathbb{E}_{p_\sigma(\tilde\mathbf{x})}\!\left[\tfrac{1}{2}\|s_\theta\|^2 - s_\theta^\top\nabla\log p_\sigma\right] + C_1
$$

$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}_{p_{\text{data},\sigma}}\!\left[\tfrac{1}{2}\|s_\theta\|^2 - s_\theta^\top\nabla_{\tilde\mathbf{x}}\log p_\sigma(\tilde\mathbf{x}|\mathbf{x})\right] + C_2
$$

The two cross-terms are equal by the law of total expectation ($\mathbb{E}_{\tilde\mathbf{x}}[s_\theta^\top\nabla\log p_\sigma] = \mathbb{E}_{\mathbf{x},\tilde\mathbf{x}}[s_\theta^\top\nabla\log p_\sigma(\tilde\mathbf{x}|\mathbf{x})]$), so $\mathcal{L}_{\text{SM}} - \mathcal{L}_{\text{DSM}} = C_1 - C_2 = \text{const}$. $\blacksquare$

**Practical form** (substituting $\boldsymbol{\epsilon} = (\tilde\mathbf{x}-\mathbf{x})/\sigma$, reparameterising $s_\theta = -\hat\epsilon_\theta/\sigma$):

$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}\!\left[\|\boldsymbol{\epsilon} - \hat\epsilon_\theta(\mathbf{x}+\sigma\boldsymbol{\epsilon})\|^2\right]
$$

This is precisely the $\mathcal{L}_{\text{simple}}$ objective of DDPM with a single noise level $\sigma$.

---

### 3.2.4 Tweedie's Formula

**Tweedie's formula** relates the posterior mean of the clean signal to the score of the noisy distribution.

**Theorem (Tweedie's Formula).** For $\tilde\mathbf{x} = \sqrt{\bar\alpha}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha}\,\boldsymbol{\epsilon}$ with $\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$:

$$
\sqrt{\bar\alpha}\,\mathbb{E}[\mathbf{x}_0|\tilde\mathbf{x}] = \tilde\mathbf{x} + (1-\bar\alpha)\,\nabla_{\tilde\mathbf{x}}\log p(\tilde\mathbf{x})
$$

**Proof.** By definition of the conditional expectation:

$$
\mathbb{E}[\mathbf{x}_0|\tilde\mathbf{x}] = \int \mathbf{x}_0\,p(\mathbf{x}_0|\tilde\mathbf{x})\,d\mathbf{x}_0 = \frac{\int \mathbf{x}_0\,p(\tilde\mathbf{x}|\mathbf{x}_0)\,p(\mathbf{x}_0)\,d\mathbf{x}_0}{p(\tilde\mathbf{x})}
$$

Taking the gradient of $\log p(\tilde\mathbf{x}) = \log\int p(\tilde\mathbf{x}|\mathbf{x}_0)p(\mathbf{x}_0)d\mathbf{x}_0$:

$$
\nabla_{\tilde\mathbf{x}}\log p(\tilde\mathbf{x}) = \frac{\int \nabla_{\tilde\mathbf{x}}\log p(\tilde\mathbf{x}|\mathbf{x}_0)\cdot p(\tilde\mathbf{x}|\mathbf{x}_0)p(\mathbf{x}_0)\,d\mathbf{x}_0}{p(\tilde\mathbf{x})} = \mathbb{E}_{p(\mathbf{x}_0|\tilde\mathbf{x})}\!\left[\nabla_{\tilde\mathbf{x}}\log p(\tilde\mathbf{x}|\mathbf{x}_0)\right]
$$

Since $p(\tilde\mathbf{x}|\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha}\mathbf{x}_0,(1-\bar\alpha)\mathbf{I})$:

$$
\nabla_{\tilde\mathbf{x}}\log p(\tilde\mathbf{x}|\mathbf{x}_0) = -\frac{\tilde\mathbf{x} - \sqrt{\bar\alpha}\mathbf{x}_0}{1-\bar\alpha}
$$

Therefore:

$$
\nabla_{\tilde\mathbf{x}}\log p(\tilde\mathbf{x}) = \frac{-\tilde\mathbf{x} + \sqrt{\bar\alpha}\mathbb{E}[\mathbf{x}_0|\tilde\mathbf{x}]}{1-\bar\alpha}
$$

Rearranging:

$$
\sqrt{\bar\alpha}\,\mathbb{E}[\mathbf{x}_0|\tilde\mathbf{x}] = \tilde\mathbf{x} + (1-\bar\alpha)\,\nabla_{\tilde\mathbf{x}}\log p(\tilde\mathbf{x}) \qquad \blacksquare
$$

**Corollary.** The optimal noise predictor $\hat\epsilon_\theta^*$ and the optimal score $s^*$ are related by:

$$
s^*(\mathbf{x}_i,i) = \nabla_{\mathbf{x}_i}\log p(\mathbf{x}_i) = -\frac{\hat\epsilon_\theta^*(\mathbf{x}_i,i)}{\sqrt{1-\bar\alpha_i}}
$$

This is the fundamental **duality** between noise prediction and score matching.

---

### 3.2.5 Noise-Conditional Score Networks (NCSN)

A single noise level $\sigma$ is insufficient: the data distribution $p_{\text{data}}$ may be concentrated on a low-dimensional manifold, where $\nabla_\mathbf{x}\log p(\mathbf{x})$ is undefined away from the manifold. Moreover, Langevin dynamics with one $\sigma$ mixes poorly across modes.

**Song & Ermon (2019)** proposed training with a **geometric sequence** of noise levels $\sigma_1 < \sigma_2 < \cdots < \sigma_L$ where $\sigma_1\approx 0$ (near-clean) and $\sigma_L$ is large enough that $p_{\sigma_L}\approx\mathcal{N}(\mathbf{0},\sigma_L^2\mathbf{I})$.

The **NCSN objective** trains a single score network $s_\theta(\mathbf{x},\sigma)$ jointly across all noise levels:

$$
\mathcal{L}_{\text{NCSN}} = \frac{1}{L}\sum_{i=1}^L\lambda(\sigma_i)\,\mathbb{E}_{p_{\text{data}}(\mathbf{x}_0),\,p_{\sigma_i}(\tilde\mathbf{x}|\mathbf{x}_0)}\!\left[\tfrac{1}{2}\left\|s_\theta(\tilde\mathbf{x},\sigma_i) + \frac{\tilde\mathbf{x}-\mathbf{x}_0}{\sigma_i^2}\right\|^2\right]
$$

The weighting $\lambda(\sigma_i) = \sigma_i^2$ balances the different magnitude of the score at each noise level.

**Annealed Langevin Dynamics** samples from $p_{\text{data}}$ by iterating from large to small $\sigma$, running $K$ Langevin steps at each level:

$$
\tilde\mathbf{x}^{(k+1)} = \tilde\mathbf{x}^{(k)} + \frac{\epsilon_i}{2}\,s_\theta(\tilde\mathbf{x}^{(k)},\sigma_i) + \sqrt{\epsilon_i}\,\mathbf{z}^{(k)}, \quad \mathbf{z}^{(k)}\sim\mathcal{N}(\mathbf{0},\mathbf{I})
$$

As $\sigma\to 0$, this converges to samples from $p_{\text{data}}$.

---

### 3.2.6 Score–DDPM Equivalence

The DDPM noise prediction loss and the NCSN score matching loss are mathematically equivalent.

**Proof.** The NCSN conditional score target for noise level $\sigma_i$ corresponding to DDPM step $i$ (i.e., $\sigma_i^2 = 1-\bar\alpha_i$) is:

$$
\nabla_{\mathbf{x}_i}\log q(\mathbf{x}_i|\mathbf{x}_0) = -\frac{\mathbf{x}_i - \sqrt{\bar\alpha_i}\mathbf{x}_0}{1-\bar\alpha_i} = -\frac{\boldsymbol{\epsilon}}{\sqrt{1-\bar\alpha_i}}
$$

So fitting $s_\theta(\mathbf{x}_i,i) \approx -\boldsymbol{\epsilon}/\sqrt{1-\bar\alpha_i}$ is equivalent to fitting $\hat\epsilon_\theta(\mathbf{x}_i,i)\approx\boldsymbol{\epsilon}$ — precisely the DDPM loss. More precisely:

$$
s_\theta(\mathbf{x}_i,i) = -\frac{\hat\epsilon_\theta(\mathbf{x}_i,i)}{\sqrt{1-\bar\alpha_i}}
$$

Thus DDPM and NCSN are two parameterisations of the same underlying function: **the conditional score of the noisy distribution**. $\blacksquare$

---

## §3.3 The Score SDE Framework

### 3.3.1 The Forward SDE

Song et al. (2021) unified DDPM and NCSN under a **continuous-time stochastic differential equation** (SDE) framework. Consider a continuous-time process $\mathbf{x}(t)$, $t\in[0,T]$, governed by the Itô SDE:

$$
\boxed{d\mathbf{x} = f(\mathbf{x},t)\,dt + g(t)\,d\mathbf{w}}
$$

where:
- $f(\mathbf{x},t)$ is the **drift coefficient** (a vector field)
- $g(t)$ is the **diffusion coefficient** (a scalar)
- $\mathbf{w}$ is a standard Wiener process (Brownian motion)

The solution defines a **probability path** $\{p_t\}_{t=0}^T$ from $p_0 = p_{\text{data}}$ to $p_T \approx p_{\text{prior}}$.

The marginal densities $p_t$ evolve according to the **Fokker-Planck equation**:

$$
\frac{\partial p_t}{\partial t} = -\nabla\cdot(f\,p_t) + \frac{g^2}{2}\,\Delta p_t
$$

**Two canonical instantiations:**

| Model | $f(\mathbf{x},t)$ | $g(t)$ | $p_T$ |
|-------|------------------|--------|-------|
| VE-SDE (NCSN) | $\mathbf{0}$ | $\sqrt{d[\sigma^2(t)]/dt}$ | $\mathcal{N}(\mathbf{0},\sigma_{\max}^2\mathbf{I})$ |
| VP-SDE (DDPM) | $-\frac{1}{2}\beta(t)\mathbf{x}$ | $\sqrt{\beta(t)}$ | $\mathcal{N}(\mathbf{0},\mathbf{I})$ |

---

### 3.3.2 VP-SDE: DDPM as a Continuous Limit

The **Variance-Preserving SDE** (VP-SDE) corresponds to the DDPM forward process in the limit $L\to\infty$.

**Drift and diffusion.** For the VP-SDE:

$$
d\mathbf{x} = -\frac{\beta(t)}{2}\mathbf{x}\,dt + \sqrt{\beta(t)}\,d\mathbf{w}
$$

where $\beta(t):[0,T]\to\mathbb{R}_{>0}$ is a continuous noise schedule with $\int_0^T\beta(t)\,dt = B(T)$.

**Perturbation kernel derivation.** Given $\mathbf{x}(0)=\mathbf{x}_0$, we solve for the mean and variance of $\mathbf{x}(t)$.

*Mean:* The drift $-\frac{\beta(t)}{2}\mathbf{x}$ gives the ODE $\frac{d\bar\mathbf{x}}{dt} = -\frac{\beta(t)}{2}\bar\mathbf{x}$, which integrates to:

$$
\mathbb{E}[\mathbf{x}(t)|\mathbf{x}_0] = \mathbf{x}_0\exp\!\left(-\frac{1}{2}\int_0^t\beta(s)\,ds\right) = e^{-B(t)/2}\mathbf{x}_0
$$

where $B(t) = \int_0^t\beta(s)\,ds$.

*Variance:* Using the Itô isometry, for a linear SDE $d\mathbf{x} = a(t)\mathbf{x}\,dt + g(t)\,d\mathbf{w}$ with $a(t) = -\beta(t)/2$:

$$
\frac{dV(t)}{dt} = 2a(t)V(t) + g^2(t) = -\beta(t)V(t) + \beta(t)
$$

with $V(0)=0$. This integrates (via integrating factor $e^{B(t)}$) to:

$$
V(t) = e^{-B(t)}\int_0^t\beta(s)e^{B(s)}\,ds = e^{-B(t)}\left[e^{B(s)}\right]_0^t = 1 - e^{-B(t)}
$$

**Perturbation kernel:**

$$
\boxed{p_t(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\;e^{-B(t)/2}\mathbf{x}_0,\;(1-e^{-B(t)})\mathbf{I}\right)}
$$

**Connection to discrete DDPM.** Setting $t = i/L$, $\beta(t) = L\beta_i$, and letting $L\to\infty$:

$$
e^{-B(i/L)} = e^{-\sum_{j=1}^i\beta_j/L\cdot L} = e^{-\sum_{j=1}^i\beta_j} \approx \prod_{j=1}^i(1-\beta_j) = \bar\alpha_i
$$

So $e^{-B(t)/2}\to\sqrt{\bar\alpha_i}$ and $1-e^{-B(t)}\to 1-\bar\alpha_i$ — exactly the discrete DDPM marginal.

---

### 3.3.3 Anderson's Reverse-Time SDE

**Theorem (Anderson 1982).** The reverse-time SDE corresponding to the forward SDE $d\mathbf{x} = f(\mathbf{x},t)\,dt + g(t)\,d\mathbf{w}$ is:

$$
\boxed{d\mathbf{x} = \left[f(\mathbf{x},t) - g(t)^2\,\nabla_\mathbf{x}\log p_t(\mathbf{x})\right]dt + g(t)\,d\bar\mathbf{w}}
$$

where $\bar\mathbf{w}$ is Brownian motion running **backwards** in time ($dt < 0$), and $p_t(\mathbf{x})$ is the marginal density of the forward process.

**Key insight.** The reverse drift has an **additional score term** $-g^2\nabla\log p_t$ compared to the forward drift. This score term "pushes" the process toward high-density regions as time is reversed.

**For the VP-SDE** (with $f = -\beta\mathbf{x}/2$, $g=\sqrt{\beta}$):

$$
d\mathbf{x} = \left[-\frac{\beta(t)}{2}\mathbf{x} - \beta(t)\,\nabla_\mathbf{x}\log p_t(\mathbf{x})\right]dt + \sqrt{\beta(t)}\,d\bar\mathbf{w}
$$

Since $\nabla_\mathbf{x}\log p_t(\mathbf{x}) \approx s_\theta(\mathbf{x},t)$ from our trained score network, we can **simulate this reverse SDE** to generate samples.

**Discrete DDPM connection.** Discretising the reverse VP-SDE recovers exactly the DDPM ancestral sampling updates:

$$
\mathbf{x}_{i-1} = \frac{1}{\sqrt{\alpha_i}}\left(\mathbf{x}_i + \beta_i\,s_\theta(\mathbf{x}_i,i)\right) + \sqrt{\beta_i}\,\mathbf{z} = \frac{1}{\sqrt{\alpha_i}}\left(\mathbf{x}_i - \frac{\beta_i}{\sqrt{1-\bar\alpha_i}}\hat\epsilon_\theta(\mathbf{x}_i,i)\right) + \sqrt{\beta_i}\,\mathbf{z}
$$

using $s_\theta = -\hat\epsilon_\theta/\sqrt{1-\bar\alpha_i}$.

---

### 3.3.4 The Probability Flow ODE

For each SDE, there exists a corresponding **deterministic ODE** with the **same marginals** $\{p_t\}$:

$$
\boxed{\frac{d\mathbf{x}}{dt} = f(\mathbf{x},t) - \frac{g(t)^2}{2}\,\nabla_\mathbf{x}\log p_t(\mathbf{x})}
$$

This is the **Probability Flow ODE** (PF-ODE).

**Properties:**
1. **Same marginals:** $\mathbf{x}(t)\sim p_t$ for all $t$ — the PF-ODE trajectories have the same distribution as the SDE.
2. **Deterministic:** Given $\mathbf{x}(T)\sim p_T$, the ODE produces a unique $\mathbf{x}(0)\sim p_0 = p_{\text{data}}$.
3. **Exact likelihood:** Since the ODE is invertible (a diffeomorphism), the likelihood of a data point $\mathbf{x}_0$ is:

$$
\log p_0(\mathbf{x}_0) = \log p_T(\mathbf{x}(T)) - \int_0^T \nabla\cdot\tilde{f}(\mathbf{x}(t),t)\,dt
$$

where $\tilde{f} = f - \frac{g^2}{2}\nabla\log p_t$ (by the instantaneous change-of-variables formula for ODEs).

4. **DDIM connection:** Discretising the PF-ODE gives the **DDIM** (Denoising Diffusion Implicit Models) sampler:

$$
\mathbf{x}_{i-1} = \sqrt{\bar\alpha_{i-1}}\,\hat\mathbf{x}_0^{(i)} + \sqrt{1-\bar\alpha_{i-1}-\eta^2\tilde\beta_i}\,\hat\epsilon_\theta(\mathbf{x}_i,i) + \eta\sqrt{\tilde\beta_i}\,\mathbf{z}
$$

with $\eta=0$ for the pure ODE (deterministic, acceleratable) and $\eta=1$ for full DDPM stochasticity.

---

### 3.3.5 Continuous Denoising Score Matching

In the continuous SDE framework, the score network is trained via a **continuous-time DSM** objective:

$$
\mathcal{L}_{\text{cDSM}} = \mathbb{E}_{t\sim\mathcal{U}[0,T],\;\mathbf{x}_0\sim p_{\text{data}},\;\mathbf{x}_t\sim p_t(\cdot|\mathbf{x}_0)}\!\left[\lambda(t)\,\|s_\theta(\mathbf{x}_t,t) - \nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t|\mathbf{x}_0)\|^2\right]
$$

For the VP-SDE with $\lambda(t) = (1-e^{-B(t)})$ (to match the $1/\sigma^2$ scaling):

$$
\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t|\mathbf{x}_0) = -\frac{\mathbf{x}_t - e^{-B(t)/2}\mathbf{x}_0}{1-e^{-B(t)}} = -\frac{\boldsymbol{\epsilon}}{\sqrt{1-e^{-B(t)}}}
$$

Substituting $s_\theta = -\hat\epsilon_\theta/\sqrt{1-e^{-B(t)}}$ recovers the **continuous-time $\mathcal{L}_{\text{simple}}$**:

$$
\mathcal{L}_{\text{cDSM}} \propto \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\!\left[\|\boldsymbol{\epsilon} - \hat\epsilon_\theta(\mathbf{x}_t,t)\|^2\right]
$$

The discrete $\mathcal{L}_{\text{simple}}$ (with uniform $i\in\{1,\ldots,L\}$) is thus a **Monte Carlo approximation** of $\mathcal{L}_{\text{cDSM}}$ with quadrature at $L$ timestep nodes.

---

### 3.3.6 Generalised Perturbation Kernel (Lemma 4.4.1)

**Lemma 4.4.1** establishes that the perturbation kernel of any linear SDE (not just VP) can be written in closed form.

**Lemma.** For the linear SDE:

$$
d\mathbf{x} = f(t)\mathbf{x}\,dt + g(t)\,d\mathbf{w}
$$

(with scalar $f$ for simplicity), the perturbation kernel is Gaussian:

$$
p_t(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\;\mu(t)\mathbf{x}_0,\;\Sigma(t)\mathbf{I}\right)
$$

where $\mu(t)$ and $\Sigma(t)$ satisfy the ODEs:

$$
\dot\mu(t) = f(t)\mu(t), \qquad \mu(0) = 1
$$
$$
\dot\Sigma(t) = 2f(t)\Sigma(t) + g^2(t), \qquad \Sigma(0) = 0
$$

**Proof.** The solution $\mathbf{x}(t)$ of the linear SDE is:

$$
\mathbf{x}(t) = e^{\int_0^t f(s)ds}\mathbf{x}_0 + \int_0^t e^{\int_s^t f(u)du}\,g(s)\,d\mathbf{w}(s)
$$

The first term gives $\mu(t) = e^{\int_0^t f(s)ds}$, satisfying $\dot\mu = f\mu$, $\mu(0)=1$.

The second term is a Gaussian (as an Itô integral of a deterministic integrand) with variance:

$$
\Sigma(t) = \int_0^t e^{2\int_s^t f(u)du}\,g^2(s)\,ds
$$

Differentiating: $\dot\Sigma = 2f(t)\Sigma(t) + g^2(t)$, $\Sigma(0)=0$. $\blacksquare$

**Corollary (VP-SDE).** With $f(t) = -\beta(t)/2$ and $g(t) = \sqrt{\beta(t)}$:

$$
\mu(t) = e^{-B(t)/2}, \qquad \Sigma(t) = 1 - e^{-B(t)}
$$

recovering the VP perturbation kernel from §3.3.2.

**Corollary (VE-SDE).** With $f(t) = 0$ and $g^2(t) = d\sigma^2/dt$:

$$
\mu(t) = 1, \qquad \Sigma(t) = \sigma^2(t)
$$

so $p_t(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\mathbf{x}_0,\sigma^2(t)\mathbf{I})$ — pure additive noise, matching NCSN.

---

## §3.4 Unified Comparison

The three perspectives — variational, score-based, and SDE — are different descriptions of the **same underlying generative model**.

| Aspect | Variational (DDPM) | Score-Based (NCSN) | SDE Framework |
|--------|-------------------|-------------------|---------------|
| **Formulation** | Hierarchical VAE with fixed encoder | EBM + score matching | Continuous-time SDE |
| **Training objective** | $\mathcal{L}_{\text{simple}} = \mathbb{E}[\|\boldsymbol{\epsilon} - \hat\epsilon_\theta\|^2]$ | $\mathcal{L}_{\text{NCSN}} = \mathbb{E}[\|s_\theta + \boldsymbol{\epsilon}/\sigma\|^2]$ | $\mathcal{L}_{\text{cDSM}}$ (continuous version of same) |
| **Noise levels** | Discrete $i\in\{1,\ldots,L\}$ | Discrete $\sigma_1 < \cdots < \sigma_L$ | Continuous $t\in[0,T]$ |
| **What is learned** | Noise predictor $\hat\epsilon_\theta(\mathbf{x}_i,i)$ | Score $s_\theta(\tilde\mathbf{x},\sigma)$ | Score $s_\theta(\mathbf{x}_t,t)$ |
| **Duality** | $\hat\epsilon = -\sqrt{1-\bar\alpha}\,s$ | $s = -\hat\epsilon/\sigma$ | $s = -\hat\epsilon/\sqrt{1-e^{-B(t)}}$ |
| **Sampling** | Ancestral: $\mathbf{x}_{i-1} = \mu_\theta(\mathbf{x}_i) + \sqrt{\tilde\beta_i}\,\mathbf{z}$ | Annealed Langevin dynamics | Reverse SDE or PF-ODE |
| **Accelerated sampling** | DDIM ($\eta=0$, sub-sequence) | — | PF-ODE (black-box ODE solver) |
| **Likelihood** | ELBO lower bound | Not directly | Exact via PF-ODE + change-of-variables |
| **Variance schedule** | Discrete $\{\beta_i\}$ | Geometric $\{\sigma_i\}$ | Continuous $\beta(t)$ |
| **Prior** | $\mathcal{N}(\mathbf{0},\mathbf{I})$ | $\mathcal{N}(\mathbf{0},\sigma_L^2\mathbf{I})$ | $p_T\approx\mathcal{N}(\mathbf{0},\mathbf{I})$ (VP) |
| **Key theoretical tool** | Jensen / ELBO | Integration by parts (Hyvärinen) | Fokker-Planck / Itô calculus |

**Core equivalence chain:**

$$
\underbrace{\hat\epsilon_\theta \approx \boldsymbol{\epsilon}}_{\text{DDPM}} \;\Longleftrightarrow\; \underbrace{s_\theta \approx \nabla\log p_t}_{\text{NCSN}} \;\Longleftrightarrow\; \underbrace{s_\theta = -\hat\epsilon_\theta/\sqrt{1-\bar\alpha}}_{\text{Tweedie / score–DDPM}}
$$

All three frameworks learn the **conditional score** of the noisy data distribution at each noise level. The DDPM ε-predictor is a scaled score network. Training and sampling differ only in parameterisation and discretisation granularity, not in the underlying mathematical object.

---

## Appendix: Key Symbols

| Symbol | Meaning |
|--------|---------|
| $L$ | Number of diffusion steps (= 1000 in PA1) |
| $\beta_i$ | Noise variance at step $i$ |
| $\alpha_i = 1-\beta_i$ | Signal retention at step $i$ |
| $\bar\alpha_i = \prod_{j=1}^i\alpha_j$ | Cumulative signal retention |
| $\tilde\beta_i = \frac{1-\bar\alpha_{i-1}}{1-\bar\alpha_i}\beta_i$ | Posterior variance |
| $\tilde\mu_i(\mathbf{x}_i,\mathbf{x}_0)$ | Posterior mean |
| $\hat\epsilon_\theta(\mathbf{x}_i,i)$ | Learned noise predictor (U-Net) |
| $s_\theta(\mathbf{x},t)$ | Score network |
| $B(t)=\int_0^t\beta(s)ds$ | Cumulative noise schedule |
| $p_t(\mathbf{x}_t|\mathbf{x}_0)$ | Perturbation kernel |
| $\mathbf{w}$ | Standard Wiener process |
| VE / VP | Variance-Exploding / Variance-Preserving SDE |
