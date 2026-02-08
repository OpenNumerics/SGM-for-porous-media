# Score-based Generative Modeling for Electric Potential Fields in Porous Battery Media

This repository contains a finished research project on using score-based generative models (SGMs) to model the distribution of electric potential fields in porous battery media. The goal is not to predict a single solution of a PDE, but to learn a generative model over physically plausible solution profiles induced by uncertainty in the porous microstructure.

**Main result**:
The model learns the variance and spatial uncertainty structure of the electric potential fields very accurately. The mean of the generated fields, however, exhibits a systematic bias. This bias is not a failure of the generative model, but a consequence of non-identifiability in the porosity field: different porous microstructures can produce nearly indistinguishable potential profiles.


## Background & Physical Model

In porous electrochemical media (e.g. battery electrodes or electrolytes), the electric potential \phi(x) satisfies a quasi-static conservation law

$$
\nabla \cdot \big( \varepsilon(x)\, \nabla \phi(x) \big) = f(x),
$$

with suitable boundary conditions. Here:
- $\phi(x)$ is the electric potential,
- $\varepsilon(x)$ is an effective transport coefficient induced by porosity and microstructure (e.g. accounting for tortuosity and reduced conductive cross-section),
- $f(x)$ represents volumetric source terms (e.g. reaction currents).

The key difficulty is that $\varepsilon(x)$ is spatially heterogeneous and uncertain. In realistic settings it is not known exactly, but only through a statistical model reflecting microstructural variability. This uncertainty induces a distribution over solution fields $\phi(x)$, rather than a single deterministic outcome.


## What Is Being Learned?

Rather than learning a surrogate that maps inputs to a single solution, this project learns a *conditional* distribution over potential fields:

$$
p(\phi \mid f, \text{BCs}),
$$

induced by randomness in the porosity field $\varepsilon(x)$. The model is trained on ensembles of PDE solutions generated from randomly sampled porosity fields and learns to generate new, physically plausible solution profiles. Importantly, the generative model does not observe $\varepsilon(x)$ directly. It only sees the resulting potential fields. As a result, it implicitly marginalizes over all porosity fields that could have produced a given $\phi(x)$.


## Method: Score-Based Generative Modeling

We use score-based diffusion models to learn the data distribution of solution fields. In short:
	1.	Generate training data by solving the porous-media PDE for many random realizations of \varepsilon(x).
	2.	Train a neural network to approximate the score (gradient of the log-density) of noisy solution fields.
	3.	Sample new potential fields by integrating the reverse-time SDE (or probability flow ODE).

This yields a generative model capable of producing diverse, physically plausible electric potential profiles that match the statistics of the PDE ensemble. A great technical introduction to SGMs can be found on [Jakiw's blog](https://jakiw.com/sgm_intro). I highly recommend checking it out!

## Neural Network Architecture & FiLM Conditioning

The score network is implemented as a convolutional neural network operating directly on discretized potential field $\phi(x)$ and electolyte concentration $c(x)$. At each time $t$, the network takes as input a noisy fields $\phi_t(x)$ and $c_t(x)$ and predicts the score $(\nabla_{\phi} \log p_t(\phi, c), \nabla_{c} \log p_t(\phi, c))$.

A key architectural contribution of this project is the use of Feature-wise Linear Modulation (FiLM) to condition the score network on physical context. To the best of our knowledge, this is the first application of FiLM-style conditioning in the context of score-based generative models for PDE solution fields.

### What Is Conditioned?

The generative model is conditioned on global and low-dimensional physical inputs, such as:
- boundary conditions,
- source term parameters,
- global scalars describing the PDE setup (e.g. load level, reaction strength, normalization factors).

Crucially, the porosity field $\varepsilon(x)$ is not provided explicitly to the network, this would make it too easy. The conditioning therefore reflects partial observability of the physical system, consistent with realistic modeling scenarios.

### Why FiLM?

Naively concatenating physical parameters to the input field (or broadcasting them as extra channels) often leads to weak conditioning, especially in deep convolutional architectures. FiLM provides a more expressive and stable mechanism by modulating intermediate feature maps:

$$
h_\ell \;\mapsto\; \gamma_\ell(c)\, h_\ell + \beta_\ell(c),
$$

where:
- $h_\ell$ are intermediate feature maps at layer $\ell$,
- $c$ denotes the conditioning variables (boundary conditions, source parameters, etc.),
- $\gamma_\ell(c)$ and $\beta_\ell(c)$ are learned functions produced by a small conditioning network (MLP).

This allows the conditioning variables to dynamically re-weight and shift internal representations, enabling the same score network to represent different physical regimes without retraining separate models.

### Architecture Overview

At a high level, the model consists of:
- A convolutional backbone processing the noisy field $(\phi_t(x), c_t(x))$,
- Time embedding of the diffusion time $t$, injected into the network,
- A FiLM conditioning network that maps physical parameters $c$ to per-layer modulation coefficients $\{\gamma_\ell, \beta_\ell\}$,
- FiLM layers applied at multiple depths of the network to modulate intermediate feature maps.

This results in a conditionally modulated diffusion model, where both the noise scale and the physical context shape the learned score field.


### Empirical Impact of FiLM Conditioning

Empirically, FiLM conditioning was critical for:
- achieving stable training across a wide range of physical regimes,
- preventing mode collapse in conditioned generation,
- enabling a single model to generalize across different boundary condition configurations,
- preserving physically meaningful spatial structures in the generated samples.

Without FiLM, the model either failed to properly condition on the physical inputs or required separate models for different regimes, significantly reducing sample efficiency.

### Relation to Physics-Informed Learning

While this project is not “physics-informed” in the sense of enforcing PDE residuals in the loss, the FiLM conditioning provides a structured way of injecting physical context into the generative model. This sits in between:
- pure data-driven generative modeling of fields, and
- fully physics-constrained methods such as PINNs or PINOs.

In practice, this combination allows the SGM to learn the geometry of the solution manifold while remaining sensitive to changes in physical setup. The approach also shares some high-level similarities to transformers such as the necessity of time embeddings.

## Results & Interpretation

What works well
- The model accurately captures the variance of the solution fields.
- Spatial correlation structure and uncertainty patterns match the ground-truth PDE ensemble.
- Samples are smooth, physically consistent, and lie on the correct solution manifold.

What is fundamentally limited
- The mean field is biased compared to the true ensemble mean.
- I think this bias is caused by non-identifiability of the porosity field $\varepsilon(x)$: multiple different porosity configurations can lead to nearly identical potential profiles.
- As a result, the conditional mean of $\phi(x)$ is not uniquely determined by the observations available to the model.

## Takeaway

The model correctly learns how uncertain the solution is and where that uncertainty lives, but the mean solution is not uniquely identifiable without conditioning on additional information about the porous microstructure. I believe this is a structural limitation of the inverse problem, not a modeling failure of diffusion models.


## Repository Structure (indicative)
