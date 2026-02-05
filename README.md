# SGM-for-porous-media

## Score-based Generative Modeling for Electric Potential Fields in Porous Battery Media

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
