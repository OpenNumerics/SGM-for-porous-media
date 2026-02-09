# Generative AI for Computational Science by Example

## Score-Based Generative Models for Uncertain PDE Solutions

This repository explores how score-based generative models (SGMs), the mathematical backbone of modern diffusion models, can be used as practical tools in computational science. Instead of learning a single deterministic surrogate for a PDE, we learn a distribution over physically plausible solution fields induced by latent micro-scale uncertainty.

The concrete example studied here comes from battery modeling: we consider coupled PDEs for electrolyte concentration $c(x)$ and electric potential $\phi(x)$ in a porous electrode, where the microscopic porosity field is unknown and treated as a latent random field. This induces a nontrivial distribution over macroscopic solution profiles, even under fixed operating conditions. The SGM learns this distribution and allows us to sample new physically plausible solution fields directly.

The core idea: Learn distributions of PDE solutions, not point predictions.

**What This Repository Demonstrates**
- Training a conditional score-based generative model on PDE solution fields.
- Using diffusion / SDE-based generative models for scientific simulations.
- How uncertainty in the latent porous microstructure induces a distribution over macroscopic states.
- Conditioining generative models on operating parameters (e.g., voltage, current flux, correlation length).
- Evaluating generative models in science via distributional statistics (mean, variance, confidence intervals), not pointwise errors.

This project is not about replacing PDE solvers. They have known almost 100 years of continuous improvement and are simply unbeatable. This project is about learning fast generative models of physically admissible solution distributions that can be embedded into the design loop (e.g. UQ, Bayesian inversion, design-space exploration).

## Why This Matters

In many scientific problems:
- The governing equations are deterministic
- The inputs are uncertain or partially observed
- The output is therefore a distribution, not a single field

SGMs provide a scalable way to learn such distributions directly from data, enabling:
- Fast sampling for uncertainty quantification
- Stochastic surrogates for expensive solvers
- Fast exploration of the design space

This repo is meant as a worked example of how generative AI can be used as a numerical tool in computational science—not as a replacement for physics, but as an extension of it.

## Problem Setting (Short Version)

We study a 1D porous battery electrode model with coupled PDEs for:
- Electrolyte concentration $c(x)$
- Electrolyte potential $\phi(x)$

The equations are deterministic given:
- Macroscopic operating conditions $(U_0, F_{\text{right}})$
- The unknown realization of the microscopic porosity field $\varepsilon(x)$

In practice, $\varepsilon(x)$ is unknown and treated as a latent random field (sampled from a Gaussian process). This induces a distribution over steady-state solution fields $(c(x), \phi(x))$. The SGM is trained to learn this conditional distribution:

$$
p\big(c(x), \phi(x) \mid U_0, \ell, F_{\text{right}}\big)
$$

without ever seeing $\varepsilon(x)$ explicitly. Here, $\ell$ is the correlation length of the Gaussian process and is also a conditioning parameter.


## Method Overview
- Generate training data by sampling porosity fields $\varepsilon(x) \sim \text{GP}(l)$
- Solve the coupled PDEs using a finite-volume method (`fvm.py`)
- Discretize solution fields onto a fixed spatial grid
- Train a conditional score network using diffusion / SDE-based score matching
- Use the learned reverse-time SDE to sample new solution fields
- ompare PDE and SGM results in distribution (mean, variance, percentiles)



## Repository Structure

├── data/					
│   c_data_multieps.pt      		  # Raw electrolyte concentration solution fields
|	phi_data_multieps.pt    		  # Raw electric potential solution fields
|	normalization.pt				  # Normalized versions of both tensors (as NN input)
|	parameters_multieps.pt			  # Values of the associated conditioning parameters
|
├── models/
|	porous_score_model.pth			  # The trained SGM score model
|
├── figures/                		  # Figures used in the writeups
│
├── fvm.py                  		  # Finite-volume solver for the coupled battery PDEs
├── plotFVM.py						  # Plotting routine for the FVM solution
├── solveFVM.py						  # Sample script to initialize and call the FVM solver
├── generatePorousSGMDataset.py       # Script to sample porosity fields and generate PDE solutions
├── PDEParameters.py				  # Values of the global PDE parameters
├── thomas.py						  # Implementation of the Thomas algorithm for tridiagonal systems used in FVM.
│
├── forwardDistribution.py			  # Implements the forward Ornstein-Uhlenbeck process. Used for determining the optimal beta_max.
├── SDEs.py							  # Implement the forward and backward SDEs used for sampling. Also defines the alpha(t) and beta(t) schedule.
├── timesteppers.py					  # Euler-Maruyama and Heun timesteppers used to generate plausable solutions
│
├── generatePorousSGMDataset.py		  # Generate the training data in data/. Parallelized but takes an hour to run.
├── PorousSGMDataset.py				  # Subclass of PyTorch's `Dataset` used for the training and validation datasets.
├── ConvFiLMScore.py				  # FiLM-Convolutional Network definition.
├── trainPorousSGMConv.py			  # Main SGM training script with the Adam optimizer. Takes 10 hours to run on my Macbook Neural Engine.
├── testPorousSGMConv.py			  # Simple testing script that generates one plausable SGM solution and compares it to another PDE solution.
├── compareFVM_SGM_means.py			  # Main testing script that runs 1000 SGM and PDE solutions (see `timesteppers.py` and `fvm.py`) and makes the main 
├── plotSGMCI.py					  # Same as `compareFVM_SGM_means.py` but without the solution-generation step.
│
└── README.md

## How to run
1. Generate the data
> ```
> python generatePorousSGMDataset.py
> ```
2. Train the SGM
> ```
> python trainPorousSGMConv.py
> ```
3. Sample new solution fields and compare statistics
> ```
> python compareFVM_SGM_means.py
> ```

## Results

The SGM reproduces:
- The mean solution profiles of the PDE ensemble
- The variance and confidence intervals
- The correct behavior near Dirichlet and Neumann boundaries (very hard!)

Crucially, the model captures uncertainty induced by latent porosity fields—even though porosity is never explicitly given to the network. It *learns* the underlying distribution.


## Citation / Attribution

If you use this code or ideas in academic work, a citation or link back to this repository is appreciated. Also see LICENSE.