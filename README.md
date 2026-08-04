# transdimensional-nested-sampler
A custom Trans-dimensional Nested Sampler combining an outer Nested Sampling loop with Eryn's Reversible Jump MCMC. Designed to detect an unknown number of Gaussian pulses in 1D noisy signals.

# Trans-dimensional Nested Sampling

This repository implements a custom hybrid Bayesian sampling algorithm for 1D signal detection. It successfully combines **Nested Sampling** (for evidence estimation) with **Reversible Jump MCMC** (for trans-dimensional parameter exploration), using the `Eryn` library as the inner sampling engine.

## Overview
The goal of this toy model is to detect an *unknown* number of Gaussian pulses hidden in stationary, white Gaussian noise. The algorithm dynamically searches for both the correct number of pulses (model selection) and their specific parameters (parameter estimation).

## Architecture
The sampler operates on a two-tier architecture:
*   **Outer Loop (Nested Sampling):** Manages a population of active "live points", iteratively replacing the point with the lowest likelihood to shrink the prior volume and calculate the Bayesian Evidence ($\mathcal{Z}$).
*   **Inner Loop (RJMCMC via Eryn):** When a live point is discarded, Eryn's `EnsembleSampler` is called to evolve a copy of another live point. It proposes cross-model jumps (adding/removing pulses) and strictly obeys a dynamic "hard constraint" (the new point's likelihood must exceed the discarded point's likelihood).

## Key Features
*   **Trans-dimensional Inference:** Dynamically jumps between models with different numbers of parameters (0 to N pulses).
*   **Reduced Parameter Space:** Pulse width ($\sigma$) is fixed, reducing the search space to 2 dimensions per pulse (Amplitude and Mean).
*   **Safety-Checked Execution:** Designed for Jupyter Notebooks with integrated checks to prevent heavy memory re-allocation during testing.

## Dependencies
*   Python 3.x
*   `numpy`
*   `matplotlib`
*   `eryn`

## Usage
The main implementation is provided as a step-by-step Jupyter Notebook. It includes the generative toy model, the `ProbDistContainer` prior setup, the constrained log-likelihood function, and the main NS + RJMCMC control loop.
