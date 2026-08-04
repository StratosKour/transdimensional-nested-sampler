# Trans-dimensional Nested Sampling

A custom hybrid Bayesian sampler for 1D signal detection. It combines **Nested
Sampling** (for evidence estimation and model selection) with **Reversible Jump
MCMC** (for trans-dimensional parameter exploration), using the `Eryn` library as
the inner sampling engine.

## Overview

The goal is to detect an *unknown* number of Gaussian pulses hidden in stationary,
white Gaussian noise. The algorithm jointly infers the number of pulses (model
order $k$) and their parameters (amplitude, mean position), and reports the
Bayesian evidence $\mathcal{Z}$ for model comparison, including the $k=0$
"no signal" hypothesis.

## Architecture

The sampler operates on a two-tier design:

* **Outer loop (Nested Sampling).** Maintains a population of live points. At each
  iteration it discards the point with the lowest likelihood, archives it as a dead
  point, and raises the likelihood threshold $\mathcal{L}_0$ that any replacement
  must exceed. This shrinking-threshold sequence is what the evidence integral is
  built from.
* **Inner loop (RJ-MCMC via Eryn).** To generate a replacement, a surviving live
  point is chosen at random as a seed and handed to Eryn's `EnsembleSampler`
  (`nwalkers=1`), which runs a short constrained random walk. Eryn is given a
  **hard-constrained** likelihood (0 inside the current threshold, $-\infty$
  outside), so the walk is a pure prior random walk subject to a pass/fail gate
  rather than a climb toward higher likelihood — exactly the "sample from the prior
  restricted to $\mathcal{L} > \mathcal{L}_0$" operation Nested Sampling requires.
  Reversible-jump moves let this walk cross between different numbers of active
  pulses.

## Key features

* **Trans-dimensional inference:** the number of active pulses (0 to `nleaves_max`)
  is inferred jointly with their parameters, not fixed in advance.
* **Reduced parameter space:** pulse width is fixed, reducing each pulse to 2 free
  parameters (amplitude, mean position).
* **k = 0 handled analytically:** a known Eryn limitation prevents RJ-MCMC from
  exploring zero active leaves, so the no-signal hypothesis is evaluated in closed
  form and recombined with the RJ-MCMC evidence for the full posterior over $k$.
* **Correctness safeguards:** explicit assertions enforce the core Nested Sampling
  invariant (each replacement strictly exceeds the discarded threshold), NaN-safe
  padding of inactive parameter slots, and counters that surface degraded sampler
  behaviour (clone fallback, stuck walkers) instead of hiding it.

## Results

Eight validation runs, covering model orders 0–4, high and low SNR, and separated
versus overlapping pulses. All notebooks include full outputs and diagnostic plots.

| Scenario | k_true | SNR | MAP | P(k_true) | log Bayes factor | Notebook |
|---|---|---|---|---|---|---|
| Null test | 0 | 0 | 0  | 0.99 | −4.80 | [Transnested_zero_pulses](./Transnested_zero_pulses.ipynb) |
| One pulse | 1 | 26.6 | 1  | 0.75 | +308 | [Transnested_one_pulse](./Transnested_one_pulse.ipynb) |
| Two pulses, separated | 2 | 38.0 | 2  | 0.81 | +674 | [Transnested_two_pulses](./Transnested_two_pulses.ipynb) |
| Two pulses, overlapping | 2 | 12.5 | 2  | 0.44 | +55 | [Transnested_two_pulses_lowSNR](./Transnested_two_pulses_lowSNR.ipynb) |
| Two pulses, overlapping (seed 1) | 2 | 12.5 | 2  | 0.50 | +75 | [..._difseed](./Transnested_two_pulses_lowSNR_difseed.ipynb) |
| Four pulses, unequal amplitudes | 4 | 75.2 | 4  | 0.82 | +2825 | [..._more_iterations](./Transnested_four_pulses_more_iterations.ipynb) |

**Notes on the remaining notebooks:**
- `Transnested_four_pulses.ipynb` uses the default 6000-iteration ceiling and does
  **not** converge cleanly (low effective sample size). Kept only for comparison
  against the converged run above — the recovered evidence agrees to within 0.03
  nats, but the parameter posteriors from the truncated run are unreliable.
- `Transnested_four_pulses_more_iterations_smaller_steps.ipynb` is a proposal
  step-size sensitivity check (`1e-4` instead of `1e-3`). It intentionally
  produces the **wrong** MAP ($k=5$) despite better-looking diagnostics
  (effective sample size, acceptance), and is kept as a documented example of why
  evidence estimates must be validated against a known ground truth rather than
  trusted from diagnostics alone.

**Summary:** no false detections in pure noise, correct MAP recovered at every
converged model order tested, and honest posterior uncertainty in the low-SNR
overlapping-source regime rather than false confidence.

## Dependencies

* Python 3.x
* `numpy`, `matplotlib`
* `eryn`
* `anesthetic` (evidence, uncertainty, and Nested Sampling weight reconstruction)
* `corner` (parameter posterior corner plots)
* `tqdm` (progress bar for the sampling loop)

## Usage

The main implementation is a step-by-step notebook
covering: the generative toy model and synthetic data injection, the
`ProbDistContainer` prior setup, the hard-constrained log-likelihood, branch
structure and live-point initialisation, the core Nested Sampling iteration with
full dead-point bookkeeping, the sampling loop with its convergence criterion, and
evidence/posterior post-processing via `anesthetic`.

The scenario notebooks listed under **Results** are copies of this notebook run
with different injected signals, noise levels, and sampler settings (see the
`[TUNABLE]` comments in Section 2, 6, 7, and 9 of the main notebook for what each
run changed).
