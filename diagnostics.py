"""
Convergence diagnostics for MCMC chains and weighted samples.
"""

import numpy as np


def integrated_autocorrelation_time(walker_by_step_array, window_constant=5,
                                    return_frozen_count=False):
    """
    Sokal's windowed estimator of the integrated autocorrelation time, the same
    algorithm emcee uses. Input is shape (n_walkers, n_steps).

    Roughly, one independent sample appears every tau steps, so a chain of
    n_steps contributes about n_steps / tau independent draws.

    Two details that are easy to get wrong and are handled explicitly here.

    Per-walker normalisation: each walker is divided by its OWN zero-lag value
    before being averaged in. Dividing the summed autocorrelation by the summed
    zero-lag value instead is a tempting shortcut that silently weights noisier
    walkers more heavily and biases the estimate.

    Frozen walkers: a walker whose value never changes has zero variance, so
    its normalised autocorrelation is 0/0. That is not a numerical nuisance --
    on the model order k it means the walker never accepted a single birth or
    death move, and its autocorrelation time is genuinely infinite. Averaging
    it in as NaN destroys the estimate for every other walker; treating it as
    zero would pretend it mixed perfectly. It is excluded and counted instead,
    which keeps the estimate meaningful and makes the frozen population visible.
    """
    walker_by_step_array = np.asarray(walker_by_step_array, dtype=float)
    n_walkers, n_steps = walker_by_step_array.shape
    centred = walker_by_step_array - walker_by_step_array.mean(axis=1, keepdims=True)

    padded_length = int(2 ** np.ceil(np.log2(n_steps)))
    normalised_autocorrelation_sum = np.zeros(n_steps)
    moving_walker_count = 0
    frozen_walker_count = 0

    for walker_index in range(n_walkers):
        if np.all(centred[walker_index] == 0.0):
            frozen_walker_count += 1
            continue
        fourier_transform = np.fft.fft(centred[walker_index], n=2 * padded_length)
        power_spectrum = fourier_transform * np.conjugate(fourier_transform)
        autocorrelation = np.fft.ifft(power_spectrum).real[:n_steps]
        if autocorrelation[0] <= 0.0:
            frozen_walker_count += 1
            continue
        normalised_autocorrelation_sum += autocorrelation / autocorrelation[0]
        moving_walker_count += 1

    if moving_walker_count == 0:
        result = np.inf
        return (result, frozen_walker_count) if return_frozen_count else result

    autocorrelation_function = normalised_autocorrelation_sum / moving_walker_count

    cumulative_tau = 2.0 * np.cumsum(autocorrelation_function) - 1.0
    window_index = np.arange(len(cumulative_tau))
    # The smallest window satisfying window >= window_constant * tau(window).
    # Too short a window biases tau low; too long is dominated by tail noise.
    passes_criterion = window_index < window_constant * cumulative_tau
    selected_window = (len(cumulative_tau) - 1 if not np.any(passes_criterion)
                       else int(np.argmin(passes_criterion)))
    tau = max(float(cumulative_tau[selected_window]), 1.0)
    return (tau, frozen_walker_count) if return_frozen_count else tau


def gelman_rubin_rhat(walker_by_step_array):
    """
    Gelman-Rubin R-hat, treating each walker as one chain. Input is shape
    (n_walkers, n_steps).

    Compares the spread of per-walker means against the spread within each
    walker. Near 1.0 means the walkers agree about where the posterior mass is.
    Well above 1 means some are still stuck somewhere the others are not.

    R-hat measures AGREEMENT, not correctness: walkers that are all stuck in
    the same wrong place will report a healthy R-hat. It has to be read
    alongside tau, not instead of it.
    """
    walker_by_step_array = np.asarray(walker_by_step_array, dtype=float)
    n_walkers, n_steps = walker_by_step_array.shape
    if n_walkers < 2 or n_steps < 2:
        return np.nan

    chain_means = walker_by_step_array.mean(axis=1)
    grand_mean = chain_means.mean()
    between_chain_variance = (n_steps / (n_walkers - 1)
                              * np.sum((chain_means - grand_mean) ** 2))
    within_chain_variance = walker_by_step_array.var(axis=1, ddof=1).mean()
    if within_chain_variance <= 0:
        return np.inf
    pooled_variance_estimate = ((n_steps - 1) / n_steps * within_chain_variance
                                + between_chain_variance / n_steps)
    return float(np.sqrt(pooled_variance_estimate / within_chain_variance))


def effective_sample_size_from_weights(weights):
    """
    Kish effective sample size for WEIGHTED samples, such as Nested Sampling
    output:

        ESS = (sum_i w_i)^2 / sum_i w_i^2

    Equal weights give ESS = N. One dominant weight gives ESS = 1. This is the
    right quantity to quote for Nested Sampling, where the stored point count
    says nothing useful because the weights span many orders of magnitude.
    """
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        return 0.0
    return float(total ** 2 / np.sum(weights ** 2))


def effective_sample_size_from_chain(walker_by_step_array):
    """
    Effective sample size for a CORRELATED chain: total draws divided by the
    autocorrelation time. Using the raw sample count instead overstates the
    information in the chain by a factor of tau, which is exactly the mistake
    that makes an MCMC contour look better determined than it is.
    """
    walker_by_step_array = np.asarray(walker_by_step_array, dtype=float)
    n_walkers, n_steps = walker_by_step_array.shape
    tau = integrated_autocorrelation_time(walker_by_step_array)
    if not np.isfinite(tau):
        return 0.0
    return float(n_walkers * n_steps / tau)
