"""
The pulse model, the likelihood, the priors, and synthetic data generation.

This is the single place these are defined. Every other script -- the Nested
Sampler, the plain RJ-MCMC baseline, the extended convergence runs, the fixed-k
sanity check, the analysis scripts -- imports from here. None of them may
contain a second copy of any function in this file, even a copy that looks
identical at the time it is written. A second copy is a second place for a
typo to hide, and a second place someone has to remember to update together
with this one.

Everything here is copied from transnested_experiment.py, Sections 2 to 4,
with one deliberate change: true_gaussian_log_likelihood takes
pulse_width_fixed as an explicit argument instead of reading it from an
enclosing closure. The formula is unchanged. The reason for the change is the
reason this file exists -- a function that only works correctly inside one
particular closure cannot be safely reused anywhere else, which is exactly
what went wrong when the extended-run scripts each wrote their own copy of
this function by hand.
"""

import numpy as np

from eryn.prior import uniform_dist, ProbDistContainer


# 1 - THE PULSE MODEL

def single_gaussian_pulse(time_axis, pulse_amplitude, pulse_mean_position,
                          pulse_width_fixed):
    """
    Evaluate ONE Gaussian pulse across the full time axis.

        f(t) = pulse_amplitude * exp( -(t - pulse_mean_position)^2 / (2 c^2) )

    pulse_width_fixed (c) is the same for every pulse in a given problem. This
    is what reduces the per-pulse dimensionality to 2 (amplitude, position)
    instead of 3.
    """
    return pulse_amplitude * np.exp(
        -((time_axis - pulse_mean_position) ** 2) / (2 * pulse_width_fixed ** 2)
    )


def combine_all_pulses(time_axis, list_of_pulse_parameter_pairs, pulse_width_fixed):
    """
    Sum an ARBITRARY number of Gaussian pulses into one noiseless model signal.

    The interface is unchanged whether it receives zero pulses or twelve,
    which is what makes the problem trans-dimensional in practice. An EMPTY
    input returns the zero signal, which is how the k = 0 hypothesis
    (no signal present) is evaluated.
    """
    combined_signal = np.zeros_like(time_axis)

    for single_pulse_parameter_pair in list_of_pulse_parameter_pairs:
        pulse_amplitude, pulse_mean_position = single_pulse_parameter_pair
        combined_signal += single_gaussian_pulse(
            time_axis, pulse_amplitude, pulse_mean_position, pulse_width_fixed)

    return combined_signal


# 2 - THE LIKELIHOOD

def true_gaussian_log_likelihood(pulse_parameters_array, time_axis, observed_data,
                                 noise_standard_deviation, pulse_width_fixed):
    """
    The physical log-likelihood. Every script that needs a likelihood value --
    to build a Nested Sampling archive, to drive a plain MCMC chain, to check
    a fixed-k posterior, to verify agreement between two runs -- calls this
    same function.

        logL = -0.5 * sum_j [ (d_j - s(t_j)) / sigma_n ]^2

    The additive constant -N/2 * log(2 pi sigma_n^2) is omitted. It is
    identical for every model compared in this project, so it cancels in every
    ratio and every weight, but it does offset the absolute logZ values that
    get reported.
    """
    model_signal = combine_all_pulses(time_axis, pulse_parameters_array, pulse_width_fixed)
    residual = (observed_data - model_signal) / noise_standard_deviation
    return -0.5 * np.sum(residual ** 2)


# 3 - PRIORS

def build_priors(amplitude_min, amplitude_max, mean_position_min, mean_position_max):
    """
    The prior distribution object Eryn samples from, for one branch ("gauss").

    Key convention, matching the coordinate array layout used everywhere in
    this project: column 0 is pulse_amplitude, column 1 is pulse_mean_position.
    """
    gauss_branch_prior_dict = {
        0: uniform_dist(amplitude_min, amplitude_max),
        1: uniform_dist(mean_position_min, mean_position_max),
    }
    return {"gauss": ProbDistContainer(gauss_branch_prior_dict)}


def assert_injection_within_prior(true_injected_pulse_parameters,
                                  amplitude_min, amplitude_max,
                                  mean_position_min, mean_position_max):
    """
    The correct answer must lie inside the prior support, or it is
    unreachable by construction. Raises AssertionError naming the offending
    pulse rather than failing later with a confusing symptom deep inside the
    sampler.
    """
    for pulse_index, (injected_amplitude, injected_mean_position) in enumerate(
            true_injected_pulse_parameters):

        assert amplitude_min <= injected_amplitude <= amplitude_max, (
            f"Injected pulse {pulse_index} amplitude {injected_amplitude} lies "
            f"outside the prior range [{amplitude_min}, {amplitude_max}].")

        assert mean_position_min <= injected_mean_position <= mean_position_max, (
            f"Injected pulse {pulse_index} mean position {injected_mean_position} "
            f"lies outside the prior range [{mean_position_min}, {mean_position_max}].")


# 4 - SYNTHETIC DATA GENERATION

def build_time_axis(number_of_time_samples, time_axis_lower_bound, time_axis_upper_bound):
    return np.linspace(time_axis_lower_bound, time_axis_upper_bound, number_of_time_samples)


def generate_synthetic_dataset(time_axis, true_injected_pulse_parameters,
                               noise_standard_deviation, pulse_width_fixed,
                               random_seed):
    """
    Build one noisy dataset from one injection.

    Calling this twice with the same random_seed reproduces the same dataset
    exactly, since it seeds NumPy's global random state before drawing the
    noise. This is the ONLY place in the project that should draw the noise
    realisation -- any script that needs "the same dataset as archive X" should
    reconstruct it by calling this function with that archive's stored seed
    and parameters, not by re-deriving the noise some other way.
    """
    np.random.seed(random_seed)

    noiseless_injected_signal = combine_all_pulses(
        time_axis, true_injected_pulse_parameters, pulse_width_fixed)

    measurement_noise = noise_standard_deviation * np.random.randn(len(time_axis))
    observed_data = noiseless_injected_signal + measurement_noise

    # Optimal (matched-filter) SNR: the highest any analysis could extract,
    # and the single number quantifying how hard this dataset is.
    optimal_matched_filter_snr = (
        np.sqrt(np.sum(noiseless_injected_signal ** 2)) / noise_standard_deviation)

    return {
        "noiseless_injected_signal": noiseless_injected_signal,
        "observed_data": observed_data,
        "snr": optimal_matched_filter_snr,
    }
