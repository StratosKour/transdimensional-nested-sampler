"""
One complete experiment, end to end.

This module owns no algorithm of its own. It builds the problem once, hands
the SAME model and likelihood objects to both samplers, collects what they
return, and writes the archive. The pieces it calls:

    pulse_model.py       the pulse model, likelihood, priors, synthetic data
    nested_sampler.py    the trans-dimensional Nested Sampling algorithm
    baseline_rjmcmc.py   the plain RJ-MCMC comparison baseline

That both samplers receive the same function objects, rather than separately
written copies of the same formula, is the point of splitting things this way.
It makes "are they solving the same problem?" a question that cannot be
answered wrongly.

The public entry point keeps the name and signature of the original
run_single_experiment, so run_sweep.py and every analysis script continue to
work unchanged.
"""

import numpy as np

from pulse_model import (
    combine_all_pulses as _pulse_model_combine_all_pulses,
    true_gaussian_log_likelihood as _pulse_model_true_gaussian_log_likelihood,
    build_priors,
    assert_injection_within_prior,
)
from nested_sampler import run_nested_sampler
from baseline_rjmcmc import run_plain_rjmcmc_baseline


def run_single_experiment(
    # Section 2 - toy model and synthetic data
    noise_standard_deviation,
    true_injected_pulse_parameters=None,
    observed_data=None,
    time_axis=None,
    random_seed=0,
    pulse_width_fixed=0.1,
    number_of_time_samples=500,
    time_axis_lower_bound=-1.0,
    time_axis_upper_bound=1.0,

    # Section 3 - prior bounds
    amplitude_min=0.0,
    amplitude_max=5.0,

    # Section 5 - branch structure
    nleaves_max_value=8,
    nleaves_min_value=1,

    # Section 6 - live point population
    number_of_live_points=None,

    # Section 7 - proposal
    gaussian_move_step_size_scale=1e-3,

    # Section 9 - loop configuration
    mcmc_steps_per_replacement=None,
    maximum_allowed_iterations=10000,
    evidence_convergence_tolerance=1e-3,

    # Section 10 - posterior combination
    prior_odds_signal_vs_no_signal=1.0,

    # Section 11 - plain RJ-MCMC baseline
    run_plain_rjmcmc_comparison=False,
    plain_rjmcmc_nwalkers=50,
    plain_rjmcmc_ntemps=10,
    plain_rjmcmc_nsteps=20000,
    plain_rjmcmc_burn_in_fraction=0.2,

    # runner options
    save_raw_to=None,
    verbose=True,
):
    """
    Run one complete experiment end to end and return a results dictionary.

    Arguments left as None are derived from the others using the notebook's own
    rules, so the defaults here reproduce the notebook exactly.

    save_raw_to : str or None
        Path to an .npz archive holding the full (logL, logL_birth, coords, mask)
        record, the posterior weights, the data, and the diagnostic histories.
        Given this file, every figure can be redrawn without re-running the
        sampler.
    """

    # 1 - REPRODUCIBILITY

    # Fixing the seed makes both the synthetic data and every random draw in the
    # samplers reproducible. Two runs differing only in this argument have
    # different noise realisations, which makes them different datasets rather
    # than repeats.
    np.random.seed(random_seed)

    # 2 - MODEL, DATA

    # Thin wrappers that pin the fixed pulse width, so the samplers can call
    # these with the short signature they expect. Both samplers are given these
    # exact objects, which is what guarantees they share a likelihood.
    def combine_all_pulses(time_axis, list_of_pulse_parameter_pairs):
        return _pulse_model_combine_all_pulses(
            time_axis, list_of_pulse_parameter_pairs, pulse_width_fixed)

    def true_gaussian_log_likelihood(pulse_parameters_array, time_axis,
                                     observed_data, noise_standard_deviation):
        return _pulse_model_true_gaussian_log_likelihood(
            pulse_parameters_array, time_axis, observed_data,
            noise_standard_deviation, pulse_width_fixed)

    # Two modes.
    #
    # SIMULATION (the sweep's mode): an injection is supplied and the data are
    # generated from it here. The truth is known, so recovery can be scored.
    #
    # REAL DATA: observed_data is supplied by the caller and there is no known
    # truth. Everything that exists only to score recovery against a known
    # answer is then None, and the sampler runs blind, exactly as it would on a
    # real measurement. The sampling itself is identical in both modes -- the
    # truth is never used to guide it, only to grade it afterwards.
    analysing_real_data = observed_data is not None

    if analysing_real_data:
        observed_data = np.asarray(observed_data, dtype=float)
        if time_axis is None:
            time_axis = np.linspace(time_axis_lower_bound, time_axis_upper_bound,
                                    len(observed_data))
        else:
            time_axis = np.asarray(time_axis, dtype=float)
            assert len(time_axis) == len(observed_data), (
                f"time_axis has {len(time_axis)} samples but observed_data has "
                f"{len(observed_data)}.")
        true_number_of_pulses = None
        noiseless_injected_signal = None
        optimal_matched_filter_snr = np.nan
    else:
        assert true_injected_pulse_parameters is not None, (
            "Supply either true_injected_pulse_parameters (to simulate a dataset) "
            "or observed_data (to analyse one you already have).")
        time_axis = np.linspace(time_axis_lower_bound, time_axis_upper_bound,
                                number_of_time_samples)

        # [DERIVED] Read straight off the injection list, so it can never fall
        # out of sync with the pulses actually injected.
        true_number_of_pulses = len(true_injected_pulse_parameters)

        noiseless_injected_signal = combine_all_pulses(
            time_axis, true_injected_pulse_parameters)

        measurement_noise = noise_standard_deviation * np.random.randn(number_of_time_samples)
        observed_data = noiseless_injected_signal + measurement_noise

        # Optimal (matched-filter) SNR: the highest any analysis could extract,
        # and the single number quantifying how hard this dataset is.
        optimal_matched_filter_snr = (
            np.sqrt(np.sum(noiseless_injected_signal ** 2)) / noise_standard_deviation
        )

    # 3 - PRIORS

    # [DERIVED] Pulses may only be centred within the observation window.
    mean_position_min = time_axis.min()
    mean_position_max = time_axis.max()

    branch_names = ["gauss"]

    priors = build_priors(amplitude_min, amplitude_max,
                          mean_position_min, mean_position_max)

    if not analysing_real_data:
        assert_injection_within_prior(
            true_injected_pulse_parameters,
            amplitude_min, amplitude_max, mean_position_min, mean_position_max)

    # 4 - THE k = 0 HYPOTHESIS, EVALUATED ANALYTICALLY

    # With zero free parameters there is nothing to integrate over, so the
    # evidence equals the likelihood at the only point the model permits: the
    # zero signal. Computed through the SAME likelihood every other point uses.
    log_evidence_no_signal_model = true_gaussian_log_likelihood(
        np.empty((0, 2)), time_axis, observed_data, noise_standard_deviation)

    # 5 - NESTED SAMPLING

    nested_result = run_nested_sampler(
        time_axis=time_axis,
        observed_data=observed_data,
        noise_standard_deviation=noise_standard_deviation,
        priors=priors,
        branch_names=branch_names,
        amplitude_min=amplitude_min,
        amplitude_max=amplitude_max,
        mean_position_min=mean_position_min,
        mean_position_max=mean_position_max,
        true_gaussian_log_likelihood=true_gaussian_log_likelihood,
        combine_all_pulses=combine_all_pulses,
        true_number_of_pulses=true_number_of_pulses,
        true_injected_pulse_parameters=true_injected_pulse_parameters,
        log_evidence_no_signal_model=log_evidence_no_signal_model,
        nleaves_max_value=nleaves_max_value,
        nleaves_min_value=nleaves_min_value,
        number_of_live_points=number_of_live_points,
        gaussian_move_step_size_scale=gaussian_move_step_size_scale,
        mcmc_steps_per_replacement=mcmc_steps_per_replacement,
        evidence_convergence_tolerance=evidence_convergence_tolerance,
        maximum_allowed_iterations=maximum_allowed_iterations,
        prior_odds_signal_vs_no_signal=prior_odds_signal_vs_no_signal,
        verbose=verbose,
    )

    # 6 - PLAIN RJ-MCMC BASELINE
    #
    # Given the same priors and the same likelihood object the Nested Sampler
    # just used. The branch structure and proposal covariance come back from
    # the Nested Sampler rather than being rebuilt, so the two cannot drift.

    baseline_result = run_plain_rjmcmc_baseline(
        time_axis=time_axis,
        observed_data=observed_data,
        noise_standard_deviation=noise_standard_deviation,
        priors=priors,
        branch_names=branch_names,
        true_gaussian_log_likelihood=true_gaussian_log_likelihood,
        ndims_per_branch=nested_result["ndims_per_branch"],
        nleaves_max=nested_result["nleaves_max"],
        nleaves_min=nested_result["nleaves_min"],
        pulse_ndim=nested_result["pulse_ndim"],
        gaussian_move_covariance=nested_result["gaussian_move_covariance"],
        run_plain_rjmcmc_comparison=run_plain_rjmcmc_comparison,
        plain_rjmcmc_ntemps=plain_rjmcmc_ntemps,
        plain_rjmcmc_nwalkers=plain_rjmcmc_nwalkers,
        plain_rjmcmc_nsteps=plain_rjmcmc_nsteps,
        plain_rjmcmc_burn_in_fraction=plain_rjmcmc_burn_in_fraction,
    )

    # 7 - ARCHIVE

    posterior_probability_per_k = nested_result["posterior_probability_per_k"]
    total_replacements = nested_result["total_replacements"]

    if save_raw_to is not None:
        # Everything needed to redraw any figure without re-running the sampler.
        np.savez_compressed(
            save_raw_to,
            logL=nested_result["logL"],
            logL_birth=nested_result["logL_birth"],
            coords=nested_result["coords"],
            mask=nested_result["mask"],
            posterior_weights=nested_result["posterior_weights"],
            flattened_active_parameters=nested_result["flattened_active_parameters"],
            flattened_parameter_weights=nested_result["flattened_parameter_weights"],
            posterior_probability_per_k=posterior_probability_per_k,
            log_evidence_history=nested_result["log_evidence_history"],
            best_live_logL_history=nested_result["best_live_logL_history"],
            active_pulse_count_history=nested_result["active_pulse_count_history"],
            time_axis=time_axis,
            observed_data=observed_data,
            noiseless_injected_signal=(noiseless_injected_signal
                                       if noiseless_injected_signal is not None
                                       else np.array([])),
            true_injected=(np.asarray(true_injected_pulse_parameters,
                                      dtype=float).reshape(-1, 2)
                           if true_injected_pulse_parameters is not None
                           else np.empty((0, 2))),
            plain_rjmcmc_posterior_per_k=(
                baseline_result["plain_rjmcmc_posterior_per_k"]
                if baseline_result["plain_rjmcmc_posterior_per_k"] is not None
                else np.array([])),
            plain_rjmcmc_flat_params=(
                baseline_result["plain_rjmcmc_flat_params"]
                if baseline_result["plain_rjmcmc_flat_params"] is not None
                else np.array([])),
            plain_rjmcmc_model_order_trace=(
                baseline_result["plain_rjmcmc_model_order_trace"]
                if baseline_result["plain_rjmcmc_model_order_trace"] is not None
                else np.array([])),
        )

    # 8 - RESULTS

    return {
        # configuration
        "noise_standard_deviation": noise_standard_deviation,
        "random_seed": random_seed,
        "true_number_of_pulses": true_number_of_pulses,
        "pulse_width_fixed": pulse_width_fixed,
        "nleaves_max": nleaves_max_value,
        "number_of_live_points": nested_result["number_of_live_points_used"],
        "gaussian_move_step_size_scale": gaussian_move_step_size_scale,
        "mcmc_steps_per_replacement": nested_result["mcmc_steps_per_replacement_used"],
        "snr": optimal_matched_filter_snr,

        # model order posterior
        "map_k": nested_result["map_k_estimate"],
        "posterior_true_k": (posterior_probability_per_k[true_number_of_pulses]
                             if (true_number_of_pulses is not None
                                 and true_number_of_pulses <= nleaves_max_value)
                             else np.nan),
        "posterior_per_k": posterior_probability_per_k.tolist(),

        # evidence
        "log_evidence": nested_result["log_evidence_signal_model"],
        "log_evidence_std": nested_result["log_evidence_signal_std"],
        "log_evidence_no_signal": log_evidence_no_signal_model,
        "log_bayes_factor": nested_result["log_bayes_factor_signal_vs_noise"],
        "information_nats": nested_result["information_nats"],

        # diagnostics
        "converged_cleanly": nested_result["loop_terminated_cleanly"],
        "termination_reason": nested_result["termination_reason"],
        "iterations": nested_result["iteration_count_completed"],
        "effective_sample_size": nested_result["effective_sample_size"],
        "clone_fallback_fraction": (
            nested_result["replacement_by_clone_fallback_count"] / total_replacements
            if total_replacements else np.nan),
        "stuck_walker_fraction": (
            nested_result["stuck_walker_count"] / total_replacements
            if total_replacements else np.nan),
        "rejected_survivor_count": nested_result["rejected_survivor_count"],

        # plain RJ-MCMC baseline
        "plain_rjmcmc_map_k": baseline_result["plain_rjmcmc_map_k"],
        "plain_rjmcmc_posterior_per_k": (
            baseline_result["plain_rjmcmc_posterior_per_k"].tolist()
            if baseline_result["plain_rjmcmc_posterior_per_k"] is not None
            else None),
    }
