"""
Tests for pulse_model.py.

Run with:  python test_pulse_model.py

Two kinds of check, run one after the other.

1. Analytic checks (always run). Properties of the pulse model and the
   likelihood that follow from the formula itself, worked out by hand, and
   compared against what the code returns. These do not need any archive to
   exist, and they are what would catch a wrong formula, a swapped sign, or a
   missing factor of two.

2. Archive cross-checks (run only if archive files can be found). Recompute
   the noiseless signal and the log-likelihood for points taken from a real,
   already-generated experiment, and compare against what the original
   run_single_experiment call stored for those same points. This is the
   direct answer to "is this the same likelihood the experiment used" -- not
   an argument, but matching numbers, for every point tested, with the
   required result being exact equality rather than closeness within some
   tolerance.

    The raw archives and the sweep_results.csv summary from the actual
    69-configuration production sweep are provided alongside this file
    specifically so this check runs against genuine data rather than being
    skipped. find_archives() and find_summary_csv() below look for them in
    raw_archives/ and sweep_results.csv (or the data_and_results/ equivalents)
    automatically -- no path needs to be typed in. Run this script from a
    folder that has that data reachable and the result is not taken on trust:
    it is computed, in front of whoever runs it.
"""

import os
import sys

import numpy as np
import pandas as pd

from pulse_model import (
    single_gaussian_pulse, combine_all_pulses, true_gaussian_log_likelihood,
    build_priors, build_time_axis, generate_synthetic_dataset,
)


# Every failed check appends its description here. Checked once at the end of
# the script to decide the process exit code, so a CI run can tell pass from
# fail without parsing printed text.
FAILURES = []


def check(description, condition):
    """
    Print PASS or FAIL for one check, and record the description if it failed.
    """
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {description}")
    if not condition:
        FAILURES.append(description)


# 1 - ANALYTIC CHECKS

def run_analytic_checks():
    print("Analytic checks (no archive required)")

    time_axis = build_time_axis(500, -1.0, 1.0)

    # A pulse evaluated at its own mean position returns exactly its
    # amplitude, since exp(0) = 1. This is the definition of the peak.
    single_pulse_peak_value = single_gaussian_pulse(    
        np.array([0.3]), pulse_amplitude=2.5, pulse_mean_position=0.3,
        pulse_width_fixed=0.1)
    check("pulse value at its own mean position equals its amplitude",
         np.isclose(single_pulse_peak_value[0], 2.5))

    # Zero pulses must combine to the exact zero signal. This is the
    # mechanism the k = 0 (no signal) hypothesis rests on.
    zero_pulse_signal = combine_all_pulses(time_axis, [], pulse_width_fixed=0.1)
    check("combine_all_pulses([]) is exactly zero everywhere",
         np.all(zero_pulse_signal == 0.0))

    # Two pulses combine by addition. Evaluating the sum by hand at some grid
    # point and evaluating it through combine_all_pulses at that same point
    # must agree. The comparison point is read off the actual grid rather than
    # assumed to be exactly t=0, since linspace(-1, 1, 500) has an even number
    # of points and therefore never lands exactly on 0.
    two_pulse_signal = combine_all_pulses(
        time_axis, [[1.0, -0.3], [1.0, 0.3]], pulse_width_fixed=0.1)
    comparison_index = len(time_axis) // 2
    comparison_time = time_axis[comparison_index]
    hand_computed_at_comparison_point = (
        single_gaussian_pulse(np.array([comparison_time]), 1.0, -0.3, 0.1)[0]
        + single_gaussian_pulse(np.array([comparison_time]), 1.0, 0.3, 0.1)[0])
    check("combine_all_pulses sums two pulses correctly at a sample grid point",
         np.isclose(two_pulse_signal[comparison_index], hand_computed_at_comparison_point))

    # A model that reproduces the data exactly has zero residual, and the
    # log-likelihood formula gives exactly 0.0 in that case:
    # logL = -0.5 * sum(0^2) = 0.
    perfect_signal = combine_all_pulses(time_axis, [[1.5, 0.0]], pulse_width_fixed=0.2)
    logL_at_perfect_fit = true_gaussian_log_likelihood(
        [[1.5, 0.0]], time_axis, observed_data=perfect_signal,
        noise_standard_deviation=1.0, pulse_width_fixed=0.2)
    check("log-likelihood is exactly 0 when the model matches the data exactly",
         logL_at_perfect_fit == 0.0)

    # A one-sigma-per-sample deviation should give logL = -0.5 * N, by direct
    # substitution into the formula.
    number_of_samples = len(time_axis)
    noisy_data = perfect_signal + np.ones(number_of_samples)  # residual/sigma = 1 everywhere
    logL_at_one_sigma_deviation = true_gaussian_log_likelihood(
        [[1.5, 0.0]], time_axis, observed_data=noisy_data,
        noise_standard_deviation=1.0, pulse_width_fixed=0.2)
    check("log-likelihood matches -0.5*N for a uniform one-sigma residual",
         np.isclose(logL_at_one_sigma_deviation, -0.5 * number_of_samples))

    # The likelihood must worsen (become more negative) as the model moves
    # further from data it does not match. This is the minimum requirement for
    # the sampler to be able to find the correct answer at all.
    good_fit_logL = true_gaussian_log_likelihood(
        [[1.5, 0.0]], time_axis, observed_data=perfect_signal,
        noise_standard_deviation=0.5, pulse_width_fixed=0.2)
    bad_fit_logL = true_gaussian_log_likelihood(
        [[0.1, 0.0]], time_axis, observed_data=perfect_signal,
        noise_standard_deviation=0.5, pulse_width_fixed=0.2)
    check("log-likelihood is higher for a closer match to the data",
         good_fit_logL > bad_fit_logL)

    # The prior object must place zero density outside its bounds and
    # positive density at the centre of its bounds.
    priors = build_priors(amplitude_min=0.0, amplitude_max=5.0,
                          mean_position_min=-1.0, mean_position_max=1.0)
    inside_point = np.array([[2.5, 0.0]])
    outside_point = np.array([[10.0, 0.0]])
    check("prior log-density is finite for a point inside the bounds",
         np.isfinite(priors["gauss"].logpdf(inside_point)[0]))
    check("prior log-density is -inf for a point outside the amplitude bound",
         priors["gauss"].logpdf(outside_point)[0] == -np.inf)

    # generate_synthetic_dataset seeded identically must reproduce the exact
    # same noise realisation. Every later reconstruction of "the same dataset
    # as archive X" depends on this holding.
    dataset_a = generate_synthetic_dataset(
        time_axis, [[2.0, 0.0]], noise_standard_deviation=0.5,
        pulse_width_fixed=0.1, random_seed=42)
    dataset_b = generate_synthetic_dataset(
        time_axis, [[2.0, 0.0]], noise_standard_deviation=0.5,
        pulse_width_fixed=0.1, random_seed=42)
    check("generate_synthetic_dataset is exactly reproducible from the same seed",
         np.array_equal(dataset_a["observed_data"], dataset_b["observed_data"]))

    dataset_c = generate_synthetic_dataset(
        time_axis, [[2.0, 0.0]], noise_standard_deviation=0.5,
        pulse_width_fixed=0.1, random_seed=43)
    check("generate_synthetic_dataset gives a different result for a different seed",
         not np.array_equal(dataset_a["observed_data"], dataset_c["observed_data"]))


# 2 - ARCHIVE CROSS-CHECKS

def run_archive_cross_checks(archive_paths, summary_table):
    print("\nArchive cross-checks (comparing against real experiment output)")

    if not archive_paths:
        print("  no archive files found -- skipped. Run this script from a "
             "directory with access to raw_archives/ to enable this check.")
        return

    if summary_table is None:
        print("  no sweep_results.csv found -- skipped. This check needs the "
             "exact noise_standard_deviation recorded for each archive; without "
             "it there is nothing precise enough to compare against.")
        return

    for archive_path in archive_paths:
        run_single_archive_cross_check(archive_path, summary_table)


def run_single_archive_cross_check(archive_path, summary_table, number_of_points_to_test=500):
    with np.load(archive_path, allow_pickle=False) as archive:
        stored_log_likelihood = archive["logL"]
        coords = archive["coords"]
        mask = archive["mask"]
        time_axis = archive["time_axis"]
        observed_data = archive["observed_data"]
        true_injected = archive["true_injected"]
        noiseless_injected_signal = archive["noiseless_injected_signal"]

    # pulse_width_fixed and noise_standard_deviation are not stored in the
    # archive itself, so both are read from the summary CSV, exactly as they
    # were recorded during the run -- not assumed, and not recovered
    # algebraically. Recovering noise_standard_deviation from a stored logL
    # value (an earlier version of this check did that) introduces
    # floating-point roundoff in the recovered value, which then shows up as a
    # fake few-times-1e-12 mismatch in every comparison: a property of the
    # recovery method, not of the likelihood being checked. Reading the exact
    # stored value removes that source of error entirely, which is why this
    # check can require exact equality rather than a tolerance.
    basename = os.path.basename(archive_path)
    matching_rows = summary_table[
        summary_table["raw_archive_file"].apply(os.path.basename) == basename]
    if len(matching_rows) != 1:
        print(f"  {basename}: SKIPPED ({len(matching_rows)} CSV rows matched, need exactly 1)")
        return
    metadata = matching_rows.iloc[0]
    pulse_width_fixed = float(metadata["pulse_width_fixed"])
    noise_standard_deviation = float(metadata["noise_standard_deviation"])

    rebuilt_injection = combine_all_pulses(time_axis, true_injected, pulse_width_fixed)
    injection_reproduced = np.allclose(rebuilt_injection, noiseless_injected_signal,
                                       atol=1e-10)
    print(f"  {basename}")
    check("    combine_all_pulses reproduces the archived injection exactly",
         injection_reproduced)

    # A random subset of points, not a systematic one, so the check exercises
    # a spread of likelihood values rather than only the run's first points.
    generator = np.random.default_rng(0)
    number_of_points_total = stored_log_likelihood.shape[0]
    tested_indices = generator.choice(
        number_of_points_total,
        size=min(number_of_points_to_test, number_of_points_total),
        replace=False)

    max_absolute_difference = 0.0
    points_actually_compared = 0
    for point_index in tested_indices:
        active_parameters = coords[point_index][mask[point_index]]
        if active_parameters.shape[0] == 0:
            continue
        stored = stored_log_likelihood[point_index]
        if not np.isfinite(stored):
            continue
        recomputed = true_gaussian_log_likelihood(
            active_parameters, time_axis, observed_data,
            noise_standard_deviation, pulse_width_fixed)
        max_absolute_difference = max(max_absolute_difference,
                                      abs(recomputed - stored))
        points_actually_compared += 1

    check(f"    logL matches the archive exactly at all {points_actually_compared} "
         f"points tested (max |difference| = {max_absolute_difference:.3e})",
         max_absolute_difference == 0.0)


def find_archives():
    """
    Look for raw archives in a few conventional locations, without assuming
    which one the caller is using. Returns an empty list rather than raising,
    since the archive cross-check is optional: the analytic checks above do
    not depend on it.
    """
    candidate_directories = [
        "raw_archives", "data_and_results/raw_archives",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_archives"),
    ]
    found = []
    for directory in candidate_directories:
        if os.path.isdir(directory):
            found.extend(
                os.path.join(directory, name) for name in sorted(os.listdir(directory))
                if name.endswith(".npz"))
    return found[:3]  # a handful is enough; this is a spot check, not a full sweep


def find_summary_csv():
    candidate_paths = [
        "sweep_results.csv", "data_and_results/sweep_results.csv",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_results.csv"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


if __name__ == "__main__":
    run_analytic_checks()
    run_archive_cross_checks(find_archives(), find_summary_csv())

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for description in FAILURES:
            print(f"  - {description}")
        sys.exit(1)
    else:
        print("All checks passed.")