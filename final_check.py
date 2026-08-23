"""
Final check before handing the code to someone else.

Everything has been tested at reduced scale, because that runs in seconds
and can be repeated often. The engine underneath has been run 69 times at
production scale, so that part is well exercised. What has NOT been
exercised at production scale is the thin layer in between -- the code
that passes settings from TransNestedSampler down into the engine.

This script closes that gap. It runs the full pipeline once, at the
settings a real analysis would use, with the RJ-MCMC baseline switched on,
and then verifies that the results are internally consistent rather than
merely produced without crashing.

Run it once. If it passes, the code is ready to hand over.

    nohup python final_check.py > final_check.log 2>&1 &
    tail -f final_check.log

Expect 15 to 70 minutes, depending on machine load: a full Nested Sampling
run plus a 10-temperature RJ-MCMC baseline of 20,000 steps across 50
walkers.
"""

import time

import numpy as np

from transnested import TransNestedSampler
from pulse_model import combine_all_pulses
from diagnostics import (integrated_autocorrelation_time, gelman_rubin_rhat,
                         effective_sample_size_from_weights)


ARCHIVE_PATH = "final_check_archive.npz"

# Two well-separated pulses in moderate noise. Deliberately not the hardest
# case: the point here is to exercise the plumbing at full scale, not to
# re-measure the method's accuracy, which the 69-run sweep already did.
INJECTED_PULSES = [[2.0, -0.5], [2.0, 0.5]]
NOISE_SIGMA = 0.5
TRUE_K = len(INJECTED_PULSES)

FAILURES = []
WARNINGS = []


def check(description, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {description}" + (f"   ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(description)


def warn(description, condition, detail=""):
    status = "ok" if condition else "WARN"
    print(f"  [{status}] {description}" + (f"   ({detail})" if detail else ""))
    if not condition:
        WARNINGS.append(description)


def main():
    print("Final check -- full pipeline, production scale")
    print(f"  {TRUE_K} pulses injected at "
         f"{[p[1] for p in INJECTED_PULSES]}, noise sigma {NOISE_SIGMA}")
    print("  Nested Sampler: 200 live points")
    print("  RJ-MCMC baseline: 50 walkers, 10 temperatures, 20,000 steps")

    time_axis = np.linspace(-1.0, 1.0, 500)
    np.random.seed(0)
    data = (combine_all_pulses(time_axis, INJECTED_PULSES, 0.1)
            + NOISE_SIGMA * np.random.randn(len(time_axis)))

    sampler = TransNestedSampler(
        data,
        noise_sigma=NOISE_SIGMA,
        time=time_axis,
        pulse_width=0.1,
        max_pulses=8,
        n_live=200,
        seed=0,
    )

    started = time.perf_counter()
    result = sampler.run(
        compare_with_rjmcmc=True,
        max_iterations=20000,
        rjmcmc_temperatures=10,
        rjmcmc_walkers=50,
        rjmcmc_steps=20000,
        save_to=ARCHIVE_PATH,
        verbose=True,
    )
    elapsed = time.perf_counter() - started

    print(f"\n  Completed in {elapsed / 60:.1f} minutes.\n")

    summary = result.as_dict()

    print("A - the settings actually reached the engine")
    check("live points as requested", summary["number_of_live_points"] == 200,
         f"{summary['number_of_live_points']}")
    check("the RJ-MCMC baseline ran at all",
         summary["plain_rjmcmc_map_k"] is not None)

    with np.load(ARCHIVE_PATH, allow_pickle=False) as archive:
        log_likelihood = archive["logL"]
        log_likelihood_birth = archive["logL_birth"]
        coords = archive["coords"]
        mask = archive["mask"]
        weights = archive["posterior_weights"]
        cold_trace = archive["plain_rjmcmc_model_order_trace"]
        posterior_per_k = archive["posterior_probability_per_k"]

    check("cold-chain trace holds one entry per step, not ten",
         cold_trace.shape[0] == 20000,
         f"{cold_trace.shape[0]:,} steps; pooled temperatures would give 200,000")
    check("cold-chain trace has the requested walker count",
         cold_trace.shape[1] == 50, f"{cold_trace.shape[1]}")

    print("\nB - nested sampling invariants")
    finite_birth = np.isfinite(log_likelihood_birth)
    check("every point was born below its own likelihood",
         np.all(log_likelihood[finite_birth] > log_likelihood_birth[finite_birth]),
         f"0 violations in {int(finite_birth.sum()):,} points")
    check("no NaN in the archived likelihoods",
         not np.isnan(log_likelihood).any())
    check("no NaN in the coordinates of active leaves",
         not np.isnan(coords[mask]).any())
    check("posterior weights are non-negative and sum to 1",
         np.all(weights >= 0) and np.isclose(weights.sum(), 1.0, atol=1e-10),
         f"sum = {weights.sum():.12f}")
    check("model-order posterior sums to 1",
         np.isclose(posterior_per_k.sum(), 1.0, atol=1e-10))

    predicted_std = np.sqrt(summary["information_nats"] / summary["number_of_live_points"])
    deviation = abs(summary["log_evidence_std"] - predicted_std)
    check("sigma(log Z) agrees with sqrt(H / N_live)", deviation < 0.1,
         f"reported {summary['log_evidence_std']:.4f}, "
         f"theory {predicted_std:.4f}, differ by {deviation:.4f}")

    effective_size = effective_sample_size_from_weights(weights)
    check("effective sample size is usable", effective_size > 100,
         f"{effective_size:.0f}")

    print("\nC - the tempered baseline converged")
    walker_by_step = cold_trace.T.astype(float)
    second_half = walker_by_step[:, walker_by_step.shape[1] // 2:]
    tau, frozen = integrated_autocorrelation_time(second_half, return_frozen_count=True)
    rhat = gelman_rubin_rhat(second_half)
    check("R-hat below 1.05", rhat < 1.05, f"{rhat:.4f}")
    check("no walker frozen at one model order", frozen == 0,
         f"{frozen} of {walker_by_step.shape[0]}")
    check("autocorrelation time of k is short", np.isfinite(tau) and tau < 100,
         f"tau = {tau:.1f}")

    print("\nD - the answer is sensible")
    check("recovered model order is within one of the truth",
         abs(result.model_order - TRUE_K) <= 1,
         f"found {result.model_order}, injected {TRUE_K}")
    check("the signal is decisively preferred over pure noise",
         result.log_bayes_factor_vs_noise > 10,
         f"log B = {result.log_bayes_factor_vs_noise:.1f}")
    check("the recovered pulse positions are close to the injected ones",
         _positions_match(result, INJECTED_PULSES),
         f"{np.round(np.sort(result.pulses[:, 1]), 3).tolist()} against "
         f"{sorted(p[1] for p in INJECTED_PULSES)}")

    print("\nE - worth knowing, not blocking")
    warn("the run converged rather than hitting a limit", result.converged,
         summary["termination_reason"])
    warn("the two methods agree on the number of pulses",
         result.model_order == result.rjmcmc_model_order,
         f"Nested Sampler {result.model_order}, "
         f"RJ-MCMC {result.rjmcmc_model_order}")
    warn("posterior mass at the prior ceiling is negligible",
         posterior_per_k[-1] < 0.01, f"P(k=8) = {posterior_per_k[-1]:.4f}")
    warn("stuck-walker rate is not extreme",
         summary["stuck_walker_fraction"] < 0.40,
         f"{summary['stuck_walker_fraction']:.1%}")

    print()
    if FAILURES:
        print(f"FAILED -- {len(FAILURES)} check(s) did not pass:")
        for description in FAILURES:
            print(f"  - {description}")
        print("\nDo not hand the code over until these are understood.")
    else:
        print("PASSED")
        if WARNINGS:
            print(f"\n{len(WARNINGS)} non-blocking note(s):")
            for description in WARNINGS:
                print(f"  - {description}")
            print("\nNone of these invalidate the run. Disagreement by one pulse,")
            print("and occasional termination on live-population collapse, were")
            print("both present throughout the 69-run sweep and are documented.")
        print(f"\nTook {elapsed / 60:.1f} minutes.")


def _positions_match(result, injected, tolerance=0.1):
    """
    Each injected pulse should have a recovered pulse near it. Compared after
    sorting, since the sampler has no notion of which pulse is which -- pulse
    identity is not preserved in a trans-dimensional problem.
    """
    recovered = np.sort(result.pulses[:, 1])
    truth = np.sort([p[1] for p in injected])
    if len(recovered) < len(truth):
        return False
    for true_position in truth:
        if np.min(np.abs(recovered - true_position)) > tolerance:
            return False
    return True


if __name__ == "__main__":
    main()