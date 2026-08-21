"""
Tests for diagnostics.py.

Run with: python test_diagnostics.py

Every check uses a sequence whose correct answer is known in advance, either
analytically or by construction, so a wrong implementation fails rather than
merely looking plausible.
"""

import numpy as np

from diagnostics import (
    integrated_autocorrelation_time, gelman_rubin_rhat,
    effective_sample_size_from_weights, effective_sample_size_from_chain,
)


FAILURES = []


def check(description, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {description}")
    if not condition:
        FAILURES.append(description)


def make_ar1_chains(phi, n_walkers, n_steps, seed):
    """
    AR(1): x_t = phi * x_{t-1} + noise. Its autocorrelation is rho(k) = phi^k,
    so the integrated autocorrelation time is exactly (1 + phi) / (1 - phi).
    """
    rng = np.random.default_rng(seed)
    chains = np.zeros((n_walkers, n_steps))
    for walker in range(n_walkers):
        value = 0.0
        for step in range(1, n_steps):
            value = phi * value + rng.standard_normal()
            chains[walker, step] = value
    return chains


def run_checks():
    print("diagnostics.py checks")

    # tau against the analytic AR(1) value, at two different correlation levels.
    for phi in (0.5, 0.9):
        expected_tau = (1 + phi) / (1 - phi)
        chains = make_ar1_chains(phi, n_walkers=8, n_steps=20000, seed=0)
        estimated_tau = integrated_autocorrelation_time(chains)
        relative_error = abs(estimated_tau - expected_tau) / expected_tau
        check(f"tau for AR(1) phi={phi}: expected {expected_tau:.1f}, "
             f"got {estimated_tau:.1f} ({relative_error:.0%} off)",
             relative_error < 0.15)

    # Frozen walkers must be counted and excluded, without disturbing tau.
    chains = make_ar1_chains(0.9, n_walkers=8, n_steps=20000, seed=0)
    tau_clean = integrated_autocorrelation_time(chains)
    chains_with_frozen = chains.copy()
    chains_with_frozen[0] = 3.0
    chains_with_frozen[1] = -7.5
    tau_mixed, frozen_count = integrated_autocorrelation_time(
        chains_with_frozen, return_frozen_count=True)
    check("two frozen walkers are counted correctly", frozen_count == 2)
    check(f"tau is not corrupted by frozen walkers "
         f"({tau_clean:.1f} clean vs {tau_mixed:.1f} with two frozen)",
         np.isfinite(tau_mixed) and abs(tau_mixed - tau_clean) / tau_clean < 0.15)

    all_frozen = np.ones((5, 1000)) * 7.0
    tau_all, frozen_all = integrated_autocorrelation_time(
        all_frozen, return_frozen_count=True)
    check("all-frozen input gives tau = inf, not NaN",
         np.isinf(tau_all) and frozen_all == 5)

    # R-hat: independent chains from one distribution should give ~1.0;
    # chains offset from each other should give clearly more than 1.
    rng = np.random.default_rng(1)
    well_mixed = rng.standard_normal((10, 5000))
    check(f"R-hat for well-mixed chains is near 1 "
         f"({gelman_rubin_rhat(well_mixed):.4f})",
         abs(gelman_rubin_rhat(well_mixed) - 1.0) < 0.01)

    stuck = rng.standard_normal((10, 5000)) + rng.standard_normal(10)[:, None] * 5
    check(f"R-hat for chains stuck at different means is well above 1 "
         f"({gelman_rubin_rhat(stuck):.2f})",
         gelman_rubin_rhat(stuck) > 2.0)

    # Weighted ESS against cases with a known answer.
    check("equal weights give ESS = N",
         np.isclose(effective_sample_size_from_weights(np.ones(1000) / 1000), 1000.0))

    one_dominant = np.zeros(1000)
    one_dominant[0] = 0.999
    one_dominant[1:] = 0.001 / 999
    check(f"one dominant weight gives ESS near 1 "
         f"({effective_sample_size_from_weights(one_dominant):.2f})",
         effective_sample_size_from_weights(one_dominant) < 1.1)

    ten_heavy = np.zeros(1000)
    ten_heavy[:10] = 0.05
    ten_heavy[10:] = 0.5 / 990
    check(f"ten heavy weights carrying half the mass give ESS around 40 "
         f"({effective_sample_size_from_weights(ten_heavy):.1f})",
         30 < effective_sample_size_from_weights(ten_heavy) < 50)

    # Chain ESS must equal total draws over tau.
    chains = make_ar1_chains(0.9, n_walkers=4, n_steps=10000, seed=3)
    tau = integrated_autocorrelation_time(chains)
    expected_ess = 4 * 10000 / tau
    check("chain ESS equals n_walkers * n_steps / tau",
         np.isclose(effective_sample_size_from_chain(chains), expected_ess))


if __name__ == "__main__":
    run_checks()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for description in FAILURES:
            print(f"  - {description}")
    else:
        print("All checks passed.")
