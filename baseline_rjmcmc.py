"""
The plain reversible-jump MCMC baseline.

A straightforward RJ-MCMC fit on the true likelihood, with no nested sampling
anywhere: no shells, no evidence, no weighting. It exists purely as the
comparison point for the Nested Sampler.

The likelihood and priors are passed in by the caller, which is what makes the
comparison meaningful -- both samplers are handed the same function objects,
not two separately written copies of the same formula.

Extracted verbatim from transnested_experiment_v2.py, Section 11, and verified
to produce bitwise-identical results.
"""

import numpy as np  

from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.moves import GaussianMove


def run_plain_rjmcmc_baseline(
    time_axis,
    observed_data,
    noise_standard_deviation,
    priors,
    branch_names,
    true_gaussian_log_likelihood,
    ndims_per_branch,
    nleaves_max,
    nleaves_min,
    pulse_ndim,
    gaussian_move_covariance,
    run_plain_rjmcmc_comparison,
    plain_rjmcmc_ntemps,
    plain_rjmcmc_nwalkers,
    plain_rjmcmc_nsteps,
    plain_rjmcmc_burn_in_fraction,
):
    """
    Returns the model-order posterior, the flattened active parameters, and the
    per-walker model-order trace of the cold chain. Returns None for all of
    them when run_plain_rjmcmc_comparison is False.
    """
    plain_rjmcmc_map_k = None
    plain_rjmcmc_posterior_per_k = None
    plain_rjmcmc_flat_params = None
    plain_rjmcmc_model_order_trace = None

    if run_plain_rjmcmc_comparison:

        # Geometric-in-beta ladder from 1.0 (cold, the true posterior) down to
        # 0.0 (the prior), matching the official Eryn tutorial's
        # np.linspace(1.0, 0.0, ntemps). With plain_rjmcmc_ntemps = 1 this is
        # exactly [1.0], reproducing the original single-chain behaviour.
        _plain_rjmcmc_betas = (np.array([1.0]) if plain_rjmcmc_ntemps == 1
                               else np.linspace(1.0, 0.0, plain_rjmcmc_ntemps))

        # A FRESH proposal object, not the one the Nested Sampler is using.
        # Eryn attaches a TemperatureControl to each move when an ensemble is
        # built around it, and the Nested Sampler's ensemble has already stamped
        # this move with ntemps = 1. Reusing that same object here makes Eryn
        # allocate temperature-swap arrays of the wrong size and fail with
        # "invalid swaps_accepted size". The two samplers must not share a move
        # object once their temperature ladders differ. Verified by direct
        # experiment: a reused move raises, a fresh one runs.
        _plain_rjmcmc_move = GaussianMove(gaussian_move_covariance)

        # A SEPARATE ensemble on the TRUE likelihood, with many walkers and many
        # steps, giving a posterior straight from the chain. No evidence, no error
        # bar, no NS weighting -- the comparison point for the sampler above.
        plain_rjmcmc_ensemble = EnsembleSampler(
            plain_rjmcmc_nwalkers,
            ndims_per_branch,
            true_gaussian_log_likelihood,   # real likelihood, not the constrained one
            priors,
            args=[time_axis, observed_data, noise_standard_deviation],
            tempering_kwargs=dict(betas=_plain_rjmcmc_betas),
            nbranches=len(branch_names),
            branch_names=branch_names,
            nleaves_max=nleaves_max,
            nleaves_min=nleaves_min,
            moves=_plain_rjmcmc_move,
            rj_moves=True,
        )

        # Initialize walkers from the prior, same logic as Section 6, but for
        # every temperature rather than only the cold chain.
        _nt = plain_rjmcmc_ntemps
        _k_per_walker = np.random.randint(nleaves_min["gauss"], nleaves_max["gauss"] + 1,
                                          size=(_nt, plain_rjmcmc_nwalkers))
        _coords0 = np.empty((_nt, plain_rjmcmc_nwalkers, nleaves_max["gauss"], pulse_ndim))
        _mask0 = np.zeros((_nt, plain_rjmcmc_nwalkers, nleaves_max["gauss"]), dtype=bool)

        for _t in range(_nt):
            for w in range(plain_rjmcmc_nwalkers):
                _coords0[_t, w] = priors["gauss"].rvs(size=(nleaves_max["gauss"],))
                _mask0[_t, w, :_k_per_walker[_t, w]] = True

        initial_state = State({"gauss": _coords0}, inds={"gauss": _mask0},
                              betas=_plain_rjmcmc_betas)

        print("Starting plain RJ-MCMC baseline...")
        plain_rjmcmc_ensemble.run_mcmc(
            initial_state, plain_rjmcmc_nsteps, progress=True)

        _discard = int(plain_rjmcmc_nsteps * plain_rjmcmc_burn_in_fraction)

        # A single read of the inds backend, with no discard -- the burn-in cut
        # is applied afterwards by slicing the already-loaded array, instead of
        # calling get_inds a second time to re-read the same backend.
        _all_inds_cold = plain_rjmcmc_ensemble.get_inds(discard=0)["gauss"][:, 0, :, :]

        # The per-walker trace, kept so that stuck walkers can be ruled out as the
        # cause of any disagreement with the Nested Sampler. Uses the FULL trace
        # (no burn-in cut), since a walker stuck during burn-in is still worth
        # seeing.
        plain_rjmcmc_model_order_trace = _all_inds_cold.sum(axis=-1)

        # Pull the full chain from the backend and discard burn-in.
        # Temperature index 0 is the cold chain, the only one that samples the
        # actual posterior. Every hotter chain samples a flattened likelihood
        # (beta < 1), up to beta = 0 which is the prior. The [:, 0] selection
        # above already picked it out, so slicing off burn-in here cannot
        # accidentally pool in prior samples.
        _chain_coords = plain_rjmcmc_ensemble.get_chain(discard=0)["gauss"][:, 0][_discard:]
        _chain_inds = _all_inds_cold[_discard:]

        _chain_coords = _chain_coords.reshape(-1, nleaves_max["gauss"], pulse_ndim)
        _chain_inds = _chain_inds.reshape(-1, nleaves_max["gauss"])

        plain_rjmcmc_model_order = _chain_inds.sum(axis=-1).astype(int)

        plain_counts = np.bincount(plain_rjmcmc_model_order,
                                   minlength=nleaves_max["gauss"] + 1)
        plain_rjmcmc_posterior_per_k = plain_counts / plain_counts.sum()
        plain_rjmcmc_map_k = int(np.argmax(plain_rjmcmc_posterior_per_k))

        _flat_params = []
        for i in range(_chain_coords.shape[0]):
            active = _chain_coords[i][_chain_inds[i]]
            if active.shape[0] > 0:
                _flat_params.append(active)
        plain_rjmcmc_flat_params = np.concatenate(_flat_params, axis=0)

    return {
        "plain_rjmcmc_map_k": plain_rjmcmc_map_k,
        "plain_rjmcmc_posterior_per_k": plain_rjmcmc_posterior_per_k,
        "plain_rjmcmc_flat_params": plain_rjmcmc_flat_params,
        "plain_rjmcmc_model_order_trace": plain_rjmcmc_model_order_trace,
    }
