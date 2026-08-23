"""
The trans-dimensional Nested Sampling algorithm.

Owns the constrained-likelihood gate, live-point bookkeeping, the worst-point
replacement loop, evidence accumulation, and the posterior built from the
resulting archive.

The pulse model, the physical likelihood and the priors are NOT defined here.
They are supplied by the caller, which gets them from pulse_model.py, so this
module and baseline_rjmcmc.py are guaranteed to be solving the same problem
rather than two problems that happen to look alike.

Extracted verbatim from transnested_experiment_v2.py, Sections 4 (the gate
only) to 10, and verified to produce bitwise-identical results.
"""

import numpy as np

from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.moves import GaussianMove
from anesthetic import NestedSamples
from tqdm import trange

def run_nested_sampler(
    time_axis,
    observed_data,
    noise_standard_deviation,
    priors,
    branch_names,
    amplitude_min,
    amplitude_max,
    mean_position_min,
    mean_position_max,
    true_gaussian_log_likelihood,
    combine_all_pulses,
    true_number_of_pulses,
    true_injected_pulse_parameters,
    log_evidence_no_signal_model,
    nleaves_max_value,
    nleaves_min_value,
    number_of_live_points,
    gaussian_move_step_size_scale,
    mcmc_steps_per_replacement,
    evidence_convergence_tolerance,
    maximum_allowed_iterations,
    prior_odds_signal_vs_no_signal,
    verbose,
):
    """
    Run one Nested Sampling analysis and return everything the caller needs to
    archive it or summarise it.

    true_gaussian_log_likelihood and combine_all_pulses are passed in rather
    than imported, so the caller can guarantee the SAME function objects are
    handed to the baseline sampler as well.
    """
    # container: writing into element [0] changes the contents of the SAME object,
    # so the closure below sees the new value on its very next call.
    current_logL0_threshold = np.array([-np.inf])

    def eryn_constrained_log_likelihood(pulse_parameters_array, time_axis,
                                        observed_data, noise_standard_deviation):
        """
        Pass/fail gate handed to Eryn in place of a real likelihood.

        Flat (0.0) inside the current shell and -inf outside, so the walker performs
        a pure prior random walk rather than climbing toward the peak. The comparison
        is strict, and NaN fails it, which is the desired behaviour.
        """
        true_log_likelihood_value = true_gaussian_log_likelihood(
            pulse_parameters_array, time_axis, observed_data, noise_standard_deviation)

        if true_log_likelihood_value > current_logL0_threshold[0]:
            return 0.0

        return -np.inf

    # SECTION 5 - BRANCH STRUCTURE SETUP

    pulse_ndim = 2
    ndims_per_branch = {"gauss": pulse_ndim}

    # nleaves_min is 1 by necessity, not by choice: Eryn returns NaN for a walker
    # with zero active leaves. The k = 0 hypothesis is computed analytically below.
    nleaves_min = {"gauss": nleaves_min_value}
    nleaves_max = {"gauss": nleaves_max_value}

    
    # This protects against a search space that cannot contain the answer. It
    # is relaxed when k is held FIXED (nleaves_min == nleaves_max): computing Z
    # for a deliberately under-specified model, k below the injected count, is
    # then the entire point, since the Bayes factor between k values is what
    # reveals what the data actually supports.
    if (true_number_of_pulses is not None
            and nleaves_min["gauss"] != nleaves_max["gauss"]):
        assert nleaves_max["gauss"] >= true_number_of_pulses, (
            f"nleaves_max = {nleaves_max['gauss']} is smaller than the true injected pulse "
            f"count ({true_number_of_pulses}). The correct answer lies outside the search "
            f"space and could never be recovered.")

    # SECTION 6 - INITIALIZING LIVE POINTS

    number_of_temperatures = 1

    # [TUNABLE] Scaled from nleaves_max rather than from the true pulse count, which
    #   is unknown in any real application. The evidence uncertainty scales as
    #   sqrt(H / nlive), and the iteration count scales linearly with it.
    if number_of_live_points is None:
        number_of_live_points = 25 * pulse_ndim * nleaves_max["gauss"] // 2

    # Step 1: the discrete part of the prior, one draw of k per live point.
    number_of_active_pulses_per_live_point = np.random.randint(
        low=nleaves_min["gauss"],
        high=nleaves_max["gauss"] + 1,
        size=(number_of_temperatures, number_of_live_points),
    )

    # Steps 2 and 3: the padded coordinate array and the boolean mask Eryn calls inds.
    live_point_coordinates = {
        "gauss": np.empty((number_of_temperatures, number_of_live_points,
                           nleaves_max["gauss"], ndims_per_branch["gauss"]))
    }
    live_point_active_leaf_mask = {
        "gauss": np.zeros((number_of_temperatures, number_of_live_points,
                           nleaves_max["gauss"]), dtype=bool)
    }

    # Step 4: fill EVERY slot with a valid prior draw, then activate the first k.
    # Filling the inactive slots too means a later birth move can never reactivate a
    # slot holding uninitialised memory.
    for temperature_index in range(number_of_temperatures):
        for live_point_index in range(number_of_live_points):

            active_pulse_count_this_live_point = number_of_active_pulses_per_live_point[
                temperature_index, live_point_index]

            live_point_coordinates["gauss"][temperature_index, live_point_index] = (
                priors["gauss"].rvs(size=(nleaves_max["gauss"],)))

            live_point_active_leaf_mask["gauss"][
                temperature_index, live_point_index,
                :active_pulse_count_this_live_point] = True

    # SECTION 7 - BUILDING THE EnsembleSampler OBJECT

    gaussian_move_covariance = {
        "gauss": np.diag(np.ones(ndims_per_branch["gauss"])) * gaussian_move_step_size_scale
    }
    in_model_proposal_move = GaussianMove(gaussian_move_covariance)

    # One walker, because one Nested Sampling iteration requires exactly one
    # replacement point. Unrelated to number_of_live_points: the population is
    # maintained by the outer loop, not by Eryn.
    number_of_walkers = 1

    ensemble = EnsembleSampler(
        number_of_walkers,
        ndims_per_branch,
        eryn_constrained_log_likelihood,   # the pass/fail gate, not the true likelihood
        priors,
        args=[time_axis, observed_data, noise_standard_deviation],
        tempering_kwargs=dict(ntemps=number_of_temperatures),
        nbranches=len(branch_names),
        branch_names=branch_names,
        nleaves_max=nleaves_max,
        nleaves_min=nleaves_min,
        moves=in_model_proposal_move,
        rj_moves=True,                     # birth and death proposals, changing k
    )

    # SECTION 8 - NESTED SAMPLING LOGIC LAYER

    # 8.1 - Live-point bookkeeping arrays

    live_point_true_log_likelihood = np.empty(number_of_live_points)

    for live_point_index in range(number_of_live_points):
        # Only ACTIVE leaves enter the likelihood. Selection is by boolean mask
        # rather than by slicing, since active slots need not be contiguous.
        active_mask_this_point = live_point_active_leaf_mask["gauss"][0, live_point_index]
        active_parameters_this_point = live_point_coordinates["gauss"][0, live_point_index][active_mask_this_point]

        live_point_true_log_likelihood[live_point_index] = true_gaussian_log_likelihood(
            active_parameters_this_point, time_axis, observed_data, noise_standard_deviation)

    # The initial points were drawn from the unconstrained prior, so their birth
    # threshold is -inf.
    live_point_birth_log_likelihood = np.full(number_of_live_points, -np.inf)

    # 8.2 - Dead-point archive

    dead_point_coordinates = np.empty((0, nleaves_max["gauss"], ndims_per_branch["gauss"]))
    dead_point_active_leaf_mask = np.empty((0, nleaves_max["gauss"]), dtype=bool)
    dead_point_log_likelihood = np.empty(0)
    dead_point_birth_log_likelihood = np.empty(0)

    # 8.3 - Refill inactive leaf slots

    def sanitize_inactive_leaf_coordinates(coordinates_array, active_mask_array):
        """
        Return a copy in which every INACTIVE slot has been overwritten with a fresh
        prior draw. Active slots are untouched and the input is never modified.

        Eryn's death moves can leave NaN in a slot once it has been deactivated. The
        likelihood never reads it while masked out, but a later birth move may
        reactivate it and read whatever it contains, at which point the NaN enters
        live arithmetic and spreads silently.
        """
        sanitized_coordinates = coordinates_array.copy()

        # Indices where the mask is False. The tilde is elementwise negation.
        inactive_slot_indices = np.where(~active_mask_array)[0]

        if len(inactive_slot_indices) > 0:
            sanitized_coordinates[inactive_slot_indices] = priors["gauss"].rvs(
                size=(len(inactive_slot_indices),))

        return sanitized_coordinates

    # 8.4 - Enforce the k >= 1 invariant

    def enforce_minimum_one_active_leaf(coordinates_array, active_mask_array):
        """
        Return (coordinates, mask) guaranteed to hold at least one active leaf.

        If the point already satisfies the invariant the inputs are returned
        unchanged. Otherwise slot 0 is activated and filled with a prior draw,
        yielding a legitimate k = 1 point rather than a corrupted one.
        """
        if active_mask_array.sum() >= 1:
            return coordinates_array, active_mask_array

        repaired_mask = active_mask_array.copy()
        repaired_coordinates = coordinates_array.copy()

        repaired_mask[0] = True
        repaired_coordinates[0] = priors["gauss"].rvs(size=(1,))[0]

        return repaired_coordinates, repaired_mask

    # 8.5 - Log-prior for a single point

    def compute_log_prior_for_one_point(active_parameters_array):
        """
        Sum the log-prior density over the ACTIVE leaves of one point.

        The per-pulse priors are independent, so the joint prior is their product and
        the log-prior is their sum. An empty input returns 0.0, the logarithm of the
        empty product.
        """
        n_active = active_parameters_array.shape[0]

        if n_active == 0:
            return 0.0

        return float(np.sum(priors["gauss"].logpdf(active_parameters_array)))

    # 8.6 - One Nested Sampling iteration

    def are_active_parameters_within_prior_bounds(active_parameters_array):
        """Return True if every active leaf lies inside the uniform prior support."""
        if active_parameters_array.shape[0] == 0:
            return True

        amplitude_ok = np.all((active_parameters_array[:, 0] >= amplitude_min) &
                              (active_parameters_array[:, 0] <= amplitude_max))

        mean_ok = np.all((active_parameters_array[:, 1] >= mean_position_min) &
                         (active_parameters_array[:, 1] <= mean_position_max))

        return bool(amplitude_ok and mean_ok)

    def evaluate_live_point_validity(coordinates_array, active_mask_array):
        """
        Return (is_usable, true_logL) for one point.

        The checks run in this order because each presupposes the last: bounds cannot
        be tested on NaN, and the likelihood should not be evaluated on out-of-bounds
        coordinates. An unusable point returns -inf, which is a signal to the caller
        and deliberately NOT written back into live_point_true_log_likelihood: doing
        so would make that point the argmin on the next iteration and reset the
        threshold, destroying the compression accumulated so far.
        """
        active_parameters = coordinates_array[active_mask_array]

        if active_parameters.shape[0] < 1:
            return False, -np.inf

        if not np.all(np.isfinite(active_parameters)):
            return False, -np.inf

        if not are_active_parameters_within_prior_bounds(active_parameters):
            return False, -np.inf

        true_logL = true_gaussian_log_likelihood(
            active_parameters, time_axis, observed_data, noise_standard_deviation)

        if not np.isfinite(true_logL):
            return False, -np.inf

        return bool(true_logL > current_logL0_threshold[0]), true_logL

    # Counters, so that degraded behaviour is visible rather than silent.
    replacement_by_eryn_count = 0
    replacement_by_clone_fallback_count = 0
    rejected_survivor_count = 0
    stuck_walker_count = 0

    def perform_one_nested_sampling_iteration(mcmc_steps_per_replacement):
        """
        Perform one complete iteration and return (discarded_logL, replacement_logL).
        The live-point and dead-point arrays are modified in place.
        """
        nonlocal dead_point_coordinates, dead_point_active_leaf_mask
        nonlocal dead_point_log_likelihood, dead_point_birth_log_likelihood
        nonlocal replacement_by_eryn_count, replacement_by_clone_fallback_count
        nonlocal rejected_survivor_count, stuck_walker_count

        # Step 1: identify the worst live point.
        worst_point_index = int(np.argmin(live_point_true_log_likelihood))
        worst_point_logL = live_point_true_log_likelihood[worst_point_index]

        # Step 2: archive it. All four arrays grow together, by one row. The
        # [None, ...] indexing inserts a leading axis so the shapes align.
        dead_point_coordinates = np.concatenate(
            [dead_point_coordinates,
             live_point_coordinates["gauss"][0, worst_point_index][None, :, :]], axis=0)

        dead_point_active_leaf_mask = np.concatenate(
            [dead_point_active_leaf_mask,
             live_point_active_leaf_mask["gauss"][0, worst_point_index][None, :]], axis=0)

        dead_point_log_likelihood = np.append(
            dead_point_log_likelihood, worst_point_logL)

        dead_point_birth_log_likelihood = np.append(
            dead_point_birth_log_likelihood,
            live_point_birth_log_likelihood[worst_point_index])

        # Step 3: raise the constraint. Written into element [0] of the shared array,
        # so the closure inside eryn_constrained_log_likelihood observes it at once.
        current_logL0_threshold[0] = worst_point_logL

        # Step 4: collect the survivors eligible to seed the replacement search.
        usable_seed_indices = []

        for candidate_index in range(number_of_live_points):

            if candidate_index == worst_point_index:
                continue

            is_usable, candidate_logL = evaluate_live_point_validity(
                live_point_coordinates["gauss"][0, candidate_index],
                live_point_active_leaf_mask["gauss"][0, candidate_index])

            if is_usable:
                # Re-synchronise the stored likelihood with the coordinates it
                # describes. Drift between the two would corrupt the choice of worst.
                live_point_true_log_likelihood[candidate_index] = candidate_logL
                usable_seed_indices.append(candidate_index)
            else:
                # A survivor tied exactly with the threshold, which is what a clone
                # and its parent produce, fails the strict inequality and lands here.
                rejected_survivor_count += 1

        if len(usable_seed_indices) == 0:
            raise RuntimeError(
                f"No usable live point remains at threshold logL = {worst_point_logL:.4f}. "
                f"The live population has collapsed, which is a genuine sampler failure "
                f"rather than a transient one.")

        # Shuffle so that no live point is systematically preferred as a seed.
        np.random.shuffle(usable_seed_indices)

        # Steps 5 to 7: try each seed until one yields a valid replacement.
        for seed_index in usable_seed_indices:

            seed_coords_clean = sanitize_inactive_leaf_coordinates(
                live_point_coordinates["gauss"][0, seed_index],
                live_point_active_leaf_mask["gauss"][0, seed_index])

            seed_mask_clean = live_point_active_leaf_mask["gauss"][0, seed_index].copy()

            # Write the cleaned padding back, so the refill persists.
            live_point_coordinates["gauss"][0, seed_index] = seed_coords_clean

            seed_log_prior_value = compute_log_prior_for_one_point(
                seed_coords_clean[seed_mask_clean])

            if not np.isfinite(seed_log_prior_value):
                continue

            # The two None axes supply the temperature and walker dimensions, giving
            # the required (ntemps, nwalkers, nleaves_max, ndim) shape. log_like is
            # 0.0 by construction rather than by evaluation: the seed passed the
            # threshold check in Step 4, so it lies inside the current shell, where
            # the constrained likelihood is identically zero.
            starting_state = State(
                {"gauss": seed_coords_clean[None, None, :, :].copy()},
                inds={"gauss": seed_mask_clean[None, None, :].copy()},
                log_like=np.array([[0.0]]),
                log_prior=np.array([[seed_log_prior_value]]))

            try:
                final_state = ensemble.run_mcmc(
                    starting_state, mcmc_steps_per_replacement, progress=False)
            except ValueError:
                # Eryn reached a non-finite internal state. Try another seed rather
                # than aborting the run.
                continue

            # The [0, 0] indexing removes the temperature and walker axes.
            replacement_active_mask = final_state.branches["gauss"].inds[0, 0].copy()

            replacement_coordinates = sanitize_inactive_leaf_coordinates(
                final_state.branches["gauss"].coords[0, 0], replacement_active_mask)

            is_usable, replacement_true_logL = evaluate_live_point_validity(
                replacement_coordinates, replacement_active_mask)

            if not is_usable:
                # The walker never left the shell boundary, or finished somewhere
                # invalid.
                continue

            # Detect a walker that rejected every proposal and returned its seed
            # unchanged. This passes every validity check, since the seed already
            # exceeded the threshold, but it is a clone in all but name and biases
            # the compression exactly as the explicit fallback below does.
            replacement_is_identical_to_seed = (
                np.array_equal(replacement_active_mask, seed_mask_clean)
                and np.allclose(replacement_coordinates[replacement_active_mask],
                                seed_coords_clean[seed_mask_clean])
            )
            if replacement_is_identical_to_seed:
                stuck_walker_count += 1

            # The invariant defining a valid Nested Sampling step.
            assert replacement_true_logL > worst_point_logL, (
                f"NESTED SAMPLING CORRECTNESS VIOLATION: replacement logL "
                f"({replacement_true_logL:.6f}) does not strictly exceed the discarded "
                f"threshold ({worst_point_logL:.6f}).")

            # Install the replacement in the slot the discarded point occupied. Its
            # birth threshold is the likelihood it had to beat.
            live_point_coordinates["gauss"][0, worst_point_index] = replacement_coordinates
            live_point_active_leaf_mask["gauss"][0, worst_point_index] = replacement_active_mask
            live_point_true_log_likelihood[worst_point_index] = replacement_true_logL
            live_point_birth_log_likelihood[worst_point_index] = worst_point_logL

            replacement_by_eryn_count += 1
            return worst_point_logL, replacement_true_logL

        # Fallback, reached only when no seed produced a valid replacement.
        #
        # Copying a survivor keeps the run alive and satisfies the invariant, but the
        # copy carries no new information. It is perfectly correlated with its parent,
        # so the compression is overestimated and logZ is biased. Every occurrence is
        # counted so the damage remains measurable.
        clone_index = usable_seed_indices[0]
        clone_logL = live_point_true_log_likelihood[clone_index]

        assert clone_logL > worst_point_logL, (
            f"NESTED SAMPLING CORRECTNESS VIOLATION (clone fallback): clone logL "
            f"({clone_logL:.6f}) does not strictly exceed the discarded threshold "
            f"({worst_point_logL:.6f}).")

        live_point_coordinates["gauss"][0, worst_point_index] = \
            live_point_coordinates["gauss"][0, clone_index].copy()

        live_point_active_leaf_mask["gauss"][0, worst_point_index] = \
            live_point_active_leaf_mask["gauss"][0, clone_index].copy()

        live_point_true_log_likelihood[worst_point_index] = clone_logL
        live_point_birth_log_likelihood[worst_point_index] = worst_point_logL

        replacement_by_clone_fallback_count += 1
        return worst_point_logL, clone_logL

    # 8.7 - One-time repair of the live population

    for _live_point_index in range(number_of_live_points):

        _coords = live_point_coordinates["gauss"][0, _live_point_index]
        _mask = live_point_active_leaf_mask["gauss"][0, _live_point_index]

        if np.isnan(_coords).any():
            _coords = sanitize_inactive_leaf_coordinates(_coords, _mask)

        if _mask.sum() == 0:
            _coords, _mask = enforce_minimum_one_active_leaf(_coords, _mask)

        live_point_coordinates["gauss"][0, _live_point_index] = _coords
        live_point_active_leaf_mask["gauss"][0, _live_point_index] = _mask

        # Recomputed unconditionally, since a repair may have changed the active set.
        live_point_true_log_likelihood[_live_point_index] = true_gaussian_log_likelihood(
            _coords[_mask], time_axis, observed_data, noise_standard_deviation)

    assert np.all(np.isfinite(live_point_true_log_likelihood)), \
        "Some live points still have non-finite log-likelihoods after repair."

    assert live_point_active_leaf_mask["gauss"][0].sum(axis=-1).min() >= 1, \
        "Some live points still have zero active leaves after repair."

    # SECTION 9 - FULL SAMPLER LOOP WITH STOPPING CRITERION

    # 9.1 - Loop configuration

    # [TUNABLE] Length of the constrained walk performed for each replacement.
    #   Too few steps and the walker cannot escape its seed, so the replacement is
    #   correlated with a point already in the population and the clone fallback
    #   fires more often.
    if mcmc_steps_per_replacement is None:
        mcmc_steps_per_replacement = 20 * pulse_ndim

    log_evidence_convergence_tolerance = np.log(evidence_convergence_tolerance)

    # 9.2 - Running evidence bookkeeping
    #
    # Monitoring state only. This estimate exists to decide when to stop; the
    # evidence reported comes from Section 10.

    running_log_evidence_estimate = -np.inf     # log(0): nothing accumulated yet
    running_log_remaining_prior_volume = 0.0    # log(1): the full prior remains
    iteration_count_completed = 0

    log_evidence_history = []
    best_live_logL_history = []
    active_pulse_count_of_best_point_history = []

    # 9.3 - Numerically stable log(exp(a) + exp(b))

    def log_add_exp(log_a, log_b):
        """
        Add two numbers represented by their logarithms, without leaving log space.

        The evidence is a sum of many extremely small likelihood-weighted volumes,
        which would underflow to zero in linear space. The largest exponent is
        factored out before exponentiating, so np.exp never receives a large positive
        argument that would overflow, nor a large negative one that would round to
        zero.
        """
        if log_a == -np.inf:
            return log_b

        if log_b == -np.inf:
            return log_a

        max_val = max(log_a, log_b)

        return max_val + np.log(np.exp(log_a - max_val) + np.exp(log_b - max_val))

    # 9.4 - Main loop

    loop_terminated_cleanly = False
    termination_reason = "reached maximum_allowed_iterations"

    for _ in trange(maximum_allowed_iterations, desc="Nested Sampling"):

        # Step A: discard the worst live point and replace it with one drawn from the
        # constrained prior. All the machinery of Section 8 sits behind this call.
        try:
            discarded_logL, replacement_logL = perform_one_nested_sampling_iteration(
                mcmc_steps_per_replacement=mcmc_steps_per_replacement)

        except RuntimeError as sampling_failure:
            termination_reason = f"live population collapsed: {sampling_failure}"
            break

        iteration_count_completed += 1

        # Step B: shrink the estimated remaining prior volume. Removing the worst of
        # N points leaves, on average, a fraction N/(N+1) of the volume, which in log
        # space is a running sum.
        running_log_remaining_prior_volume += np.log(
            number_of_live_points / (number_of_live_points + 1))

        # Step C: accumulate this dead point's contribution, w = L * X / N.
        log_weight_this_point = (discarded_logL
                                 + running_log_remaining_prior_volume
                                 - np.log(number_of_live_points))

        running_log_evidence_estimate = log_add_exp(running_log_evidence_estimate,
                                                    log_weight_this_point)

        # Step D: diagnostics.
        best_live_logL = live_point_true_log_likelihood.max()
        best_live_index = int(live_point_true_log_likelihood.argmax())
        active_pulse_count_of_best_point = live_point_active_leaf_mask["gauss"][0, best_live_index].sum()

        # The best live likelihood cannot decrease: the best point is never the
        # worst, so it is never discarded, and every point entering the population
        # exceeds the current threshold. A decrease is proof of a bookkeeping fault.
        if len(best_live_logL_history) > 0:
            assert best_live_logL >= best_live_logL_history[-1] - 1e-8, (
                f"Best live logL decreased from {best_live_logL_history[-1]:.6f} to "
                f"{best_live_logL:.6f} at iteration {iteration_count_completed}. This "
                f"is impossible under correct bookkeeping.")

        log_evidence_history.append(running_log_evidence_estimate)
        best_live_logL_history.append(best_live_logL)
        active_pulse_count_of_best_point_history.append(active_pulse_count_of_best_point)

        # Step E: stopping criterion, evaluated in log space. The bound is
        # deliberately optimistic, so satisfying it is a real guarantee that
        # continuing would not change the result materially.
        log_max_remaining_contribution = best_live_logL + running_log_remaining_prior_volume

        log_remaining_contribution_ratio = (log_max_remaining_contribution
                                            - running_log_evidence_estimate)

        if log_remaining_contribution_ratio < log_evidence_convergence_tolerance:
            loop_terminated_cleanly = True
            termination_reason = "evidence convergence criterion satisfied"
            break

    if verbose:
        print(f"    loop finished after {iteration_count_completed} iterations "
              f"({termination_reason})")

    # SECTION 10 - POST-PROCESSING

    # 10.1 - Fold the surviving live points into the archive.
    #
    # They still hold a real share of the remaining prior mass, which is what the
    # stopping criterion asserts to be small rather than zero.

    final_all_coordinates = np.concatenate(
        [dead_point_coordinates, live_point_coordinates["gauss"][0]], axis=0)

    final_all_active_leaf_mask = np.concatenate(
        [dead_point_active_leaf_mask, live_point_active_leaf_mask["gauss"][0]], axis=0)

    final_all_log_likelihood = np.concatenate(
        [dead_point_log_likelihood, live_point_true_log_likelihood], axis=0)

    final_all_birth_log_likelihood = np.concatenate(
        [dead_point_birth_log_likelihood, live_point_birth_log_likelihood], axis=0)

    number_of_points_total = final_all_log_likelihood.shape[0]

    # A point cannot be born above its own likelihood, since it had to exceed the
    # threshold in force at its birth.
    assert np.all(final_all_birth_log_likelihood <= final_all_log_likelihood + 1e-8), \
        "Found a point whose birth threshold exceeds its own logL. Bookkeeping bug."

    # 10.2 - Build the anesthetic table.
    #
    # The row_id column travels through anesthetic's internal sort by logL, so the
    # returned weights can be mapped back onto our original row order exactly.

    model_order_per_point = final_all_active_leaf_mask.sum(axis=-1).astype(float)
    original_row_index_per_point = np.arange(number_of_points_total, dtype=float)

    nested_samples = NestedSamples(
        data=np.column_stack([model_order_per_point, original_row_index_per_point]),
        logL=final_all_log_likelihood,
        logL_birth=final_all_birth_log_likelihood,
        columns=["k", "row_id"],
    )

    # 10.3 - Evidence, uncertainty, and information

    log_evidence_signal_model = float(nested_samples.logZ())
    information_nats = float(nested_samples.D_KL())

    # The returned object is a WeightedSeries rather than an ndarray, and np.std
    # would dispatch to its own weighted implementation, which in some versions
    # rejects the keyword arguments NumPy passes through. Converting first avoids that.
    log_evidence_bootstrap_draws = nested_samples.logZ(nsamples=1000)
    log_evidence_signal_std = float(np.std(np.asarray(log_evidence_bootstrap_draws)))

    # A ratio of evidences, independent of any prior on the hypotheses.
    log_bayes_factor_signal_vs_noise = (log_evidence_signal_model
                                        - log_evidence_no_signal_model)

    # 10.4 - Full posterior over k, including k = 0

    posterior_weights_anesthetic_order = np.asarray(nested_samples.get_weights())
    row_id_anesthetic_order = np.asarray(nested_samples["row_id"]).astype(int)

    # Undo anesthetic's permutation: place each weight at the position its point
    # occupies in final_all_coordinates and model_order_per_point.
    posterior_weights = np.zeros(number_of_points_total)
    posterior_weights[row_id_anesthetic_order] = posterior_weights_anesthetic_order
    posterior_weights = posterior_weights / posterior_weights.sum()

    log_prior_odds_signal_vs_no_signal = np.log(prior_odds_signal_vs_no_signal)
    log_signal_mass_unnormalised = (log_evidence_signal_model
                                    + log_prior_odds_signal_vs_no_signal)
    log_evidence_total = log_add_exp(log_signal_mass_unnormalised,
                                     log_evidence_no_signal_model)

    signal_model_posterior_mass = np.exp(log_signal_mass_unnormalised - log_evidence_total)
    no_signal_posterior_mass = np.exp(log_evidence_no_signal_model - log_evidence_total)

    # k = 0 takes its share directly. Every k >= 1 takes its conditional weight
    # within the signal model, rescaled onto the absolute scale shared with k = 0.
    k_values_full_range = np.arange(0, nleaves_max["gauss"] + 1)
    posterior_probability_per_k = np.zeros_like(k_values_full_range, dtype=float)

    posterior_probability_per_k[0] = no_signal_posterior_mass

    for k_value in range(1, nleaves_max["gauss"] + 1):
        conditional_mass_this_k = posterior_weights[model_order_per_point == k_value].sum()
        posterior_probability_per_k[k_value] = (conditional_mass_this_k
                                                * signal_model_posterior_mass)

    assert np.isclose(posterior_probability_per_k.sum(), 1.0, atol=1e-6), \
        f"Posterior over k does not sum to 1 (got {posterior_probability_per_k.sum():.6f})."

    map_k_estimate = int(k_values_full_range[np.argmax(posterior_probability_per_k)])

    # 10.5 - Flattened parameter posterior
    #
    # Each active leaf inherits its parent point's full weight rather than a share of
    # it, because each leaf represents one inferred pulse in its own right.

    flattened_active_parameters = []
    flattened_parameter_weights = []

    for point_index in range(number_of_points_total):

        active_mask_this_point = final_all_active_leaf_mask[point_index]
        active_params_this_point = final_all_coordinates[point_index][active_mask_this_point]

        if active_params_this_point.shape[0] == 0:
            continue

        flattened_active_parameters.append(active_params_this_point)
        flattened_parameter_weights.append(
            np.full(active_params_this_point.shape[0], posterior_weights[point_index]))

    flattened_active_parameters = np.concatenate(flattened_active_parameters, axis=0)
    flattened_parameter_weights = np.concatenate(flattened_parameter_weights, axis=0)
    flattened_parameter_weights = flattened_parameter_weights / flattened_parameter_weights.sum()

    # Nested Sampling weights are extremely uneven, so the archive is far larger than
    # the number of points actually supporting the posterior. A small effective
    # sample makes the parameter posteriors unreliable however many rows exist.
    effective_sample_size = float(1.0 / np.sum(posterior_weights ** 2))

    total_replacements = replacement_by_eryn_count + replacement_by_clone_fallback_count

    return {
        "logL": final_all_log_likelihood,
        "logL_birth": final_all_birth_log_likelihood,
        "coords": final_all_coordinates,
        "mask": final_all_active_leaf_mask,
        "posterior_weights": posterior_weights,
        "flattened_active_parameters": flattened_active_parameters,
        "flattened_parameter_weights": flattened_parameter_weights,
        "posterior_probability_per_k": posterior_probability_per_k,
        "map_k_estimate": map_k_estimate,
        "log_evidence_signal_model": log_evidence_signal_model,
        "log_evidence_signal_std": log_evidence_signal_std,
        "log_bayes_factor_signal_vs_noise": log_bayes_factor_signal_vs_noise,
        "information_nats": information_nats,
        "effective_sample_size": effective_sample_size,
        "loop_terminated_cleanly": loop_terminated_cleanly,
        "termination_reason": termination_reason,
        "iteration_count_completed": iteration_count_completed,
        "log_evidence_history": np.asarray(log_evidence_history),
        "best_live_logL_history": np.asarray(best_live_logL_history),
        "active_pulse_count_history": np.asarray(active_pulse_count_of_best_point_history),
        "replacement_by_clone_fallback_count": replacement_by_clone_fallback_count,
        "stuck_walker_count": stuck_walker_count,
        "rejected_survivor_count": rejected_survivor_count,
        "total_replacements": total_replacements,
        "gaussian_move_covariance": gaussian_move_covariance,
        "ndims_per_branch": ndims_per_branch,
        "nleaves_max": nleaves_max,
        "nleaves_min": nleaves_min,
        "pulse_ndim": pulse_ndim,
        "number_of_live_points_used": number_of_live_points,
        "mcmc_steps_per_replacement_used": mcmc_steps_per_replacement,
    }
