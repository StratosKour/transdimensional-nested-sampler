"""
Trans-dimensional Nested Sampling, as a black box.

How many Gaussian pulses are in this noisy signal, and where are they? You do
not need to know anything about nested sampling, reversible jump, or Eryn to
use this.

    from transnested import TransNestedSampler

    sampler = TransNestedSampler(my_data, noise_sigma=0.5)
    result = sampler.run()

    print(result)                 # readable summary
    result.model_order            # most probable number of pulses
    result.posterior_over_k       # probability of each possible number
    result.log_evidence           # log Z, with result.log_evidence_error
    result.pulses                 # the recovered pulse parameters
    result.plot()                 # one figure with everything

To try it without data of your own:

    result = TransNestedSampler.demo()

This module contains no algorithm of its own. It is a convenience layer over
experiment.py, which orchestrates nested_sampler.py, baseline_rjmcmc.py and
pulse_model.py. Anything you can do here you can do by calling those directly;
this just removes the need to.

WHAT THIS ASSUMES ABOUT YOUR PROBLEM

The signal model is a sum of Gaussian pulses of COMMON, KNOWN width, on a
one-dimensional axis, in Gaussian noise of known standard deviation:

    d(t) = sum_i  A_i * exp( -(t - mu_i)^2 / (2 c^2) )  +  noise

Each pulse therefore has two free parameters, amplitude and position; the
width c is fixed and supplied by you (pulse_width). The number of pulses is
what is being inferred.

If your problem does not fit that description, this will still run and still
return an answer, but the answer will be "the best sum of Gaussian pulses
describing your data", which may not be what you want. Fitting a different
model would mean changing pulse_model.py; the samplers themselves already
take the likelihood as an argument and do not care what it is.
"""

import numpy as np

try:
    from experiment import run_single_experiment
except ImportError as missing_module:
    raise ImportError(
        "transnested.py cannot run on its own. It is the handle; the machinery "
        "lives in five files that must sit in the SAME FOLDER as this one:\n\n"
        "    transnested.py       (this file, the interface)\n"
        "    experiment.py        (runs one analysis end to end)\n"
        "    nested_sampler.py    (the nested sampling algorithm)\n"
        "    baseline_rjmcmc.py   (the plain RJ-MCMC comparison)\n"
        "    pulse_model.py       (the pulse model, likelihood and priors)\n\n"
        "Copy the whole folder, not just this file, then run again.\n"
        f"(underlying error: {missing_module})") from missing_module


class TransNestedResult:
    """
    The outcome of one analysis. Returned by TransNestedSampler.run(); you do
    not construct this yourself.
    """

    def __init__(self, raw_result, time_axis, observed_data, pulse_width,
                 noise_sigma, archive_path, injected_pulses=None):
        self._raw = raw_result
        self.time = time_axis
        self.data = observed_data
        self.pulse_width = pulse_width
        self.noise_sigma = noise_sigma
        self.archive_path = archive_path
        self.injected_pulses = injected_pulses

    # The headline answers

    @property
    def model_order(self):
        """Most probable number of pulses (the MAP estimate)."""
        return int(self._raw["map_k"])

    @property
    def posterior_over_k(self):
        """
        Probability of each possible number of pulses, as an array indexed by
        k. Entry 0 is the probability that there is no signal at all.
        """
        return np.asarray(self._raw["posterior_per_k"], dtype=float)

    @property
    def model_order_uncertainty(self):
        """
        Posterior mean and standard deviation of the number of pulses. The MAP
        alone hides how confident the answer is: two analyses can share a MAP
        while one is nearly certain and the other nearly flat.
        """
        probabilities = self.posterior_over_k
        k_values = np.arange(len(probabilities))
        mean = float(np.sum(k_values * probabilities))
        variance = float(np.sum((k_values - mean) ** 2 * probabilities))
        return mean, np.sqrt(variance)

    @property
    def log_evidence(self):
        """log Z for the signal model. Only comparable across models fitted to
        the SAME dataset."""
        return float(self._raw["log_evidence"])

    @property
    def log_evidence_error(self):
        return float(self._raw["log_evidence_std"])

    @property
    def log_bayes_factor_vs_noise(self):
        """
        How strongly the data prefer "there is a signal" over "this is pure
        noise". Above about 5 is strong, above 10 decisive.
        """
        return float(self._raw["log_bayes_factor"])

    @property
    def pulses(self):
        """
        The recovered pulses at the most probable model order, as an array of
        shape (model_order, 2): column 0 amplitude, column 1 position.

        These are posterior means over the samples at that model order. The
        full posterior is in the archive if you need it.
        """
        with np.load(self.archive_path, allow_pickle=False) as archive:
            coordinates = archive["coords"]
            active = archive["mask"]
            weights = archive["posterior_weights"]

        at_map_k = active.sum(axis=-1) == self.model_order
        if at_map_k.sum() == 0:
            return np.empty((0, 2))

        selected_coordinates = coordinates[at_map_k]
        selected_active = active[at_map_k]
        selected_weights = weights[at_map_k]
        selected_weights = selected_weights / selected_weights.sum()

        # Pool every active leaf, weighted by its parent point's posterior
        # weight, then sort by position. Leaves have no fixed identity across
        # samples in a trans-dimensional problem, so they are clustered by
        # position rather than matched by index.
        leaf_parameters, leaf_weights = [], []
        for row_index in range(selected_coordinates.shape[0]):
            active_here = selected_coordinates[row_index][selected_active[row_index]]
            leaf_parameters.append(active_here)
            leaf_weights.append(np.full(active_here.shape[0],
                                        selected_weights[row_index]))
        leaf_parameters = np.concatenate(leaf_parameters, axis=0)
        leaf_weights = np.concatenate(leaf_weights)

        # Split the pooled leaves into model_order clusters by position, using
        # weighted quantiles, and take the weighted mean within each.
        order = np.argsort(leaf_parameters[:, 1])
        sorted_parameters = leaf_parameters[order]
        sorted_weights = leaf_weights[order]
        cumulative = np.cumsum(sorted_weights) / sorted_weights.sum()

        recovered = []
        for cluster_index in range(self.model_order):
            lower = cluster_index / self.model_order
            upper = (cluster_index + 1) / self.model_order
            in_cluster = (cumulative > lower) & (cumulative <= upper)
            if in_cluster.sum() == 0:
                continue
            cluster_weights = sorted_weights[in_cluster]
            cluster_weights = cluster_weights / cluster_weights.sum()
            recovered.append([
                float(np.sum(sorted_parameters[in_cluster, 0] * cluster_weights)),
                float(np.sum(sorted_parameters[in_cluster, 1] * cluster_weights)),
            ])
        return np.asarray(recovered)

    @property
    def converged(self):
        """
        Whether the run stopped because the evidence had converged, rather than
        hitting a limit. If False, log_evidence is a LOWER BOUND: the remaining
        live-point contribution was never added. The model order is usually
        still usable; the evidence should be quoted with care.
        """
        return bool(self._raw["converged_cleanly"])

    @property
    def termination_reason(self):
        return str(self._raw["termination_reason"])

    # The comparison baseline, if it was run

    @property
    def rjmcmc_model_order(self):
        """
        The plain RJ-MCMC baseline's answer, or None if it was not run.
        Available only when run(compare_with_rjmcmc=True).
        """
        value = self._raw.get("plain_rjmcmc_map_k")
        return int(value) if value is not None else None

    @property
    def rjmcmc_posterior_over_k(self):
        value = self._raw.get("plain_rjmcmc_posterior_per_k")
        return np.asarray(value, dtype=float) if value is not None else None

    # Convenience

    def as_dict(self):
        """Everything, flat, for writing to a table."""
        return dict(self._raw)

    def compare(self, verbose=True):
        """
        Side-by-side comparison of the Nested Sampler against the plain
        RJ-MCMC baseline, on the same data through the same likelihood.

        Returns a dictionary of the comparison, and prints it unless
        verbose=False. Returns None if the baseline was not run -- pass
        compare_with_rjmcmc=True to run().

        What to look at: whether the two agree on the number of pulses, and
        how much probability each assigns to its own answer. Disagreement by
        one is common and not alarming; the 69-configuration sweep behind this
        code found the two agreeing in 77% of cases and never differing by
        more than one.
        """
        if self.rjmcmc_model_order is None:
            if verbose:
                print("No baseline to compare against. "
                     "Run with compare_with_rjmcmc=True.")
            return None

        nested_posterior = self.posterior_over_k
        baseline_posterior = np.asarray(self.rjmcmc_posterior_over_k, dtype=float)

        # Compared over the k values both actually cover, since the two arrays
        # can differ in length if the baseline never visited the highest k.
        shared_length = min(len(nested_posterior), len(baseline_posterior))
        comparison = {
            "nested_sampler_model_order": self.model_order,
            "rjmcmc_model_order": self.rjmcmc_model_order,
            "agree": self.model_order == self.rjmcmc_model_order,
            "difference": self.model_order - self.rjmcmc_model_order,
            "nested_sampler_posterior": nested_posterior,
            "rjmcmc_posterior": baseline_posterior,
            "nested_sampler_confidence": float(nested_posterior[self.model_order]),
            "rjmcmc_confidence": float(baseline_posterior[self.rjmcmc_model_order]),
            # Total variation distance: half the summed absolute difference.
            # 0 means identical posteriors, 1 means no overlap at all. A single
            # number for "how different are these two answers", which the MAP
            # alone cannot express.
            "total_variation_distance": float(
                0.5 * np.abs(nested_posterior[:shared_length]
                             - baseline_posterior[:shared_length]).sum()),
        }
        if self.injected_pulses is not None:
            comparison["true_model_order"] = len(self.injected_pulses)

        if verbose:
            print("Nested Sampler vs plain RJ-MCMC")
            print("  " + "-" * 52)
            print(f"  {'':<26}{'Nested':>10}{'RJ-MCMC':>12}")
            print(f"  {'number of pulses':<26}"
                 f"{comparison['nested_sampler_model_order']:>10}"
                 f"{comparison['rjmcmc_model_order']:>12}")
            print(f"  {'confidence in that':<26}"
                 f"{comparison['nested_sampler_confidence']:>10.3f}"
                 f"{comparison['rjmcmc_confidence']:>12.3f}")
            print("  " + "-" * 52)

            if "true_model_order" in comparison:
                true_k = comparison["true_model_order"]
                print(f"  true answer: {true_k}"
                     f"   Nested "
                     f"{'correct' if comparison['nested_sampler_model_order'] == true_k else 'off by ' + format(comparison['nested_sampler_model_order'] - true_k, '+d')}"
                     f",  RJ-MCMC "
                     f"{'correct' if comparison['rjmcmc_model_order'] == true_k else 'off by ' + format(comparison['rjmcmc_model_order'] - true_k, '+d')}")

            if comparison["agree"]:
                print("  The two methods AGREE on the number of pulses.")
            else:
                print(f"  They DISAGREE by {abs(comparison['difference'])}. "
                     f"Disagreement by one is common and was seen in 23% of a "
                     f"69-configuration sweep.")
            print(f"  Total variation distance between the two posteriors: "
                 f"{comparison['total_variation_distance']:.3f} "
                 f"(0 = identical, 1 = no overlap)")

        return comparison

    def signal_at(self, time_axis=None):
        """The recovered noiseless signal, evaluated on any axis you like."""
        from pulse_model import combine_all_pulses
        if time_axis is None:
            time_axis = self.time
        return combine_all_pulses(time_axis, self.pulses, self.pulse_width)

    def __repr__(self):
        mean_k, std_k = self.model_order_uncertainty
        lines = [
            "TransNestedResult",
            f"  pulses found        : {self.model_order}"
            f"   (posterior mean {mean_k:.2f} +/- {std_k:.2f})",
            f"  P(that many)        : {self.posterior_over_k[self.model_order]:.3f}",
            f"  log Z               : {self.log_evidence:.2f} +/- {self.log_evidence_error:.2f}",
            f"  log B vs pure noise : {self.log_bayes_factor_vs_noise:.1f}"
            f"   ({'decisive' if self.log_bayes_factor_vs_noise > 10 else 'weak'} signal detection)",
            f"  converged           : {self.converged}",
        ]
        if self.rjmcmc_model_order is not None:
            agreement = ("agrees" if self.rjmcmc_model_order == self.model_order
                         else "DISAGREES")
            lines.append(f"  plain RJ-MCMC says  : {self.rjmcmc_model_order}  ({agreement})")
        if self.injected_pulses is not None:
            lines.append(f"  true answer was     : {len(self.injected_pulses)}"
                        f"   ({'correct' if len(self.injected_pulses) == self.model_order else 'MISSED'})")
        if not self.converged:
            lines.append(f"  NOTE: {self.termination_reason}")
        return "\n".join(lines)

    def plot(self, filename=None, show=True):
        """
        One figure: the data with the recovered signal, and the posterior over
        the number of pulses.
        """
        import matplotlib
        if filename is not None and not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(13, 4.2),
                                    gridspec_kw={"width_ratios": [2, 1]})

        axes[0].plot(self.time, self.data, color="#7f8c8d", linewidth=0.8,
                     alpha=0.8, label="data", zorder=1)
        if self.injected_pulses is not None:
            from pulse_model import combine_all_pulses
            axes[0].plot(self.time,
                         combine_all_pulses(self.time, self.injected_pulses,
                                            self.pulse_width),
                         color="#2e7d32", linewidth=1.8, linestyle="--",
                         label="true signal", zorder=3)
        axes[0].plot(self.time, self.signal_at(), color="#1f4e79", linewidth=1.8,
                     label=f"recovered ({self.model_order} pulses)", zorder=4)
        for amplitude, position in self.pulses:
            axes[0].plot([position], [amplitude], marker="v", color="#1f4e79",
                         markersize=8, zorder=5)
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("amplitude")
        axes[0].set_title("Data and recovered signal")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.25)

        probabilities = self.posterior_over_k
        k_values = np.arange(len(probabilities))
        bar_width = 0.4 if self.rjmcmc_posterior_over_k is not None else 0.7
        offset = bar_width / 2 if self.rjmcmc_posterior_over_k is not None else 0.0
        axes[1].bar(k_values - offset, probabilities, width=bar_width,
                    color="#1f4e79", label="Nested Sampler")
        if self.rjmcmc_posterior_over_k is not None:
            rj = self.rjmcmc_posterior_over_k
            axes[1].bar(np.arange(len(rj)) + offset, rj, width=bar_width,
                        color="#c0392b", label="plain RJ-MCMC")
        if self.injected_pulses is not None:
            axes[1].axvline(len(self.injected_pulses), color="#2e7d32",
                            linestyle="--", linewidth=1.8, label="true")
        axes[1].set_xticks(k_values)
        axes[1].set_xlabel("number of pulses k")
        axes[1].set_ylabel("P(k | data)")
        axes[1].set_title("How many pulses?")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.25)

        figure.tight_layout()
        if filename is not None:
            figure.savefig(filename, dpi=160, bbox_inches="tight")
            print(f"figure written to {filename}")
        if show:
            plt.show()
        return figure


class TransNestedSampler:
    """
    Infer how many Gaussian pulses are in a noisy one-dimensional signal, and
    what their parameters are.

        sampler = TransNestedSampler(data, noise_sigma=0.5)
        result = sampler.run()

    Parameters
    ----------
    data : array
        Your measured signal.
    noise_sigma : float
        Standard deviation of the noise. This is assumed known; the method does
        not fit it.
    time : array, optional
        The axis your data live on. Defaults to evenly spaced points on
        [-1, 1], which only affects the units the positions come back in.
    pulse_width : float
        The common, fixed width c of every pulse. This is the one thing you
        must set sensibly for your problem: it is the scale at which two nearby
        pulses stop being distinguishable.
    max_pulses : int
        The largest number of pulses considered. If the answer comes back equal
        to this, raise it -- the posterior is being truncated.
    amplitude_range : (float, float)
        Prior range for pulse amplitudes. Must contain the real answer.
    min_pulses : int
        Lower limit on k. Defaults to 1. Setting min_pulses equal to
        max_pulses fixes the number of pulses: the sampler then does ordinary
        fixed-dimension parameter estimation at that k, and returns the
        evidence for exactly that model. Running this at several k values and
        comparing the evidences is the reference way to do model selection
        without relying on trans-dimensional moves at all.
        It cannot be 0 -- the no-signal case is evaluated separately and
        analytically, and always appears as P(k=0) in the result.
    n_live : int, optional
        Nested Sampling live points. More is more accurate and slower.
        Defaults to a value scaled from max_pulses (200 at max_pulses = 8).
        This is the main accuracy/speed dial: try 50 for a quick look, 200 for
        a result you intend to quote.
    steps_per_replacement : int, optional
        How many constrained MCMC steps are taken to generate each replacement
        live point. Defaults to 40. Raise it if the run reports a high stuck
        walker fraction, which means the constrained walk is failing to move.
    step_size : float
        Scale of the Gaussian proposal used by that walk. Defaults to 1e-3.
        Too large and almost every proposal is rejected; too small and the
        walk barely moves. Only worth changing if your amplitude and position
        ranges are very different from the defaults.
    seed : int
        Fixes the random draws, so the same call twice gives the same answer.

    The three parameters above configure the Nested Sampler, which is what this
    object IS, so they are set here. The plain RJ-MCMC baseline is an optional
    comparison you switch on at run time, so ITS settings -- rjmcmc_walkers,
    rjmcmc_temperatures, rjmcmc_steps, rjmcmc_burn_in -- are arguments to
    run() instead. See help(TransNestedSampler.run).
    """

    def __init__(self, data, noise_sigma, time=None, pulse_width=0.1,
                 max_pulses=8, min_pulses=1, amplitude_range=(0.0, 5.0),
                 n_live=None, steps_per_replacement=None, step_size=1e-3,
                 seed=0):
        self.data = np.asarray(data, dtype=float)
        self.noise_sigma = float(noise_sigma)
        self.time = np.asarray(time, dtype=float) if time is not None else None
        self.pulse_width = float(pulse_width)
        self.max_pulses = int(max_pulses)
        self.min_pulses = int(min_pulses)
        self.amplitude_range = amplitude_range
        self.n_live = n_live
        self.steps_per_replacement = steps_per_replacement
        self.step_size = float(step_size)
        self.seed = int(seed)
        self._injected_pulses = None  # set only by demo()

        assert 1 <= self.min_pulses <= self.max_pulses, (
            f"min_pulses ({self.min_pulses}) must be at least 1 and no more "
            f"than max_pulses ({self.max_pulses}). min_pulses cannot be 0: the "
            f"no-signal case is always evaluated separately and analytically, "
            f"and appears as P(k=0) in the result.")

        assert self.noise_sigma > 0, "noise_sigma must be positive."
        assert self.data.ndim == 1, "data must be one-dimensional."
        if self.time is not None:
            assert len(self.time) == len(self.data), (
                f"time has {len(self.time)} samples but data has {len(self.data)}.")

    def run(self, compare_with_rjmcmc=True, max_iterations=20000,
            convergence_tolerance=1e-3,
            rjmcmc_walkers=50, rjmcmc_temperatures=10, rjmcmc_steps=20000,
            rjmcmc_burn_in=0.2,
            save_to="transnested_result.npz", verbose=True, **advanced):
        """
        Run the analysis.

        The Nested Sampler's own settings (n_live, steps_per_replacement,
        step_size) belong to the sampler and are set when you construct it.
        The settings below belong to the optional RJ-MCMC comparison, which is
        why they live here instead.

        compare_with_rjmcmc : bool
            Also fit a plain reversible-jump MCMC baseline, for comparison.
            ON BY DEFAULT: the comparison is the point of this code, and a
            result without it cannot be checked against anything. Roughly
            doubles the runtime; pass False when you only want the Nested
            Sampling answer and are in a hurry.
        max_iterations : int
            Safety ceiling for the Nested Sampler. A run that hits this stops
            early and reports converged = False.
        convergence_tolerance : float
            The run stops once the remaining live points could add no more than
            this to the evidence. Smaller means a longer, more accurate run.
            1e-3 is the default and is usually right.
        **advanced
            Anything else is passed straight through to
            experiment.run_single_experiment -- for settings deliberately not
            surfaced here, such as prior_odds_signal_vs_no_signal.

        rjmcmc_temperatures : int
            Rungs in the baseline's parallel-tempering ladder. Leave at 10
            unless you have a reason not to: at a single temperature the
            baseline was measured to fail on this problem at high
            signal-to-noise -- R-hat 2 to 4, some walkers never changing model
            order at all, and a pulse count inflated by three. With ten rungs
            it converges (R-hat 1.000) and agrees with the Nested Sampler in
            77% of a 69-configuration sweep, always within one pulse. Setting
            this to 1 reproduces the failure, which is occasionally useful to
            demonstrate but is not a fair comparison.
        rjmcmc_walkers : int
            Walkers per temperature.
        rjmcmc_steps : int
            Steps each walker takes.
        rjmcmc_burn_in : float
            Fraction of the chain discarded before the posterior is formed.

        save_to : str
            Where to write the full archive. Everything in the result object,
            plus the complete posterior, is recoverable from this file.
        """
        if verbose:
            print(f"Analysing {len(self.data)} samples, "
                 f"noise sigma = {self.noise_sigma}, "
                 f"considering up to {self.max_pulses} pulses"
                 + (f", with RJ-MCMC comparison "
                    f"({rjmcmc_walkers} walkers x {rjmcmc_temperatures} temperatures "
                    f"x {rjmcmc_steps:,} steps)" if compare_with_rjmcmc else "")
                 + "...")

        if compare_with_rjmcmc and rjmcmc_temperatures == 1:
            print("NOTE: rjmcmc_temperatures = 1. The baseline was measured to "
                 "mix badly at a single temperature on this problem; expect it "
                 "to overestimate the pulse count. Use 10 for a fair comparison.")

        raw_result = run_single_experiment(
            noise_standard_deviation=self.noise_sigma,
            observed_data=self.data,
            time_axis=self.time,
            true_injected_pulse_parameters=self._injected_pulses,
            random_seed=self.seed,
            pulse_width_fixed=self.pulse_width,
            amplitude_min=self.amplitude_range[0],
            amplitude_max=self.amplitude_range[1],
            nleaves_max_value=self.max_pulses,
            nleaves_min_value=self.min_pulses,
            evidence_convergence_tolerance=convergence_tolerance,
            number_of_live_points=self.n_live,
            mcmc_steps_per_replacement=self.steps_per_replacement,
            gaussian_move_step_size_scale=self.step_size,
            maximum_allowed_iterations=max_iterations,
            run_plain_rjmcmc_comparison=compare_with_rjmcmc,
            plain_rjmcmc_ntemps=rjmcmc_temperatures,
            plain_rjmcmc_nwalkers=rjmcmc_walkers,
            plain_rjmcmc_nsteps=rjmcmc_steps,
            plain_rjmcmc_burn_in_fraction=rjmcmc_burn_in,
            save_raw_to=save_to,
            verbose=verbose,
            **advanced,
        )

        time_axis = self.time
        if time_axis is None:
            time_axis = np.linspace(-1.0, 1.0, len(self.data))

        result = TransNestedResult(
            raw_result, time_axis, self.data, self.pulse_width,
            self.noise_sigma, save_to, self._injected_pulses)

        if verbose:
            print()
            print(result)

        # Only meaningful when k is free to vary. With min_pulses ==
        # max_pulses the answer equals max_pulses by construction, so the
        # warning would fire on every fixed-k run and mean nothing.
        if result.model_order == self.max_pulses and self.min_pulses != self.max_pulses:
            print(f"\nWARNING: the answer equals max_pulses ({self.max_pulses}). "
                 f"The posterior is probably truncated -- rerun with a larger "
                 f"max_pulses.")

        return result

    @classmethod
    def demo(cls, n_pulses=2, noise_sigma=1.0, seed=0, compare_with_rjmcmc=False,
             quick=True, **kwargs):
        """
        Generate a synthetic dataset with a known answer and analyse it, so the
        method can be tried without supplying data. The true answer is shown in
        the summary and marked on the plot.

            result = TransNestedSampler.demo()
            result.plot()

        The defaults are two well-separated pulses in fairly strong noise,
        chosen because that configuration was measured to converge in about a
        minute. Harder settings (more pulses, less noise) are more interesting
        and take considerably longer -- try n_pulses=4, noise_sigma=0.5 once
        you have a feel for it.

        On expected accuracy: across a 69-configuration sweep this method
        recovered the exact number of pulses in 62% of cases, and was off by
        one in almost all of the rest, always upward. A demo that returns
        k_true + 1 has therefore not failed -- it has done the normal thing.
        This is why the posterior over k matters more than the single most
        probable value, and why both are reported.

        quick : bool
            True (the default) uses a reduced live-point population so the demo
            returns quickly. The answer is real; the evidence is simply less
            precise than a production run gives. Pass quick=False for full
            settings, which takes tens of minutes.
        compare_with_rjmcmc : bool
            Off by default here, because the baseline roughly doubles an
            already long run. Turn it on when you want the comparison, not when
            you just want to see the method work.
        """
        from pulse_model import combine_all_pulses

        pulse_width = kwargs.pop("pulse_width", 0.1)
        time_axis = np.linspace(-1.0, 1.0, 500)

        if n_pulses == 1:
            injected = [[2.0, 0.0]]
        else:
            injected = [[2.0, float(position)]
                        for position in np.linspace(-0.7, 0.7, n_pulses)]

        np.random.seed(seed)
        noiseless = combine_all_pulses(time_axis, injected, pulse_width)
        data = noiseless + noise_sigma * np.random.randn(len(time_axis))

        if quick:
            kwargs.setdefault("n_live", 50)

        # Anything named rjmcmc_* configures the baseline, which is a run()
        # argument, not a sampler property. Separate them before constructing.
        rjmcmc_settings = {name: kwargs.pop(name) for name in list(kwargs)
                           if name.startswith("rjmcmc_")}

        sampler = cls(data, noise_sigma=noise_sigma, time=time_axis,
                      pulse_width=pulse_width, seed=seed, **kwargs)
        sampler._injected_pulses = injected

        print(f"Demo: {n_pulses} pulses injected at "
             f"{[round(p[1], 3) for p in injected]}, noise sigma {noise_sigma}. "
             f"The sampler is not told any of this.")
        if quick:
            print("Quick mode: reduced live points for speed. "
                 "Pass quick=False for production settings.")
        print()

        return sampler.run(compare_with_rjmcmc=compare_with_rjmcmc,
                           **rjmcmc_settings)


if __name__ == "__main__":
    # python transnested.py  -- runs the demo end to end
    result = TransNestedSampler.demo()
    result.plot(filename="transnested_demo.png", show=False)
