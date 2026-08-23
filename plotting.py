"""
Diagnostic plots for a TransNestedResult.

Every function here takes a TransNestedResult -- the object TransNestedSampler.run()
already returns -- and nothing else. The archive .npz file that result.archive_path
points to IS still read, because the iteration histories, the raw pulse posterior,
and the RJ-MCMC comparison data live there and nowhere else. That reading happens
entirely INSIDE these functions: a notebook cell calling plot_diagnostics(result)
never sees a file path, never calls np.load, and never has to know the archive
exists at all.

This is a thin layer on top of the tested pipeline (transnested.py, pulse_model.py),
not a second implementation of anything. No likelihood, no prior, no sampling logic
lives here -- only matplotlib and corner calls over arrays the archive already holds.

Every function calls plt.close() on its figure(s) after plt.show(), so a bare
call as the last line of a notebook cell (e.g. plot_diagnostics(result)) never
displays the figure twice -- once from show(), once from Jupyter's own
auto-display of the returned Figure object.
"""

import numpy as np
import matplotlib.pyplot as plt
import corner


# 1 - THE THREE-PANEL RUN DIAGNOSTIC (evidence, likelihood climb, model order)

def plot_diagnostics(result, figsize=(14, 4)):
    """
    Evidence convergence, likelihood climb, and model order over Nested Sampling
    iterations -- the same three panels used throughout the project's own
    development to confirm a run behaved correctly, not just that it finished.

    Requires the archive at result.archive_path, opened and closed inside this
    function only.
    """
    with np.load(result.archive_path, allow_pickle=False) as archive:
        log_evidence_history = archive["log_evidence_history"]
        best_live_logL_history = archive["best_live_logL_history"]
        active_pulse_count_history = archive["active_pulse_count_history"]

    nleaves_max = len(result.posterior_over_k) - 1

    figure, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].plot(log_evidence_history)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Running log Z")
    axes[0].set_title("Evidence convergence")

    axes[1].plot(best_live_logL_history)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Best live logL")
    axes[1].set_title("Likelihood climb")
    if result.injected_pulses is not None:
        log_likelihood_at_truth = _log_likelihood_at_injected_truth(result)
        axes[1].axhline(log_likelihood_at_truth, color="crimson", linestyle="--",
                        label="logL at injected truth")
        axes[1].legend(fontsize=8)

    axes[2].plot(active_pulse_count_history)
    axes[2].axhline(nleaves_max, color="grey", linestyle=":", label="nleaves_max")
    if result.injected_pulses is not None:
        axes[2].axhline(len(result.injected_pulses), color="crimson", linestyle="--",
                        label="true k")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Active pulses (best live point)")
    axes[2].set_title("Model order over time")
    axes[2].legend(fontsize=8)

    figure.tight_layout()
    plt.show()
    plt.close(figure)
    return figure


# 2 - POSTERIOR ON MODEL ORDER

def plot_posterior(result, figsize=(7, 4)):
    """
    Bar chart of P(k | data). Needs nothing from the archive -- posterior_over_k
    is already a property of the result object -- but is provided here so every
    figure in a notebook comes from one consistent module.
    """
    probabilities = result.posterior_over_k
    k_values = np.arange(len(probabilities))

    true_k = len(result.injected_pulses) if result.injected_pulses is not None else None
    bar_colors = ["crimson" if k == true_k else "steelblue" for k in k_values]

    figure = plt.figure(figsize=figsize)
    plt.bar(k_values, probabilities, color=bar_colors)
    plt.xlabel("Number of active pulses, k")
    plt.ylabel("Posterior probability")
    plt.title("Posterior on model order (number of pulses)")
    plt.xticks(k_values)
    plt.tight_layout()
    plt.show()
    plt.close(figure)
    return figure


# 3 - CORNER PLOT OF THE RECOVERED PULSE PARAMETERS

def plot_corner(result, figsize=None):
    """
    Weighted corner plot of every active leaf across the archive, pooled the same
    way result.pulses pools them: each leaf inherits its parent point's full
    Nested Sampling weight. Overlays the injected truth when it is known.

    Requires the archive, opened and closed inside this function only.
    """
    with np.load(result.archive_path, allow_pickle=False) as archive:
        flattened_active_parameters = archive["flattened_active_parameters"]
        flattened_parameter_weights = archive["flattened_parameter_weights"]

    corner_figure = corner.corner(
        flattened_active_parameters,
        weights=flattened_parameter_weights,
        labels=["Amplitude", "Mean position"],
        show_titles=True,
    )

    if result.injected_pulses is not None:
        corner_axes = np.array(corner_figure.axes).reshape((2, 2))
        scatter_ax = corner_axes[1, 0]
        true_params_array = np.asarray(result.injected_pulses)
        scatter_ax.scatter(true_params_array[:, 0], true_params_array[:, 1],
                           color="crimson", marker="*", s=120, zorder=10,
                           label="Injected truth")
        scatter_ax.legend(fontsize=8)

    plt.show()
    plt.close(corner_figure)
    return corner_figure


# 4 - NESTED SAMPLER vs PLAIN RJ-MCMC COMPARISON

def plot_rjmcmc_comparison(result, figsize=(11, 4)):
    """
    Three views of the comparison against the plain RJ-MCMC baseline: the
    model-order posteriors side by side, an overlaid corner plot of the pulse
    parameters, and the per-walker model-order trace of the RJ-MCMC cold chain
    (useful for spotting stuck walkers).

    Returns None and prints a message if compare_with_rjmcmc=False was used when
    result.run() was called, since there is nothing to compare against.

    Requires the archive, opened and closed inside this function only.
    """
    if result.rjmcmc_model_order is None:
        print("No RJ-MCMC baseline to compare against. "
             "Re-run with compare_with_rjmcmc=True.")
        return None

    with np.load(result.archive_path, allow_pickle=False) as archive:
        plain_rjmcmc_flat_params = archive["plain_rjmcmc_flat_params"]
        plain_rjmcmc_model_order_trace = archive["plain_rjmcmc_model_order_trace"]
        flattened_active_parameters = archive["flattened_active_parameters"]
        flattened_parameter_weights = archive["flattened_parameter_weights"]

    true_k = len(result.injected_pulses) if result.injected_pulses is not None else None
    k_values = np.arange(len(result.posterior_over_k))

    # 4.1 - "How many pulses?" -- both methods on one set of axes, grouped bars,
    # same convention as plot_overview's third panel

    rjmcmc_posterior = result.rjmcmc_posterior_over_k
    figure_bars = plt.figure(figsize=(7, 4.2))

    bar_width = 0.4
    offset = bar_width / 2
    plt.bar(k_values - offset, result.posterior_over_k, width=bar_width,
           color="#1f4e79", label="Nested Sampler")
    plt.bar(np.arange(len(rjmcmc_posterior)) + offset, rjmcmc_posterior,
           width=bar_width, color="#c0392b", label="plain RJ-MCMC")
    if true_k is not None:
        plt.axvline(true_k, color="#2e7d32", linestyle="--", linewidth=1.8, label="true")
    plt.xticks(k_values)
    plt.xlabel("number of pulses k")
    plt.ylabel("P(k | data)")
    plt.title("How many pulses?")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()
    plt.close(figure_bars)

    # 4.2 - overlaid corner plot

    figure_corner = corner.corner(
        plain_rjmcmc_flat_params, color="grey",
        labels=["Amplitude", "Mean position"], hist_kwargs={"density": True})
    corner.corner(
        flattened_active_parameters, weights=flattened_parameter_weights,
        fig=figure_corner, color="crimson", hist_kwargs={"density": True})
    plt.suptitle("Grey: plain RJ-MCMC | Crimson: trans-dimensional Nested Sampler")
    plt.show()
    plt.close(figure_corner)

    # 4.3 - per-walker model-order trace, to catch stuck walkers by eye

    figure_trace = plt.figure(figsize=(8, 4))
    plt.plot(plain_rjmcmc_model_order_trace, alpha=0.3, lw=0.5)
    if true_k is not None:
        plt.axhline(true_k, color="crimson", linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("k per walker")
    plt.title("RJ-MCMC per-walker model order trace")
    plt.show()
    plt.close(figure_trace)

    return figure_bars, figure_corner, figure_trace


# 5 - ONE-FIGURE OVERVIEW (grey/crimson overlay, for saving to disk)

def plot_overview(result, filename=None, show=True, figsize=(15, 4.2)):
    """
    One figure, meant to replace TransNestedResult.plot() as the summary saved
    to disk by analysis.py -- without editing transnested.py itself.

    When the RJ-MCMC baseline was run: amplitude and position posteriors
    overlaid (grey = plain RJ-MCMC, crimson = Nested Sampler), plus the
    posterior over k, in the same grey/crimson convention as
    plot_rjmcmc_comparison's corner plot above.

    When no baseline was run (compare_with_rjmcmc=False): falls back to the
    data-and-recovered-signal view, since there is nothing grey to overlay.
    This fallback reproduces TransNestedResult.plot() exactly, so nothing
    breaks for a result that was never asked to run the comparison.

    filename, show : same meaning as TransNestedResult.plot() -- pass
    filename=... to save, show=False when running unattended (e.g. under
    nohup, with no display to open a window on).
    """
    if filename is not None and not show:
        import matplotlib
        matplotlib.use("Agg")

    has_comparison = result.rjmcmc_posterior_over_k is not None

    if has_comparison:
        figure, axes = plt.subplots(1, 3, figsize=figsize,
                                    gridspec_kw={"width_ratios": [1, 1, 1.2]})

        with np.load(result.archive_path, allow_pickle=False) as archive:
            plain_rjmcmc_flat_params = archive["plain_rjmcmc_flat_params"]
            flattened_active_parameters = archive["flattened_active_parameters"]
            flattened_parameter_weights = archive["flattened_parameter_weights"]

        axes[0].hist(plain_rjmcmc_flat_params[:, 0], bins=40, density=True,
                    histtype="step", color="grey", linewidth=1.5,
                    label="plain RJ-MCMC")
        axes[0].hist(flattened_active_parameters[:, 0], bins=40, density=True,
                    weights=flattened_parameter_weights, histtype="step",
                    color="crimson", linewidth=1.5, label="Nested Sampler")
        axes[0].set_xlabel("Amplitude")
        axes[0].set_ylabel("density")
        axes[0].set_title("Amplitude posterior")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.25)

        axes[1].hist(plain_rjmcmc_flat_params[:, 1], bins=40, density=True,
                    histtype="step", color="grey", linewidth=1.5,
                    label="plain RJ-MCMC")
        axes[1].hist(flattened_active_parameters[:, 1], bins=40, density=True,
                    weights=flattened_parameter_weights, histtype="step",
                    color="crimson", linewidth=1.5, label="Nested Sampler")
        axes[1].set_xlabel("Mean position")
        axes[1].set_title("Position posterior")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.25)

        posterior_axis = axes[2]
    else:
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.2),
                                    gridspec_kw={"width_ratios": [2, 1]})

        axes[0].plot(result.time, result.data, color="#7f8c8d", linewidth=0.8,
                    alpha=0.8, label="data", zorder=1)
        if result.injected_pulses is not None:
            from pulse_model import combine_all_pulses
            axes[0].plot(result.time,
                        combine_all_pulses(result.time, result.injected_pulses,
                                           result.pulse_width),
                        color="#2e7d32", linewidth=1.8, linestyle="--",
                        label="true signal", zorder=3)
        axes[0].plot(result.time, result.signal_at(), color="#1f4e79", linewidth=1.8,
                    label=f"recovered ({result.model_order} pulses)", zorder=4)
        for amplitude, position in result.pulses:
            axes[0].plot([position], [amplitude], marker="v", color="#1f4e79",
                        markersize=8, zorder=5)
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("amplitude")
        axes[0].set_title("Data and recovered signal")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.25)

        posterior_axis = axes[1]

    probabilities = result.posterior_over_k
    k_values = np.arange(len(probabilities))
    bar_width = 0.4 if has_comparison else 0.7
    offset = bar_width / 2 if has_comparison else 0.0
    posterior_axis.bar(k_values - offset, probabilities, width=bar_width,
                       color="#1f4e79", label="Nested Sampler")
    if has_comparison:
        rj = result.rjmcmc_posterior_over_k
        posterior_axis.bar(np.arange(len(rj)) + offset, rj, width=bar_width,
                           color="#c0392b", label="plain RJ-MCMC")
    if result.injected_pulses is not None:
        posterior_axis.axvline(len(result.injected_pulses), color="#2e7d32",
                               linestyle="--", linewidth=1.8, label="true")
    posterior_axis.set_xticks(k_values)
    posterior_axis.set_xlabel("number of pulses k")
    posterior_axis.set_ylabel("P(k | data)")
    posterior_axis.set_title("How many pulses?")
    posterior_axis.legend(fontsize=8)
    posterior_axis.grid(alpha=0.25)

    figure.tight_layout()
    if filename is not None:
        figure.savefig(filename, dpi=160, bbox_inches="tight")
        print(f"figure written to {filename}")
    if show:
        plt.show()
    plt.close(figure)
    return figure


# 6 - INTERNAL HELPER

def _log_likelihood_at_injected_truth(result):
    """
    Recompute logL at the injected truth, for the reference line in
    plot_diagnostics. Uses the SAME likelihood the sampler used, imported from
    pulse_model rather than redefined here, so this module never becomes a second
    place the formula could drift.
    """
    from pulse_model import true_gaussian_log_likelihood

    return true_gaussian_log_likelihood(
        np.asarray(result.injected_pulses), result.time, result.data,
        result.noise_sigma, result.pulse_width)