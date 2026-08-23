"""
Run one TransNested analysis on your own data.

Edit the SETTINGS block below, then run:

    python analysis.py                    # in the foreground
    nohup python analysis.py > run.log &   # unattended, survives closing the terminal

Progress and the final result print to the terminal (or to run.log, under
nohup). A results archive and a summary figure are written to disk.



EXAMPLE to generate a synthetic dataset, save it to disk, and then run the analysis on it.
Remove the (""" """). Else you can use your own data by setting DATA_FILE to the path of your 1D array.s

number_of_time_samples = 500
time_axis_lower_bound = -1.0
time_axis_upper_bound = 1.0
noise_sigma = 0.5
pulse_width = 0.1
seed = 42

injected_pulses = [
    [3.0, -0.4],  
    [2.5,  0.5]   
]

time_axis = build_time_axis(number_of_time_samples, time_axis_lower_bound, time_axis_upper_bound)

dataset = generate_synthetic_dataset(
    time_axis, 
    injected_pulses, 
    noise_sigma, 
    pulse_width, 
    random_seed=seed
)

my_data = dataset["observed_data"]

np.save("my_data.npy", my_data)
"""



import numpy as np
from pulse_model import build_time_axis, generate_synthetic_dataset
from plotting import plot_overview
from transnested import TransNestedSampler

# Your data
DATA_FILE = "my_data.npy"       # path to your own 1D data array
NOISE_SIGMA = 0.5               # noise standard deviation of your data

# The model
PULSE_WIDTH = 0.1                # FIXED width shared by every pulse.
                                 # Set to roughly the width of the features
                                 # you expect -- it is not fitted.
MAX_PULSES = 8                   # largest number of pulses considered
MIN_PULSES = 1                   # set equal to MAX_PULSES to lock the count
AMPLITUDE_RANGE = (0.0, 5.0)     # prior range for pulse amplitudes

# The nested sampler
N_LIVE = 200                     # accuracy/speed dial. 50 for a quick look.
STEPS_PER_REPLACEMENT = None     # None = automatic (40)
STEP_SIZE = 1e-3
SAMPLER_SEED = 0

# The run
MAX_ITERATIONS = 20000
CONVERGENCE_TOLERANCE = 1e-3

# The rjmcmc comparison (optional, on by default)
COMPARE_WITH_RJMCMC = True
RJMCMC_TEMPERATURES = 10
RJMCMC_WALKERS = 50
RJMCMC_STEPS = 20000
RJMCMC_BURN_IN = 0.2

# Output
SAVE_ARCHIVE_TO = "my_result.npz"
SAVE_FIGURE_TO = "my_result.png"
VERBOSE = True


def main():
    print(f"Loading data from {DATA_FILE} ...")
    my_data = np.load(DATA_FILE)
    print(f"  {len(my_data)} samples loaded.\n")

    sampler = TransNestedSampler(
        my_data,
        noise_sigma=NOISE_SIGMA,
        pulse_width=PULSE_WIDTH,
        max_pulses=MAX_PULSES,
        min_pulses=MIN_PULSES,
        amplitude_range=AMPLITUDE_RANGE,
        n_live=N_LIVE,
        steps_per_replacement=STEPS_PER_REPLACEMENT,
        step_size=STEP_SIZE,
        seed=SAMPLER_SEED,
    )

    result = sampler.run(
        compare_with_rjmcmc=COMPARE_WITH_RJMCMC,
        max_iterations=MAX_ITERATIONS,
        convergence_tolerance=CONVERGENCE_TOLERANCE,
        rjmcmc_temperatures=RJMCMC_TEMPERATURES,
        rjmcmc_walkers=RJMCMC_WALKERS,
        rjmcmc_steps=RJMCMC_STEPS,
        rjmcmc_burn_in=RJMCMC_BURN_IN,
        save_to=SAVE_ARCHIVE_TO,
        verbose=VERBOSE,
    )

    print()
    print(result)

    if COMPARE_WITH_RJMCMC:
        print()
        result.compare()

    # show=False: this must not try to open a window, since nohup has no
    # display to open one on. The figure is written to disk instead.
    plot_overview(result, filename=SAVE_FIGURE_TO, show=False)  
    print(f"\nFigure written to {SAVE_FIGURE_TO}")
    print(f"Full archive written to {SAVE_ARCHIVE_TO}")


if __name__ == "__main__":
    main()