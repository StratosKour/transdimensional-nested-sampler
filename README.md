# TransNested

**How many pulses are hiding in your noisy signal — and where?**

TransNested is a small Python library for a specific problem: your data is a
noisy 1D signal, you believe it contains some number of Gaussian pulses, and
you don't know how many. TransNested finds out — both the number and their
positions and amplitudes — and gives you a full probability for each possible
count, not just a single guess.

```python
from transnested import TransNestedSampler

result = TransNestedSampler.demo()     # try it with no data of your own
print(result)
```

---

## The idea in one paragraph

Under the hood this runs two different samplers on your data and compares
them. The main one is a **trans-dimensional Nested Sampler**: a custom outer
loop that treats "how many pulses" as an unknown to be sampled, using Eryn's
reversible-jump MCMC internally to explore each step. The second is a **plain
RJ-MCMC fit**, included as a built-in sanity check. Both are handed the exact
same likelihood function, so they can never quietly end up solving two
different problems — which matters, because that comparison is the whole
point of this project.

---

## Install

```bash
pip install -r requirements.txt
```

Needs `numpy`, `scipy`, `matplotlib`, `eryn`, `anesthetic`, `corner`.

All files must stay in **one folder**. `transnested.py` imports the rest by
plain name; if a file is missing it says so by name rather than failing with
a confusing error.

---

## What your data must look like

Before anything else, three things:

**The data** is a one-dimensional NumPy array of measurements, evenly spaced.
Nothing else — no headers, no time column. If you have it in a file, load it
first: `np.load("my_data.npy")` or `np.loadtxt("my_data.txt")`.

**The noise level** (`noise_sigma`) is the standard deviation of the noise in
your data, and it is **assumed known**. TransNested does not fit it. If you
get this badly wrong, everything downstream is wrong: too low and the sampler
invents pulses to explain noise, too high and it misses real ones.

**The pulse width** (`pulse_width`) is the common, fixed width of every pulse,
and it is **not fitted** either. Set it to roughly the width of the features
you expect. This is the one setting you must think about: it is the scale at
which two nearby pulses stop being distinguishable.

The time axis is optional. Leave it out and the code assumes evenly spaced
points on `[-1, 1]`, which only affects the units the recovered positions come
back in. Supply your own with `time=...` if you want physical units.

---

## Three ways to use it

The three share one engine. Whichever you pick, the numbers are the same.

| | file | best for |
|---|---|---|
| 1 | `transnested.py` | first look, or calling from your own Python |
| 2 | `analysis.py` | one run on your own data, unattended |
| 3 | `TransNested_Analysis.ipynb` | exploring interactively, with plots |

---

### 1. `transnested.py` — the library

**Start here. Run it with no arguments:**

```bash
python transnested.py
```

This builds a synthetic signal with a known answer, analyses it *without being
told that answer*, prints a summary, and writes `transnested_demo.png`. About
a minute. Nothing to configure, no data needed — it exists so you can confirm
the install works and see the output format before committing to anything.

**The same demo from Python, with control over the difficulty:**

```python
from transnested import TransNestedSampler

result = TransNestedSampler.demo(n_pulses=2, noise_sigma=1.0, quick=True)
```

- `n_pulses` — how many pulses to inject (spread evenly across the window)
- `noise_sigma` — larger means a harder problem
- `quick=True` (default) uses fewer live points so it returns fast; the answer
  is real, the evidence is simply less precise. `quick=False` for full settings
- `compare_with_rjmcmc=False` (default here) — turn it on to also run the
  baseline, which roughly doubles the runtime

Once you have seen it work, the demo has served its purpose. Everything below
is the real interface.

**On your own data:**

```python
import numpy as np
from transnested import TransNestedSampler

my_data = np.load("my_data.npy")

sampler = TransNestedSampler(my_data, noise_sigma=0.5, pulse_width=0.1)
result = sampler.run()

print(result)
```

Only `data` and `noise_sigma` are required. The full constructor:

```python
TransNestedSampler(
    data,                        # your 1D array
    noise_sigma,                 # noise standard deviation, assumed known
    time=None,                   # your own axis; default is linspace(-1, 1, len(data))
    pulse_width=0.1,             # fixed width shared by every pulse
    max_pulses=8,                # largest count considered
    min_pulses=1,                # set equal to max_pulses to lock the count
    amplitude_range=(0.0, 5.0),  # prior range for amplitudes; must contain the answer
    n_live=200,                  # accuracy/speed dial; 50 for a quick look
    steps_per_replacement=None,  # None = automatic (40)
    step_size=1e-3,              # proposal scale
    seed=0,                      # fixes every random draw
)
```

These configure the Nested Sampler, which is what the object *is*. The
RJ-MCMC comparison is optional and belongs to the run, so its settings go to
`run()` instead:

```python
result = sampler.run(
    compare_with_rjmcmc=True,        # on by default
    rjmcmc_temperatures=10,          # leave at 10 — see below
    rjmcmc_walkers=50,
    rjmcmc_steps=20000,
    rjmcmc_burn_in=0.2,
    max_iterations=20000,            # safety ceiling for the Nested Sampler
    convergence_tolerance=1e-3,
    save_to="transnested_result.npz",
    verbose=True,
)
```

**Locking the pulse count.** Setting `min_pulses = max_pulses = k` turns off
the trans-dimensional moves entirely: the sampler does ordinary fixed-dimension
parameter estimation at that `k` and returns the evidence for exactly that
model. Running this at several values of `k` and comparing the evidences is the
reference way to do model selection without relying on reversible jump at all.

`min_pulses` cannot be 0. The no-signal case is evaluated separately and
analytically, and always appears as `P(k=0)` in the result.

---

### 2. `analysis.py` — one run, unattended

A single script with every setting in a block at the top. Edit that block,
run it, come back to a figure and an archive on disk.

**Step 1 — put your data in a `.npy` file.**

If you already have data:

```python
import numpy as np
np.save("my_data.npy", my_array)
```

If you want synthetic data to try it on, `analysis.py` carries a ready-made
block for exactly that, commented out near the top. Uncomment it, set the
injection you want, and run the file once — it writes `my_data.npy` and the
analysis proceeds on it.

**Step 2 — edit the settings block.** Every value is a plain constant with a
comment. The ones worth attention:

```python
DATA_FILE = "my_data.npy"        # path to your 1D array
NOISE_SIGMA = 0.5                # noise standard deviation of your data
PULSE_WIDTH = 0.1                # fixed width, set to match your features
MAX_PULSES = 8                   # raise if the answer hits this ceiling
N_LIVE = 200                     # 50 for a quick look, 200 to quote
COMPARE_WITH_RJMCMC = True       # roughly doubles the runtime
SAVE_ARCHIVE_TO = "my_result.npz"
SAVE_FIGURE_TO = "my_result.png"
```

**Step 3 — run it.**

```bash
python analysis.py                                # in the foreground
nohup python analysis.py > run.log 2>&1 &          # unattended
tail -f run.log                                    # watch progress; Ctrl+C stops watching, not the run
```

Under `nohup` the run survives closing the terminal or disconnecting. Expect
roughly 15–20 minutes at the default settings with the comparison on.

**What you get back:** the summary and the comparison printed to the terminal
(or to `run.log`), a three-panel figure at `SAVE_FIGURE_TO`, and the full
archive at `SAVE_ARCHIVE_TO`. Given the archive, every figure can be redrawn
without re-running anything.

---

### 3. `TransNested_Analysis.ipynb` — interactive

Eight sections. Settings live in one cell; everything below reads from it.

| section | what it does | cost |
|---|---|---|
| 1 | imports | instant |
| 2 | **all settings** — the only cell you edit | instant |
| 3 | builds the data and runs both samplers | **minutes to tens of minutes** |
| 4 | printed summary and comparison | instant |
| 5 | run diagnostics: evidence convergence, likelihood climb, model order | instant |
| 6 | posterior over the number of pulses | instant |
| 7 | corner plot of the recovered pulses | instant |
| 8 | comparison against plain RJ-MCMC | instant |

Section 3 is the only expensive one. Sections 4–8 read the result that
Section 3 already produced, so you can re-run them freely — to adjust a plot,
say — without repeating the sampling.

Unlike `analysis.py`, the notebook **generates its own synthetic data** from
the injection you specify in Section 2, so there is no file to prepare. To
analyse your own data instead, replace the `generate_synthetic_dataset` call
in Section 3 with your own array and drop the `_injected_pulses` line — with
no known truth, the plots simply omit the reference markers.

**Two practical notes.** If you edit any `.py` file while the notebook is
open, restart the kernel before re-running: Python does not reload an
already-imported module, so the old version stays in memory. And end plotting
cells with a semicolon — `plot_posterior(result);` — otherwise Jupyter
displays the returned figure a second time.

---

## Reading a result

```python
result.model_order                # most probable number of pulses
result.posterior_over_k           # probability of every count, as an array
result.model_order_uncertainty    # posterior mean and standard deviation of k
result.pulses                     # recovered [amplitude, position] per pulse
result.log_evidence               # log Z, with result.log_evidence_error
result.log_bayes_factor_vs_noise  # signal vs pure noise; >10 is decisive
result.converged                  # True if the run finished properly
result.termination_reason         # why it stopped
result.rjmcmc_model_order         # the baseline's answer, or None
result.compare()                  # prints and returns the full comparison
result.signal_at()                # the recovered noiseless signal
result.as_dict()                  # every number above, in one dictionary
```

**Two things worth knowing before you trust an answer:**

- Use `posterior_over_k`, not just `model_order`. A run that is 95% sure of
  its answer and one that is 35% sure report the same `model_order` — the
  posterior is what tells them apart.
- If `result.converged` is `False`, the run stopped early rather than
  settling, and `log_evidence` is a lower bound rather than a final number.
<img width="1471" height="606" alt="fig02_model_order_error" src="https://github.com/user-attachments/assets/8abc1d3f-953b-4517-814d-7bdfe1c07aef" />

---

## What's in the folder

| file | what it does |
|---|---|
| `transnested.py` | **the interface — the only file you need to read** |
| `analysis.py` | one run on your own data, edit-and-go |
| `TransNested_Analysis.ipynb` | the same thing interactively, with plots |
| `experiment.py` | wires the pieces together for one analysis |
| `nested_sampler.py` | the Nested Sampling algorithm |
| `baseline_rjmcmc.py` | the RJ-MCMC comparison |
| `pulse_model.py` | the pulse shape, likelihood, and priors — defined once, here |
| `plotting.py` | every figure the project draws |
| `diagnostics.py` | convergence checks |
| `test_pulse_model.py` | confirms the model and likelihood are correct |
| `test_diagnostics.py` | confirms the convergence checks are correct |
| `final_check.py` | one full-scale run with 18 built-in sanity checks |

To confirm the install works:

```bash
python test_pulse_model.py     # seconds
python test_diagnostics.py     # seconds
python final_check.py          # one full production run, ~15-70 min
```

---

## What it found, on real testing

Across 69 test configurations (varying noise level, true pulse count, and how
close together the pulses were):

- Exact pulse count recovered: **62%** of the time (Nested Sampler),
  **65%** (RJ-MCMC, with the temperature ladder)
- The two methods **agreed with each other 77%** of the time, and were never
  more than one pulse apart when they disagreed

So this isn't a claim that the Nested Sampler is more accurate — the two are
close. The finding is that it reaches the same answer for roughly a fortieth
of the likelihood evaluations: about 245,000 against 10,000,000.

The one place a real difference showed up: when two pulses were very close
together (about half a pulse-width apart), the Nested Sampler told them apart
correctly every time, and the RJ-MCMC baseline merged them into one. That is
the only situation, across all 69 tests, where the baseline under-counted.

<img width="1461" height="688" alt="fig01_model_order_confusion" src="https://github.com/user-attachments/assets/43661390-93a2-4954-b1d8-cd2c4319b1b8" />

![Uploading fig02_model_order_error.png…]()


---

## Limits, honestly

- The pulse width is **fixed**, not fitted. Pulses of varying width are not
  supported.
- The noise level is **assumed known**.
- The count is capped at `max_pulses`. If the answer piles up at that limit,
  raise it — you get a warning when this happens.
- The method tends to slightly **over-count**, not under-count, when wrong.
- A small fraction of very-high-signal runs stop before fully converging;
  `result.converged` always tells you when.
- Replacement points come from a finite MCMC walk rather than an exact draw
  from the constrained prior, so they are not perfectly independent. Measured
  against an analytically known evidence at k = 1, the scatter across seeds
  exceeded the quoted error bar by up to a factor of three. Absolute `log Z`
  should therefore be read with that in mind; ratios between models, which is
  what the pulse count depends on, are far less affected.

---

## Changing the model

Everything about the signal shape — the pulse formula, the likelihood, the
priors — lives in `pulse_model.py`, and only there. Both samplers read from
it, so if you ever need a different kind of pulse, that is the one file to
change.
