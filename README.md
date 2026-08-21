# TransNested

**How many pulses are hiding in your noisy signal — and where?**

TransNested is a small Python library for a specific problem: your data is a
noisy 1D signal, you believe it contains some number of Gaussian pulses, and
you don't know how many. TransNested finds out — both the number and their
positions and amplitudes — and gives you a full probability for each possible
count, not just a single guess.

It is a library, not a notebook or a script. Use it from a terminal, a plain
Python file, a Jupyter notebook, or inside a larger pipeline — it behaves
exactly the same way everywhere.

```python
from transnested import TransNestedSampler

result = TransNestedSampler.demo()     # try it with no data of your own
print(result)
result.plot()
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

---

## Three ways to use it

**1. See it work, no data needed**

```bash
python transnested.py
```

Makes a synthetic signal with a known answer, analyses it without being told
that answer, and saves a summary plot. About a minute.

**2. Interactively, in a notebook**

Open `TransNested_Control_Panel.ipynb`. All the settings live in one cell;
edit it and run the cells below to get results and plots. This is the easiest
way to explore, but it is not required — the notebook is just one convenient
way to call the same library.

**3. On your own data, in a script**

```python
from transnested import TransNestedSampler

sampler = TransNestedSampler(my_data, noise_sigma=0.5)
result = sampler.run()
print(result)
```

Only `data` and `noise_sigma` are required. Everything else has a default.

---

## Reading a result

```python
result.model_order              # most probable number of pulses
result.posterior_over_k         # probability of every possible count, as an array
result.pulses                   # recovered [amplitude, position] for each pulse
result.log_evidence             # log Z, with result.log_evidence_error
result.converged                # True if the run finished properly
result.compare()                # prints and returns the comparison against RJ-MCMC
result.plot()                   # one figure: the fit, and the probability of each k
result.as_dict()                # every number above, in one dictionary
```

**Two things worth knowing before you trust an answer:**

- Use `posterior_over_k`, not just `model_order`. A run that is 95% sure of
  its answer and one that is 35% sure can report the same `model_order` —
  the posterior is what tells them apart.
- If `result.converged` is `False`, the run stopped early rather than
  settling, and `log_evidence` is a lower bound rather than a final number.
  `result.termination_reason` says why.

---

## Settings

Two different things get configured in two different places, because they
are two different kinds of setting.

**How the Nested Sampler itself runs — set when you build the object:**

```python
TransNestedSampler(
    data, noise_sigma,
    pulse_width=0.1,       # width shared by every pulse (fixed, not fitted)
    max_pulses=8,          # largest number of pulses considered
    min_pulses=1,          # set equal to max_pulses to lock the count
    n_live=200,            # more = more accurate, slower. try 50 for a quick look
)
```

**Whether and how to run the RJ-MCMC comparison — set when you call `run()`:**

```python
sampler.run(
    compare_with_rjmcmc=True,     # on by default
    rjmcmc_temperatures=10,       # leave this at 10 — see below
)
```

### Why 10 temperatures for one sampler and 1 for the other

The Nested Sampler needs only a single internal temperature: its shrinking
likelihood constraint already does the job that parallel tempering is for.
Plain RJ-MCMC has no equivalent mechanism, so it depends on the ladder. This
was measured directly, at high signal-to-noise, changing nothing else:

| | agreement with the truth |
|---|---|
| RJ-MCMC, 1 temperature | wrong by 3 pulses |
| RJ-MCMC, 10 temperatures | correct |
| Nested Sampler (always 1 temperature) | correct |

Running with `rjmcmc_temperatures=1` reproduces that failure, and prints a
warning saying so.

---

## What's in the folder

| file | what it does |
|---|---|
| `transnested.py` | **the interface — the only file you need to read** |
| `TransNested_Control_Panel.ipynb` | one-cell-of-settings notebook |
| `experiment.py` | runs one full analysis |
| `nested_sampler.py` | the Nested Sampling algorithm |
| `baseline_rjmcmc.py` | the RJ-MCMC comparison |
| `pulse_model.py` | the pulse shape, likelihood, and priors — defined once, here |
| `diagnostics.py` | convergence checks used internally |
| `test_pulse_model.py` | confirms the model and likelihood are correct |
| `test_diagnostics.py` | confirms the convergence checks are correct |
| `final_check.py` | one full-scale run with 18 built-in sanity checks |

All files stay in one folder — `transnested.py` imports the rest. If a file
is missing, it says so by name rather than failing with a confusing error.

To check everything works, run:

```bash
python test_pulse_model.py
python test_diagnostics.py
python final_check.py          # slower: one full production run, ~40-70 min
```

---

## What it found, on real testing

Across 69 test configurations (varying noise level, true pulse count, and
how close together the pulses were):

- Exact pulse count recovered: **62%** of the time (Nested Sampler),
  **65%** (RJ-MCMC, with the temperature ladder)
- The two methods **agreed with each other 77%** of the time, and were never
  more than one pulse apart when they disagreed

So this isn't a claim that the Nested Sampler is more accurate — the two are
close. The finding is that it reaches the same answer using one sampling
chain where the alternative needs ten.

The one place a real difference showed up: when two pulses were very close
together (about half a pulse-width apart), the Nested Sampler told them apart
correctly every time, and the RJ-MCMC baseline merged them into one. That is
the only situation, across all 69 tests, where the baseline under-counted.

---

## Limits, honestly

- The pulse width is **fixed**, not fitted — set `pulse_width` to match your
  data. Pulses of varying width are not supported, for now.
- The noise level is **assumed known**.
- The count is capped at `max_pulses`. If the answer piles up at that limit,
  raise it — you'll get a warning if this happens.
- The method tends to slightly **over-count**, not under-count, when it's
  wrong.
- A small fraction of very-high-signal runs stop before fully converging;
  `result.converged` always tells you when this happens.

---

## Changing the model

Everything about the signal shape — the pulse formula, the likelihood, the
priors — lives in `pulse_model.py`, and only there. Both samplers read from
it, so if you ever need a different kind of pulse, that's the one file to
change.
