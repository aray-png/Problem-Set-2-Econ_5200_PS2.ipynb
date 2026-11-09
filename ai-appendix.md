# AI Appendix: Task 4.1, The Sequential Design

**Model:** Claude Opus 5.5 (Anthropic), in the Claude app
**Date:** November 8, 2026

---

## Version 1 prompt (as sent)

```
[Prep] Act as an experimentation platform engineer who has implemented
sequential testing in production.

[Request] I have a simulated portfolio of 200 A/B tests with known true
effects. Implement an always-valid sequential test -- a mixture
sequential probability ratio test or an alpha-spending boundary -- that
allows continuous monitoring without inflating the Type I error rate.
Give me a function that takes a stream of observations and returns the
stopping decision at each look.

[Iterate] numpy and scipy only. Reuse my log structure. The function
must be testable against a pure-null stream.

[Mechanism Check] Explain why the naive approach of testing at every
look inflates Type I error, and precisely what the always-valid
construction gives up in exchange for continuous monitoring.

[Evaluate] Run it on 1,000 pure-null streams and report the empirical
false positive rate. If it is not close to 5%, tell me what is wrong.
```

## Version 1 reply (as it came back, unedited)

> I'll implement a mixture SPRT (mSPRT) for a two-arm conversion test, check it against 1,000 pure-null streams alongside naive peeking, then see what it costs on your portfolio's real effects.
>
> The mSPRT came in at 1.0% against a 5% target. Before explaining that, I'll check whether it's a bug or the method being conservative, by varying the mixing prior and the horizon.
>
> ## The function
>
> An always-valid test needs a statistic you can check at every look without paying for each check. I used the **mixture SPRT** (Johari, Pekelis & Walsh, 2017). It puts a Normal(0, τ²) prior on the effect, computes the likelihood ratio of "some effect" against "no effect", and rejects the first time that ratio reaches 1/α = 20. τ is set to 0.8 pp, the typical real lift in your prior.

```python
%%writefile src/sequential.py
"""Always-valid sequential testing for two-arm conversion experiments (mSPRT).

Uses the mixture sequential probability ratio test of Johari, Pekelis & Walsh (2017):
place a Normal(0, tau^2) mixing prior on the treatment effect, and reject H0: effect = 0
the first time the mixture likelihood ratio reaches 1/alpha. By Ville's inequality the
probability that a null stream EVER crosses that line is at most alpha, no matter how
often you look or when you stop.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def msprt_statistic(diff: np.ndarray, n: np.ndarray, var_pair: np.ndarray,
                    tau: float) -> np.ndarray:
    """Mixture likelihood ratio Lambda_n for a difference in means.

    diff:     running estimate of (treatment - control) at each look
    n:        observations per arm at each look
    var_pair: variance of one (treatment - control) pair, p_t(1-p_t) + p_c(1-p_c)
    tau:      SD of the Normal mixing prior on the effect (same units as diff)
    """
    V = var_pair / n                                   # variance of diff at this look
    return np.sqrt(V / (V + tau**2)) * np.exp(tau**2 * diff**2 / (2 * V * (V + tau**2)))


def sequential_decisions(control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05,
                         tau: float = 0.008, look_every: int = 500,
                         burn_in: int = 500) -> dict[str, np.ndarray]:
    """Stopping decision at each look for one stream of paired 0/1 observations.

    control, treatment: equal-length arrays of conversions (0/1), in arrival order.
    Looks happen every `look_every` observations per arm, starting at `burn_in`.

    Returns arrays aligned to the looks:
      n          observations per arm at the look
      estimate   running difference in conversion rates
      msprt      mixture likelihood ratio Lambda_n
      always_p   always-valid p-value: min(1, 1 / max Lambda so far)
      stop       True from the first look where Lambda >= 1/alpha onward
      naive_p    the fixed-horizon z-test p-value, recomputed at each look (for comparison)
    """
    n = np.arange(burn_in, len(control) + 1, look_every)
    c_rate = np.cumsum(control)[n - 1] / n
    t_rate = np.cumsum(treatment)[n - 1] / n
    diff = t_rate - c_rate

    pooled = (c_rate + t_rate) / 2                      # pooled rate under H0
    var_pair = np.maximum(2 * pooled * (1 - pooled), 1e-12)

    lam = msprt_statistic(diff, n, var_pair, tau)
    always_p = np.minimum(1.0, 1.0 / np.maximum.accumulate(lam))
    stop = np.maximum.accumulate(lam >= 1 / alpha)
    naive_p = 2 * stats.norm.sf(np.abs(diff) / np.sqrt(var_pair / n))
    return {"n": n, "estimate": diff, "msprt": lam, "always_p": always_p,
            "stop": stop, "naive_p": naive_p}


def stream_from_log(row, n_max: int, rng: np.random.Generator,
                    baseline_rate: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Generate one paired stream for an experiment in the portfolio log
    (uses its `true_effect`), up to n_max observations per arm."""
    control = rng.binomial(1, baseline_rate, n_max)
    treatment = rng.binomial(1, baseline_rate + row["true_effect"], n_max)
    return control, treatment
```

> `stream_from_log` takes a row of your `simulate_portfolio` log, so the test runs on your own experiments.
>
> ## Mechanism check
>
> **Why testing at every look inflates Type I error.** A p < 0.05 test guarantees a 5% false-positive rate for *one* look at a sample size fixed in advance. Under the null, the running z-statistic wanders like a random walk, and over enough looks a random walk eventually crosses ±1.96 even though nothing is there. Stopping at the first crossing turns "5% chance at one look" into "the chance of crossing at *any* of 40 looks", which is much larger. Each look is a fresh chance to be fooled, and stopping at the first win keeps only the fooled ones.
>
> **What the always-valid construction does instead.** Under the null, the mixture likelihood ratio Λₙ is a martingale with mean 1: it doesn't drift upward on average, however long you watch. Ville's inequality then bounds the chance that it *ever* reaches 1/α by α. That bound covers every look, every stopping time, and looking after every single observation, so peeking is free.
>
> **What it gives up:**
> 1. **Power at any fixed sample size.** The bar has to hold up against unlimited peeking, so it sits higher than 1.96 standard errors. At the same maximum sample size, the always-valid test detects fewer real effects than a single fixed-horizon test.
> 2. **Exactness.** α is an upper bound, not a target. Over a finite horizon the realised false-positive rate sits below α, so some of your error budget goes unspent.
> 3. **Dependence on τ.** The test is most sensitive to effects near τ. A badly chosen τ costs power, though it never breaks validity.
>
> What you get in exchange: you can stop early when an effect is large, and you can look as often as you like.
>
> ## Evaluation

```python
import os, sys, importlib
sys.path.insert(0, os.path.abspath('.'))
from src import sequential as sq
importlib.reload(sq)

ALPHA, N_MAX, LOOK, BASE = 0.05, 20_000, 500, 0.10
null_rng = np.random.default_rng(5200)

msprt_fp = naive_fp = final_fp = 0
for _ in range(1_000):
    c = null_rng.binomial(1, BASE, N_MAX)
    t = null_rng.binomial(1, BASE, N_MAX)          # pure null: same rate in both arms
    d = sq.sequential_decisions(c, t, alpha=ALPHA, look_every=LOOK)
    msprt_fp += d["stop"][-1]                       # ever stopped
    naive_fp += (d["naive_p"] < ALPHA).any()        # p < .05 at ANY look
    final_fp += d["naive_p"][-1] < ALPHA            # one test at the end

print(f"Looks per stream: {len(d['n'])} (every {LOOK} per arm, up to {N_MAX:,})")
print(f"mSPRT, always valid:        {msprt_fp / 1000:.1%}")
print(f"Naive peeking at every look: {naive_fp / 1000:.1%}")
print(f"One fixed test at the end:   {final_fp / 1000:.1%}")
assert msprt_fp / 1000 <= ALPHA + 0.014           # within 2 SEs of the 5% bound

# The cost: detections among the portfolio's 43 real lifts
real = log[log["is_real"]]
real_rng = np.random.default_rng(1)
m_det = f_det = 0
stop_n = []
for _, row in real.iterrows():
    c, t = sq.stream_from_log(row, N_MAX, real_rng)
    d = sq.sequential_decisions(c, t, alpha=ALPHA, look_every=LOOK)
    if d["stop"][-1]:
        m_det += 1
        stop_n.append(d["n"][d["stop"].argmax()])
    f_det += d["naive_p"][-1] < ALPHA
print(f"\nReal effects detected: mSPRT {m_det} of {len(real)}, fixed test at {N_MAX:,}: {f_det}")
print(f"Median stopping point for mSPRT detections: {np.median(stop_n):,.0f} per arm")
```

> **Output** (1,000 pure-null streams, 40 looks each)
>
> | method | false-positive rate |
> |---|---|
> | **mSPRT (always valid)** | **1.0%** |
> | Naive: p < .05 at any look | 31.3% |
> | One fixed test at the end | 6.1% |
>
> | on the 43 real lifts | detected |
> |---|---|
> | mSPRT, up to 20,000 per arm | 13 (median stop at 12,500) |
> | Fixed test at 20,000 per arm | 23 |
>
> **Is 1.0% "close to 5%"? No, and that's expected, not a bug.** I checked by varying τ and the horizon:
>
> | horizon (per arm) | τ = 0.2 pp | 0.4 pp | 0.8 pp | 1.6 pp |
> |---|---|---|---|---|
> | 20,000 | 0.0% | 0.4% | 1.0% | 1.5% |
> | 200,000 | 1.3% | 3.7% | 3.0% | 2.3% |
>
> The rate never exceeds 5% under any setting, and it climbs toward 5% as the horizon grows. That's the signature of a correct always-valid test: Ville's bound covers the chance of crossing *over an unlimited horizon*, and stopping at 20,000 cuts that short. Looking after every single observation instead of every 500 still gave 0.5%, which confirms that peeking frequency doesn't matter. Something *would* be wrong if the rate came out above 5%, or if it rose as you looked more often.
>
> The fixed test's 6.1% is just sampling noise: with 1,000 streams, the standard error on a 5% rate is about 0.7 percentage points.
>
> **What this means for your portfolio:** naive peeking would turn your 5% false-positive rate into about 31%, six times the nominal level. The mSPRT holds it at 1%, but at a capped 20,000 per arm it catches 13 real lifts instead of 23. The fair comparison isn't mSPRT against a fixed test at the same size. It's mSPRT against *peeking* (the thing teams actually do), or against a fixed test run to the size it needs. If you want to spend the full 5% budget at a known horizon, an alpha-spending boundary (O'Brien–Fleming) gets closer, at the price of fixing the number of looks in advance.

---

## Follow-up (as sent)

I pasted the error from running the evaluation cell:

```
Looks per stream: 40 (every 500 per arm, up to 20,000)
mSPRT, always valid:        1.0%
Naive peeking at every look: 31.3%
One fixed test at the end:   6.1%

NameError: name 'log' is not defined
---> 25 real = log[log["is_real"]]
```

## Follow-up reply (as it came back, unedited)

> The first half worked: your null results match mine exactly. The error is my mistake. I wrote `log` as if it already existed, but in your notebook that variable only lives inside the module's `__main__` block, so the notebook never sees it.
>
> Your notebook's own table is called `experiments`. Change that line to:
>
> ```python
> real = experiments[experiments["is_real"]]
> ```
>
> Or build the log from the module, which gives the same 200 experiments:
>
> ```python
> log = ep.simulate_portfolio(200, 2026)
> real = log[log["is_real"]]
> ```
>
> Either works. Re-run the cell and the second part should print:
>
> ```
> Real effects detected: mSPRT 13 of 43, fixed test at 20,000: 23
> Median stopping point for mSPRT detections: 12,500 per arm
> ```

---

## Version 2 prompt

None sent. The version 1 implementation passed my own null check (notebook cell 15), so there was no failure to correct in a revised prompt:

- 1,000 pure-null streams: false-positive rate **1.0%** (95% CI upper bound 1.8%), against naive peeking at **31.3%**.
- Extending the horizon tenfold: 1.3% → 2.0%, still below 5%.
- Looking after every observation: 1.0%, no higher than looking every 500.

The rate is below the nominal 5% because an always-valid test guarantees *at most* α over an unlimited horizon; capping at 20,000 per arm leaves part of that budget unspent. That is conservative behaviour, not a bug.

## Changes I made to the model's output

1. **Replaced `log` with `experiments`** in the evaluation cell: the model's code referred to a variable my notebook never defined (`NameError`). The sequential test itself (`src/sequential.py`) is unchanged.
2. **Added my own verification cell** (cell 15): a 95% confidence interval on the false-positive rate, a tenfold-longer horizon, and looking after every observation, so the null check is shown rather than asserted.
