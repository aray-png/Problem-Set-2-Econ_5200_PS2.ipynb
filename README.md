# Auditing an A/B Testing Programme: The Winner's Curse

**ECON 5200 · Problem Set 2**

A client ships any A/B test with p < 0.05. I simulated a year of their experimentation programme, 200 tests where I know the true effect of every change, and audited what that rule actually delivers against what it promises.

## Objective

I wanted to measure how much a p < 0.05 shipping rule overstates the gains it reports, and find a rule that delivers more real improvement.

## Methodology

- **Simulated the portfolio.** I drew 200 true effects on a 10% baseline conversion rate from a mixture prior: 75% are nulls (Normal(0, 0.1 pp)) and 25% are real lifts (|Normal(0.8 pp, 0.4 pp)|). Sample sizes ran from 1,000 to 50,000 per arm, so power varied widely. I recorded each test's true effect so every decision could be checked.
- **Applied the client's rule.** I shipped every test with p < 0.05 and a positive estimate, then compared the shipped estimates with the shipped true effects.
- **Measured the exaggeration by power.** I split tests into low, medium and high power and computed the inflation factor (average estimate ÷ average true effect) in each band.
- **Bootstrapped the realised return.** The sum of true effects among shipped tests has no closed-form standard error, so I wrote my own resampling loop, checked it against `scipy.stats.bootstrap`, and reported a BCa interval.
- **Designed an alternative rule.** I shrank each estimate toward the prior and shipped when the shrunken estimate exceeded 0.3 pp, then compared it with the client's rule, a stricter p < 0.01 rule and an 80% power floor.
- **Built a module.** `src/experiment_portfolio.py` simulates a portfolio, applies a rule and evaluates it, with docstrings, type hints and assertions.
- **Tested an AI-written sequential design.** I asked an AI for an always-valid sequential test (mSPRT) that allows continuous monitoring, then checked it myself on 1,000 pure-null streams.

## Key Findings

- **Most shipped wins were real, but their size was inflated.** The rule shipped 23 of 200 tests, and 83% were real. Their estimates added up to +27.74 conversion points, but the true gain was +17.35. Shipped estimates averaged 1.60x the truth.
- **The inflation is worst in small tests.** Shipped low-power tests overstated their effect by 2.38x, medium-power by 1.43x and high-power by 1.24x. A small test only clears the significance bar when noise pushes its estimate up.
- **The realised return is uncertain, and the claimed return falls outside it.** The bootstrap gave a 95% BCa interval of +10.8 to +26.3 points. The +27.74 the team would report lies above it.
- **Shrinkage beat the client's rule.** My rule shipped 36 tests and realised +22.7 points, against +17.35 for p < 0.05. Its forecast of +23.2 was close to what it delivered. Stricter rules did worse (+11.3 for p < 0.01, +7.5 for the power floor) because they threw away real but modest lifts.
- **The trade-off is more duds.** My rule's false-ship rate was 28% against 17%. That costs little when shipping a null is free, and the result would reverse if each shipped change carried an upkeep cost near the 0.3-point hurdle.
- **Peeking at results inflates false positives sixfold.** Testing at p < 0.05 at every one of 40 looks gave a 31.3% false-positive rate on null streams. The mSPRT held it to 1.0%, at the cost of catching 13 of 43 real lifts instead of 23 at the same maximum sample size.

## Repository Contents

| File | What it is |
|---|---|
| `Econ_5200_PS2.ipynb` | The full analysis, run top to bottom |
| `src/experiment_portfolio.py` | Simulator, shipping rules and evaluation |
| `src/sequential.py` | Always-valid sequential test (mSPRT) |
| `ai-appendix.md` | The AI transcript for the sequential design |

## Running It

Open the notebook in Google Colab and choose **Runtime › Restart session and run all**. Every random draw uses a fixed seed, so a single top-to-bottom run reproduces each number above.
