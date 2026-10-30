# Two-exponential model extension (2026-10-30)

This branch adds [`mets_double_exponential_rstan_comparison.R`](mets_double_exponential_rstan_comparison.R) as an **incremental** exact-LOO RStan analysis alongside the original seven-model results. It does not alter the main branch or rerun the seven original models.

## Candidate equations

Using the same learned nonnegative simplex MetS component weights, `x = M %*% mets`, the extension fits three interpretations of the proposed formula.

1. **Two_exp_unrestricted_A**: `A*exp(B*x+C)+(1-A)*exp(D*x+E)`, with real (unrestricted) `A`. This is the literal equation; negative BAG is possible if `A` lies outside `[0,1]`.
2. **Two_exp_convex_A**: the identical equation, with `0 < A < 1`. This cannot produce negative BAG values.
3. **Two_exp_convex_A_offset**: `m+A*exp(B*x+C)+(1-A)*exp(D*x+E)`, with `0 < A < 1`; the intercept permits negative BAG.

All three estimate the same 5-component simplex as the seven existing candidate models. The slopes are ordered `B>D` to remove label-switching symmetry, and all parameters have documented proper priors. Because `A`, `C`, and `E` are only weakly distinguishable from the likelihood, review effective sample sizes, R-hat and divergences.

## Running locally

Requires R, a C++ compiler, and RStan. In a checked-out copy of this branch:

```r
install.packages("rstan") # if not already installed
source("mets_double_exponential_rstan_comparison.R")
```

The run writes and caches results in `mets_double_exponential_table_s7_rstan_results/`. `FRESH_RUN <- FALSE` resumes without invalidating compatible cached fits. Default sampler settings match the earlier reference run: four chains; 3,000 warmup + 3,000 retained iterations per chain for full-data fits, and 1,500 warmup + 2,000 retained iterations per chain for the 32 exact-LOO fits.

It uses the same `G*n_i/sum_training(n)` participant weighting in each training fold and `(32*n_i/27375)*heldout_log_predictive_density` for the participant-weighted primary score. The final CSV includes the *historical* seven-model numbers for reference; only the three new models are actually fitted in this script.

**These are aggregate adjusted BAG LS means** for 32 constellations, not individual patient data. The source study's covariance between estimates is unknown. No causal effect is established.

## Preliminary optimization-only check

The following values are from a separate Python weighted nonlinear least-squares run. These are **not RStan MCMC/ELPD results** and are intended as a diagnostic rather than a definitive Bayesian ranking.

| Model | Full-data weighted RMSE | Exact-LOO weighted RMSE |
|---|---:|---:|
| Linear | 0.1864 | 0.2275 |
| Normalized exponential | 0.1774 | **0.2221** |
| Positive two-exponential | 0.3028 | 0.3899 |
| Positive two-exponential with intercept | 0.1774 | 0.2569 |
| Signed two-exponential (positive-minus-positive reparameterization of unrestricted A) | 0.1774 | 0.2484 |

The signed fit is an algebraic subclass of the unconstrained-`A` model (with one coefficient of each sign), and does not establish the global RStan fit for unrestricted-`A`.

The original committed normalized-exponential model has participant-weighted RStan exact-LOO ELPD **2.64180** and RMSE **0.21723**. It should only be compared with new RStan scores *after* running the new Bayesian code, not by equating its RStan criterion with the optimization-only scores above.

## Reproducibility

The RStan extension writes exact-LOO pointwise predictions, aggregate model comparison, fold diagnostics, posterior parameter summaries, posterior curves, and R session information. The model code and run settings are fingerprinted for cache compatibility.
