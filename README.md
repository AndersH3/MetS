# MetS — Metabolic Syndrome and Brain Age Gap Modeling

Bayesian model comparison of the relationship between **metabolic syndrome (MetS) component constellations** and **brain age gap (BAG)** using published aggregate estimates from Table S7 of Dove et al. (2026).

The repository contains parallel **PyStan** and **RStan** implementations of a participant-weighted seven-model analysis. The primary analysis fits and compares nonlinear response curves to the 32 possible constellations of five MetS components and evaluates predictive performance with **exact leave-one-constellation-out cross-validation (exact LOO)**.

> **Scope:** this is exploratory secondary modeling of published aggregate BAG estimates. It is not an individual-level analysis and should not be interpreted as recovering participant-level likelihoods or causal effects.

## Data

The scripts embed all 32 Table S7 constellations formed by five binary MetS components:

1. central adiposity;
2. elevated blood pressure;
3. hyperglycemia;
4. elevated triglycerides;
5. low HDL cholesterol.

For each constellation the analysis uses:

- the published **BAG least-squares mean (years)**;
- the corresponding participant count;
- the five-component binary pattern.

The embedded counts sum to **27,375 participants**. Both implementations validate the counts and BAG values internally before fitting.

Data source used by the code:

> Dove et al. (2026), *Alzheimer's & Dementia*. DOI: [10.1002/alz.71563](https://doi.org/10.1002/alz.71563)

The scripts use the **Table S7 BAG LS means**, not the reported beta coefficients.

## Participant weighting

The primary estimand gives more influence to constellations represented by more participants while deliberately avoiding treatment of the 32 aggregate BAG estimates as 27,375 independent observations.

For a fit containing **G** constellations, the weight for constellation **i** is

    w_i = G * n_i / sum_j(n_j)

so that

    sum_i(w_i) = G
    mean(w_i) = 1

For the full dataset,

    w_i = 32 * n_i / 27375

For an exact-LOO training fold with 31 constellations, the weights are recomputed from the training data:

    w_i,train = 31 * n_i / sum_training(n_i)

This normalization redistributes influence among aggregate observations without inflating the nominal likelihood size.

### Primary and sensitivity scoring

For held-out constellation **i**, the primary predictive contribution is

    weighted_lpd_i = (32 * n_i / 27375) * lpd_i

The repository reports both:

- **participant-weighted exact-LOO ELPD** — the primary criterion;
- **equal-constellation exact-LOO ELPD** — a sensitivity analysis in which the same participant-weighted fitted models are evaluated with equal weight across the 32 held-out constellations.

The sensitivity analysis therefore changes the **evaluation weighting**, not the training rule.

## Model family

For constellation **i**, define a weighted MetS score

    x_i = M_i^T * w

where **M_i** is the five-element MetS indicator vector and **w** is a simplex of learned non-negative component weights summing to one. BAG is modeled as

    mu_i = m + k * f(x_i)

with **k > 0**.

Seven candidate models are fitted:

| Model | Response shape |
| --- | --- |
| **Linear** | f(x) = x |
| **Power** | f(x) = x^p, with p > 0 |
| **Normalized exponential** | normalized exponential with p = 1 |
| **Power-exponential hybrid** | normalized exponential-power family with free p and a |
| **Direct p-ridge** | hybrid constrained by an estimated linear p-vs-a relation |
| **Hard log-p ridge** | hybrid constrained by an estimated linear log(p)-vs-a relation |
| **Soft log-p ridge** | log(p) ridge with residual variation around the relation |

The nonlinear models use a numerically stable, branchless implementation of

    H(x,p,a) = [exp(a * x^p) - 1] / [exp(a) - 1]

with a smooth approximation around **a = 0**.

The ridge relations are learned from the unrestricted hybrid model. During exact LOO they are re-estimated **inside each training fold**, so the held-out BAG value does not participate in the ridge relation used to predict itself.

## Exact-LOO model comparison

With 32 constellations and seven models, the outer exact-LOO analysis requires **224 held-out model fits**, in addition to the full-data fits. Fold results are cached so completed work can be reused.

The analysis also computes:

- participant-weighted and equal-constellation RMSE/MAE;
- stacking weights;
- Bayesian-bootstrap pseudo-BMA+ weights;
- Bayesian-bootstrap model-ranking stability;
- full-data and fold-level HMC diagnostics;
- posterior summaries for MetS component weights and model parameters;
- posterior BAG curves;
- a blood-pressure-only posterior prediction summary.

## Results from the committed RStan run

The committed console log reports the following primary participant-weighted exact-LOO ranking:

| Rank | Model | ELPD | RMSE | MAE |
| ---: | --- | ---: | ---: | ---: |
| 1 | Normalized exponential | **2.64180** | 0.21723 | 0.12722 |
| 2 | Direct p-ridge | 2.31923 | 0.22165 | 0.13807 |
| 3 | Hard log-p ridge | 2.30284 | 0.22115 | 0.13485 |
| 4 | Linear | 1.70159 | 0.22515 | 0.14150 |
| 5 | Power | 1.50031 | 0.21734 | 0.12839 |
| 6 | Soft log-p ridge | 1.48019 | 0.22053 | 0.13328 |
| 7 | Power-exponential hybrid | 1.28052 | 0.21936 | 0.12738 |

The **normalized-exponential model also ranks first in the equal-constellation sensitivity analysis**, with ELPD (-24.2438).

This ranking should not be read as a sharp model-selection result. For example, the participant-weighted ELPD difference between the normalized-exponential and direct-p-ridge models is only about **0.323**, with a pointwise approximate SE of about **0.540**. The small number of aggregate observations and adaptive history of the candidate family warrant conservative interpretation.

### Model-weight results

For the primary participant-weighted criterion, the committed run gives approximately:

| Model | Stacking weight | BB pseudo-BMA+ |
| --- | ---: | ---: |
| Linear | 0.0872 | 0.1257 |
| Power | ~0 | 0.1025 |
| Normalized exponential | **0.9128** | **0.2363** |
| Power-exponential hybrid | ~0 | 0.0825 |
| Direct p-ridge | ~0 | 0.1804 |
| Hard log-p ridge | ~0 | 0.1887 |
| Soft log-p ridge | ~0 | 0.0839 |

Participant-weighted Bayesian-bootstrap winner frequencies were:

- normalized exponential: **55.9%**;
- hard log-p ridge: **21.5%**;
- linear: **13.4%**;
- power: **5.2%**;
- direct p-ridge: **4.0%**;
- hybrid and soft log-p ridge: negligible in this diagnostic.

The stacking mixture improves the participant-weighted score only slightly over the best single model (about **+0.0109 ELPD**).

### MCMC diagnostics

In the committed RStan run:

- all seven full-data fits had **0 divergences**;
- all exact-LOO fits had **0 divergences**;
- no full-data or exact-LOO fit hit the configured maximum tree depth;
- full-data maximum R-hat was about **1.0031**;
- worst exact-LOO R-hat across the summarized models/folds was about **1.0071**.

These diagnostics address sampler behavior, not model validity or the limitations of the aggregate data.

## Repository contents

| File | Description |
| --- | --- |
| `mets_brain_age_participant_weighted_reference_pystan_v3_table_s7.py` | PyStan 3 reference implementation |
| `mets_brain_age_participant_weighted_reference_rstan_table_s7_final_fixed.r` | RStan translation/refactor with exact-LOO caching and `loo` model weights |
| `run_rstan_rstudio.txt` | Console output from the committed RStan run |
| `rstan_output_github.zip` | Archived RStan output files |
| `LICENSE` | GNU Affero General Public License v3 |

## Running the PyStan implementation

### Requirements

- Python 3
- a working C/C++ build environment suitable for PyStan
- Python packages: `pystan`, `arviz`, `numpy`, `pandas`, `scipy`, `matplotlib`

Example setup:

```bash
git clone https://github.com/AndersH3/MetS.git
cd MetS

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install pystan arviz numpy pandas scipy matplotlib

python3 mets_brain_age_participant_weighted_reference_pystan_v3_table_s7.py
```

The PyStan script writes to:

```text
mets_brain_age_participant_weighted_table_s7_results/
```

By default the Python implementation has `FRESH_RUN = True`, so review that setting before rerunning a costly analysis if you want to preserve cached results.

## Running the RStan implementation

### Requirements

- R
- a working C++ toolchain compatible with RStan
- `rstan`
- `loo`

Install the required R packages:

```r
install.packages(c("rstan", "loo"))
```

Then run from R or RStudio:

```r
source("mets_brain_age_participant_weighted_reference_rstan_table_s7_final_fixed.r")
```

The RStan script writes to:

```text
mets_brain_age_participant_weighted_table_s7_results_rstan/
```

The committed R version has `FRESH_RUN <- FALSE`. Existing compatible exact-LOO cache files are therefore reused. Set it to `TRUE` only when you intentionally want to delete the output directory and recompute the fits from scratch.

## Sampling configuration

The reference configuration is deliberately intensive:

| Setting | Full-data fits | Exact-LOO fits |
| --- | ---: | ---: |
| Chains | 4 | 4 |
| Post-warmup samples / chain | 3000 | 2000 |
| Warmup / chain | 3000 | 1500 |
| `adapt_delta` | 0.99 | 0.99 |
| `max_treedepth` | 15 | 15 |

Other fixed settings include:

- seed base: `2026081902`;
- 10,000 Bayesian-bootstrap draws for pseudo-BMA+ model weights;
- 50,000 bootstrap draws for ranking stability;
- 401 points for posterior response curves.

Exact LOO is computationally much more expensive than a single full-data fit. The cache exists specifically to make interrupted/resumed analyses practical.

## Principal RStan outputs

The R implementation writes, among other files:

- `table_s7_embedded_data_exact.csv`
- `participant_weights.csv`
- `full_data_mcmc_diagnostics.csv`
- `full_data_weight_summary.csv`
- `full_data_parameter_summary.csv`
- `full_data_ridge_relation.csv`
- `blood_pressure_only_predictions.csv`
- `posterior_curves.csv`
- `exact_loo_pointwise.csv`
- `ridge_relation_by_fold.csv`
- `exact_loo_model_comparison_PARTICIPANT_WEIGHTED_PRIMARY.csv`
- `exact_loo_model_comparison_equal_constellation_sensitivity.csv`
- `model_weights_PARTICIPANT_WEIGHTED_PRIMARY.csv`
- `model_weights_equal_constellation_sensitivity.csv`
- `bootstrap_ranking_PARTICIPANT_WEIGHTED_PRIMARY.csv`
- `bootstrap_differences_PARTICIPANT_WEIGHTED_PRIMARY.csv`
- `exact_loo_mcmc_diagnostics.csv`
- `participant_weighted_stacking_diagnostics.csv`
- `posterior_mean_curves.png`
- `exact_loo_PARTICIPANT_WEIGHTED_PRIMARY.png`
- `analysis_manifest.rds` / `analysis_manifest.txt`
- `sessionInfo.txt`
- cached fold fits and full posterior draws.

## Interpretation and limitations

Several limitations are structural and cannot be repaired by more computation:

1. **Aggregate outcome data.** The 32 BAG values are adjusted constellation-level LS means, not individual participant BAG observations.
2. **Participant-count weighting is a modeling choice.** It changes relative influence; it does not reconstruct 27,375 independent BAG observations.
3. **Unknown covariance.** The covariance matrix among the adjusted Table S7 BAG estimates is unavailable and is therefore not modeled.
4. **Model-development history.** If the nonlinear candidate family was developed using these same 32 values, exact LOO cannot undo that historical model-search/adaptation bias.
5. **Small effective sample.** The predictive comparison contains 32 aggregate held-out units, so uncertainty in model differences is substantial.
6. **No causal interpretation.** Associations between MetS constellations and BAG do not establish that changing a component would cause the predicted BAG change.
7. **External validation is still required.** Stronger claims require independent data, ideally with participant-level outcomes and uncertainty/covariance information for the adjusted estimates.

## Reproducibility notes

The R implementation records a run hash based on model code and important settings, saves a manifest and `sessionInfo()`, and versions exact-LOO cache files by that run hash. This is intended to reduce accidental reuse of incompatible cached fits.

The committed RStan log was produced with:

- R 4.6.1;
- rstan 2.39.0.9000;
- loo 2.10.1;
- run hash `1af08f291b17801f`.

Future package versions can change numerical details even when the model specification is unchanged.

## License

This repository is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [LICENSE](LICENSE).

## Citation

If you reuse the analysis code, cite this repository and the underlying source publication containing Table S7. The numerical Table S7 BAG LS means and participant counts are derived from the source study, not generated by this repository.
