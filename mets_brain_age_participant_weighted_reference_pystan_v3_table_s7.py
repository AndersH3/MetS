#!/usr/bin/env python3
"""
Participant-weighted seven-model PyStan analysis of MetS and BAG
using exact Table S7 BAG LS means
================================

This is the revised reference program using the exact two-decimal BAG LS means
and participant counts reported in Table S7 for all 32 MetS constellations.
The 32 constellation-level BAG estimates are weighted DIRECTLY according to
the number of participants in each constellation.

TABLE S7 DATA SOURCE USED IN THIS VERSION
-----------------------------------------
The BAG response vector contains the Table S7 BAG LS Mean values, NOT the
beta coefficients.  The constellation order remains the program's existing
binary-pattern order; Table S7 values have been rearranged into that order.
The participant counts and all 32 binary constellation definitions are also
checked internally against the embedded Table S7 reference values.

PRIMARY WEIGHTING RULE
----------------------
For any fitted dataset containing G constellations:

    weight_i = G * n_i / sum_j(n_j)

Therefore:

    sum_i weight_i = G
    mean(weight_i) = 1

For the full 32-constellation fit:

    weight_i = 32 * n_i / 27375

Consequences:
* A constellation with twice as many participants has twice the influence
  on the curve-fitting residual criterion.
* The total likelihood weight remains equivalent to G observations rather
  than pretending that the 32 adjusted BAG estimates are 27,375 independent
  BAG observations.
* Because sum(weight_i)=G, the total coefficient on log(sigma) in the
  weighted Gaussian likelihood is the same as in an ordinary G-observation
  Gaussian likelihood. The weighting primarily redistributes influence
  among the 32 residuals instead of inflating the nominal data size.

EXACT LOO
---------
For an outer exact-LOO fold containing 31 training constellations, the
TRAINING weights are recomputed as

    weight_i_train = 31 * n_i / sum_training(n_i)

so that every training fit still has average likelihood weight 1.

For evaluation of the held-out constellation, two predictive criteria are
reported:

1. PRIMARY PARTICIPANT-WEIGHTED score

       weighted_lpd_i = full_weight_i * lpd_i

   where

       full_weight_i = 32*n_i/27375.

   The primary total score is

       participant_weighted_ELPD = sum_i weighted_lpd_i.

2. UNWEIGHTED CONSTELLATION sensitivity analysis

       ordinary_ELPD = sum_i lpd_i.

Thus both scientific questions remain visible:
* participant-weighted prediction;
* equal-constellation prediction.

MODELS
------
1. Linear
2. Power
3. Normalized exponential
4. Unrestricted normalized exponential-power hybrid
5. Direct p-ridge
6. Hard log-p ridge
7. Soft log-p ridge

The three ridge procedures remain nested correctly inside exact LOO:
the held-out BAG value never contributes to the ridge relation used to
predict itself.

NUMERICAL FIXES RETAINED
------------------------
This version also retains the numerical improvements identified in the report:

* no arbitrary finite [-8,8] bounds on a, log_p, or z;
* no parameter-dependent if/else branch in the normalized exponential-power
  response;
* no artificial sigma floor;
* positive k and sigma use Stan's native lower-bound constraints;
* Direct p-ridge preserves p = alpha + beta*a exactly while guaranteeing
  p > 0 with a native positive p parameter and a derived a;
* nonlinear posterior predictions are calculated draw-by-draw.

IMPORTANT DATA LIMITATIONS
--------------------------
Direct participant-count weighting is intentional here because it is the
requested estimand. It does NOT turn the 32 adjusted BAG estimates into
27,375 individual observations.

The program still cannot recover:
* individual-level UK Biobank BAG observations;
* the true covariance matrix among the 32 adjusted BAG estimates;
* historical model-search bias from development on these same 32 values;
* external validation information that has not been supplied.

Accordingly, this remains exploratory secondary modeling of published
aggregate estimates.

RUN
---
    cd ~/Downloads
    python3 mets_brain_age_participant_weighted_reference_pystan_v3_table_s7.py

DEPENDENCIES
------------
    python3 -m pip install pystan arviz numpy pandas scipy matplotlib
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import warnings
from importlib import metadata
from pathlib import Path
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stan
from scipy.optimize import minimize
from scipy.special import logsumexp


# =====================================================================
# 0. CONFIGURATION
# =====================================================================

FRESH_RUN = True

OUTPUT_DIR = Path("mets_brain_age_participant_weighted_table_s7_results")
CACHE_DIR = OUTPUT_DIR / "exact_loo_cache"
STAN_DIR = OUTPUT_DIR / "stan_models"
FULL_DRAW_DIR = OUTPUT_DIR / "full_fit_draws"

# Frozen candidate set and pre-declared primary model.
PRESPECIFIED_PRIMARY_MODEL = "Normalized_exponential"

# Full-data MCMC.
FULL_CHAINS = 4
FULL_NUM_SAMPLES = 3000
FULL_NUM_WARMUP = 3000

# Exact-LOO MCMC.
CV_CHAINS = 4
CV_NUM_SAMPLES = 2000
CV_NUM_WARMUP = 1500

ADAPT_DELTA = 0.99
MAX_TREEDEPTH = 15

SEED_BASE = 2026081902

# Smooth branchless approximation uses
# b = sqrt(a^2 + SMOOTH_A_EPS^2).
# 1e-12 makes the modification at a≈0 numerically negligible.
SMOOTH_A_EPS = 1e-12

N_BB_MODEL_WEIGHTS = 10000
N_BOOTSTRAP_RANKING = 50000
X_GRID_N = 401

COMPONENT_NAMES = [
    "central_adiposity",
    "elevated_blood_pressure",
    "hyperglycemia",
    "elevated_triglycerides",
    "low_HDL_cholesterol",
]

MODEL_NAMES = [
    "Linear",
    "Power",
    "Normalized_exponential",
    "Power_exponential_hybrid",
    "Direct_p_ridge",
    "Hard_logp_ridge",
    "Soft_logp_ridge",
]

RIDGE_MODEL_NAMES = {
    "Direct_p_ridge",
    "Hard_logp_ridge",
    "Soft_logp_ridge",
}


# =====================================================================
# 1. EMBEDDED PUBLISHED DATA
# =====================================================================

# Columns:
# 0 central adiposity
# 1 elevated blood pressure
# 2 hyperglycemia
# 3 elevated triglycerides
# 4 low HDL cholesterol

M = np.asarray(
    [
        [0,0,0,0,0],
        [0,0,0,0,1],
        [1,0,0,0,1],
        [0,0,1,0,0],
        [0,0,0,1,0],
        [0,0,0,1,1],
        [1,0,0,0,0],
        [1,0,1,0,0],
        [1,0,1,1,0],
        [0,1,0,0,0],
        [0,0,1,1,0],
        [1,0,0,1,1],
        [1,0,1,0,1],
        [1,0,0,1,0],
        [0,1,0,1,0],
        [0,1,1,0,0],
        [1,0,1,1,1],
        [1,1,0,0,0],
        [0,1,0,0,1],
        [0,1,0,1,1],
        [1,1,1,0,0],
        [1,1,0,1,0],
        [0,1,1,1,0],
        [0,0,1,1,1],
        [1,1,0,0,1],
        [1,1,0,1,1],
        [0,0,1,0,1],
        [1,1,1,1,0],
        [0,1,1,1,1],
        [0,1,1,0,1],
        [1,1,1,1,1],
        [1,1,1,0,1],
    ],
    dtype=int,
)

# Exact BAG LS Mean values from Table S7, rearranged into the M-row order
# used by this program.  These are NOT the beta coefficients.
BAG = np.asarray(
    [
        -0.45,  #  1: None
        -0.39,  #  2: HDL
        -0.22,  #  3: Adip+HDL
        -0.18,  #  4: Gluc
        -0.19,  #  5: Trig
         0.09,  #  6: Trig+HDL
         0.10,  #  7: Adip
         0.14,  #  8: Adip+Gluc
         0.34,  #  9: Adip+Trig+Gluc
         0.42,  # 10: BP
         0.54,  # 11: Trig+Gluc
         0.69,  # 12: Adip+Trig+HDL
         0.73,  # 13: Adip+Gluc+HDL
         0.75,  # 14: Adip+Trig
         0.73,  # 15: Trig+BP
         0.86,  # 16: BP+Gluc
         0.93,  # 17: Adip+Trig+Gluc+HDL
         0.94,  # 18: Adip+BP
         0.85,  # 19: BP+HDL
         0.91,  # 20: Trig+BP+HDL
         1.19,  # 21: Adip+BP+Gluc
         1.25,  # 22: Adip+Trig+BP
         1.30,  # 23: Trig+BP+Gluc
         1.35,  # 24: Trig+Gluc+HDL
         1.35,  # 25: Adip+BP+HDL
         1.58,  # 26: Adip+Trig+BP+HDL
         1.62,  # 27: Gluc+HDL
         2.04,  # 28: Adip+Trig+BP+Gluc
         2.18,  # 29: Trig+BP+Gluc+HDL
         2.19,  # 30: BP+Gluc+HDL
         2.47,  # 31: All five
         2.77,  # 32: Adip+BP+Gluc+HDL
    ],
    dtype=float,
)

N_COUNTS = np.asarray(
    [
        4850,
        834, 172, 288, 948, 569, 497, 51, 31,
        5388, 84, 226, 47, 244, 1987, 437, 63,
        1094, 2034, 1845, 186, 857, 252, 75,
        740, 1257, 98, 204, 465, 451, 727, 374,
    ],
    dtype=int,
)

CONSTELLATION_ID = np.arange(1, 33, dtype=int)
N_TOTAL = int(N_COUNTS.sum())

# Independent reference copy used only for validation/auditing.
TABLE_S7_N_COUNTS_REFERENCE = np.asarray(
    [
        4850,
        834, 172, 288, 948, 569, 497, 51, 31,
        5388, 84, 226, 47, 244, 1987, 437, 63,
        1094, 2034, 1845, 186, 857, 252, 75,
        740, 1257, 98, 204, 465, 451, 727, 374,
    ],
    dtype=int,
)

TABLE_S7_BAG_LS_MEAN_REFERENCE = np.asarray(
    [
        -0.45, -0.39, -0.22, -0.18, -0.19, 0.09, 0.10, 0.14,
         0.34,  0.42,  0.54,  0.69,  0.73, 0.75, 0.73, 0.86,
         0.93,  0.94,  0.85,  0.91,  1.19, 1.25, 1.30, 1.35,
         1.35,  1.58,  1.62,  2.04,  2.18, 2.19, 2.47, 2.77,
    ],
    dtype=float,
)

TABLE_S7_CONSTELLATION_LABELS = [
    "None",
    "HDL",
    "Adip+HDL",
    "Gluc",
    "Trig",
    "Trig+HDL",
    "Adip",
    "Adip+Gluc",
    "Adip+Trig+Gluc",
    "BP",
    "Trig+Gluc",
    "Adip+Trig+HDL",
    "Adip+Gluc+HDL",
    "Adip+Trig",
    "Trig+BP",
    "BP+Gluc",
    "Adip+Trig+Gluc+HDL",
    "Adip+BP",
    "BP+HDL",
    "Trig+BP+HDL",
    "Adip+BP+Gluc",
    "Adip+Trig+BP",
    "Trig+BP+Gluc",
    "Trig+Gluc+HDL",
    "Adip+BP+HDL",
    "Adip+Trig+BP+HDL",
    "Gluc+HDL",
    "Adip+Trig+BP+Gluc",
    "Trig+BP+Gluc+HDL",
    "BP+Gluc+HDL",
    "Adip+Trig+BP+Gluc+HDL",
    "Adip+BP+Gluc+HDL",
]


# =====================================================================
# 2. OUTPUT DIRECTORY
# =====================================================================

if FRESH_RUN and OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
STAN_DIR.mkdir(parents=True, exist_ok=True)
FULL_DRAW_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 3. DATA VALIDATION
# =====================================================================

def validate_data() -> None:
    assert M.shape == (32, 5)
    assert BAG.shape == (32,)
    assert N_COUNTS.shape == (32,)
    assert np.isin(M, [0, 1]).all()
    assert np.isfinite(BAG).all()
    assert N_TOTAL == 27375
    assert (M[0] == 0).all()

    # Exact Table S7 audit checks.
    assert np.array_equal(
        N_COUNTS,
        TABLE_S7_N_COUNTS_REFERENCE,
    ), "Participant counts do not match Table S7."

    assert np.array_equal(
        BAG,
        TABLE_S7_BAG_LS_MEAN_REFERENCE,
    ), "BAG LS Mean values do not match Table S7."

    expected_component_counts = np.asarray(
        [6770, 18298, 3833, 9834, 9977],
        dtype=int,
    )

    assert np.array_equal(
        N_COUNTS @ M,
        expected_component_counts,
    )

    expected_by_number = {
        0: 4850,
        1: 7955,
        2: 6770,
        3: 4710,
        4: 2363,
        5: 727,
    }

    actual_by_number = {
        j: int(
            N_COUNTS[
                M.sum(axis=1) == j
            ].sum()
        )
        for j in range(6)
    }

    assert actual_by_number == expected_by_number


validate_data()

TABLE_S7_EMBEDDED_DATA = pd.DataFrame(
    {
        "constellation_id": CONSTELLATION_ID,
        "constellation": TABLE_S7_CONSTELLATION_LABELS,
        "participant_count": N_COUNTS,
        "BAG_LS_mean_years": BAG,
        **{
            COMPONENT_NAMES[j]: M[:, j]
            for j in range(5)
        },
    }
)

TABLE_S7_EMBEDDED_DATA.to_csv(
    OUTPUT_DIR / "table_s7_embedded_data_exact.csv",
    index=False,
)

print("\nExact Table S7 data validation: PASS")
print("  32/32 participant counts match embedded Table S7 reference.")
print("  32/32 BAG LS Mean values match embedded Table S7 reference.")
print("  BAG values are the Table S7 LS means, not beta coefficients.")


# =====================================================================
# 4. PARTICIPANT WEIGHTS
# =====================================================================

def participant_weights(
    indices: np.ndarray,
) -> np.ndarray:
    """
    Normalize counts to mean weight 1 within the current fitted dataset.

    If G=len(indices):
        w_i = G*n_i/sum(n_j)
        sum(w_i)=G.
    """
    idx = np.asarray(indices, dtype=int)
    counts = N_COUNTS[idx].astype(float)
    G = idx.size

    w = G * counts / counts.sum()

    if not np.isclose(w.sum(), float(G), rtol=0, atol=1e-12):
        raise RuntimeError("Participant weights do not sum to G.")

    return w


ALL_IDX = np.arange(32, dtype=int)
FULL_WEIGHTS = participant_weights(ALL_IDX)

PARTICIPANT_WEIGHT_TABLE = pd.DataFrame(
    {
        "constellation_id": CONSTELLATION_ID,
        "participant_count": N_COUNTS,
        "full_fit_weight": FULL_WEIGHTS,
        "weight_fraction_of_total": FULL_WEIGHTS / FULL_WEIGHTS.sum(),
        "BAG": BAG,
        **{
            COMPONENT_NAMES[j]: M[:, j]
            for j in range(5)
        },
    }
)

PARTICIPANT_WEIGHT_TABLE.to_csv(
    OUTPUT_DIR / "participant_weights.csv",
    index=False,
)

print("\nParticipant-count weighting:")
print(f"  total participants = {N_TOTAL}")
print(f"  sum of full-data weights = {FULL_WEIGHTS.sum():.12f}")
print(f"  mean full-data weight = {FULL_WEIGHTS.mean():.12f}")
print(f"  minimum weight = {FULL_WEIGHTS.min():.6f}")
print(f"  maximum weight = {FULL_WEIGHTS.max():.6f}")
print(
    "  max/min influence ratio = "
    f"{FULL_WEIGHTS.max()/FULL_WEIGHTS.min():.2f}"
)


# =====================================================================
# 5. SMOOTH BRANCHLESS NORMALIZED EXPONENTIAL-POWER FUNCTION
# =====================================================================

# Exact target family:
#
#                    exp(a*x^p)-1
#   H(x,p,a) =       ------------
#                      exp(a)-1
#
# A numerically stable sign-specific representation exists, but branching
# on the sign of parameter a triggers Stan's parameter-dependent control-
# flow warning.
#
# We use b = sqrt(a^2 + eps^2), a smooth approximation to |a|:
#
# H ≈ exp[-(a+b)(1-x^p)/2]
#      * expm1(-b*x^p) / expm1(-b)
#
# Every exponential argument is non-positive. It is exact at x=0 and x=1
# and differs from the exact family only in an infinitesimal neighborhood
# around a=0 controlled by eps=1e-12.

STAN_FUNCTIONS = f"""
functions {{
  real normalized_exp_power(real x, real p, real a) {{
    real xp = pow(x, p);
    real b = sqrt(square(a) + {SMOOTH_A_EPS:.17g} * {SMOOTH_A_EPS:.17g});
    real expo = -0.5 * (a + b) * (1.0 - xp);
    return
      exp(expo)
      * expm1(-b * xp)
      / expm1(-b);
  }}
}}
"""


# =====================================================================
# 6. COMMON STAN DATA BLOCK
# =====================================================================

STAN_DATA = r"""
data {
  int<lower=1> G;
  matrix[G,5] M;
  vector[G] bag;
  vector<lower=0>[G] obs_weight;
}
"""


# =====================================================================
# 7. SEVEN WEIGHTED STAN MODELS
# =====================================================================

# k and sigma are logically positive, so use Stan's native lower-bound
# constraints. Stan handles the unconstraining transform and Jacobian
# internally. This avoids the pedantic "2 priors" warning caused by the
# earlier hand-written log transform + Jacobian terms.
#
# The priors below are half-Normal(0,5) for k and half-Student-t(4,0,1)
# for sigma, up to constants that do not depend on model parameters.

STAN_LINEAR = (
    STAN_DATA
    +
r"""
parameters {
  simplex[5] mets;
  real<lower=0> k;
  real m;
  real<lower=0> sigma;
}
transformed parameters {
  vector[G] x = M * mets;
  vector[G] mu = rep_vector(m, G) + k*x;
}
model {
  mets ~ dirichlet(rep_vector(1.0, 5));
  m ~ normal(0, 3);
  k ~ normal(0, 5);
  sigma ~ student_t(4, 0, 1);

  for (g in 1:G)
    target +=
      obs_weight[g]
      * normal_lpdf(bag[g] | mu[g], sigma);
}
"""
)


STAN_POWER = (
    STAN_DATA
    +
r"""
parameters {
  simplex[5] mets;
  real<lower=0> k;
  real m;
  real log_p;
  real<lower=0> sigma;
}
transformed parameters {
  real p = exp(log_p);
  vector[G] x = M * mets;
  vector[G] mu;

  for (g in 1:G)
    mu[g] = m + k*pow(x[g], p);
}
model {
  mets ~ dirichlet(rep_vector(1.0, 5));
  m ~ normal(0, 3);
  log_p ~ normal(0, 0.5);
  k ~ normal(0, 5);
  sigma ~ student_t(4, 0, 1);

  for (g in 1:G)
    target +=
      obs_weight[g]
      * normal_lpdf(bag[g] | mu[g], sigma);
}
"""
)


STAN_EXP = (
    STAN_FUNCTIONS
    +
    STAN_DATA
    +
r"""
parameters {
  simplex[5] mets;
  real<lower=0> k;
  real m;
  real a;
  real<lower=0> sigma;
}
transformed parameters {
  real p = 1.0;
  vector[G] x = M * mets;
  vector[G] mu;

  for (g in 1:G)
    mu[g] =
      m
      + k*normalized_exp_power(x[g], p, a);
}
model {
  mets ~ dirichlet(rep_vector(1.0, 5));
  m ~ normal(0, 3);
  a ~ normal(0, 1);
  k ~ normal(0, 5);
  sigma ~ student_t(4, 0, 1);

  for (g in 1:G)
    target +=
      obs_weight[g]
      * normal_lpdf(bag[g] | mu[g], sigma);
}
"""
)


STAN_HYBRID = (
    STAN_FUNCTIONS
    +
    STAN_DATA
    +
r"""
parameters {
  simplex[5] mets;
  real<lower=0> k;
  real m;
  real log_p;
  real a;
  real<lower=0> sigma;
}
transformed parameters {
  real p = exp(log_p);
  vector[G] x = M * mets;
  vector[G] mu;

  for (g in 1:G)
    mu[g] =
      m
      + k*normalized_exp_power(x[g], p, a);
}
model {
  mets ~ dirichlet(rep_vector(1.0, 5));
  m ~ normal(0, 3);
  log_p ~ normal(0, 0.5);
  a ~ normal(0, 1);
  k ~ normal(0, 5);
  sigma ~ student_t(4, 0, 1);

  for (g in 1:G)
    target +=
      obs_weight[g]
      * normal_lpdf(bag[g] | mu[g], sigma);
}
"""
)


# Direct p-ridge:
#
#     p = alpha_p + beta_p*a
#
# p must be positive because x^p is part of the scientific model.
# Sample p directly with its logical lower bound and derive a:
#
#     a = (p-alpha_p)/beta_p
#
# This preserves the exact linear ridge and avoids an arbitrary finite
# bound on a.  If the intended prior is a ~ Normal(0,1), changing
# variables from p to a contributes |da/dp| = 1/|beta_p|.  beta_p is
# data within each fit, so this factor is constant with respect to the
# sampled parameters and can be omitted without changing the posterior.
# Stan supplies the usual lower-bound Jacobian for p automatically.

STAN_DIRECT_RIDGE = (
    STAN_FUNCTIONS
    +
r"""
data {
  int<lower=1> G;
  matrix[G,5] M;
  vector[G] bag;
  vector<lower=0>[G] obs_weight;
  real alpha_p;
  real beta_p;
}
parameters {
  simplex[5] mets;
  real<lower=0> k;
  real m;
  real<lower=0> p;
  real<lower=0> sigma;
}
transformed parameters {
  real log_p = log(p);
  real a = (p - alpha_p) / beta_p;
  vector[G] x = M * mets;
  vector[G] mu;

  for (g in 1:G)
    mu[g] =
      m
      + k*normalized_exp_power(x[g], p, a);
}
model {
  mets ~ dirichlet(rep_vector(1.0, 5));
  m ~ normal(0, 3);
  target += normal_lpdf(a | 0, 1);
  k ~ normal(0, 5);
  sigma ~ student_t(4, 0, 1);

  for (g in 1:G)
    target +=
      obs_weight[g]
      * normal_lpdf(bag[g] | mu[g], sigma);
}
"""
)


STAN_HARD_LOGP_RIDGE = (
    STAN_FUNCTIONS
    +
r"""
data {
  int<lower=1> G;
  matrix[G,5] M;
  vector[G] bag;
  vector<lower=0>[G] obs_weight;
  real alpha_logp;
  real beta_logp;
}
parameters {
  simplex[5] mets;
  real<lower=0> k;
  real m;
  real a;
  real<lower=0> sigma;
}
transformed parameters {
  real log_p = alpha_logp + beta_logp*a;
  real p = exp(log_p);
  vector[G] x = M * mets;
  vector[G] mu;

  for (g in 1:G)
    mu[g] =
      m
      + k*normalized_exp_power(x[g], p, a);
}
model {
  mets ~ dirichlet(rep_vector(1.0, 5));
  m ~ normal(0, 3);
  a ~ normal(0, 1);
  k ~ normal(0, 5);
  sigma ~ student_t(4, 0, 1);

  for (g in 1:G)
    target +=
      obs_weight[g]
      * normal_lpdf(bag[g] | mu[g], sigma);
}
"""
)


STAN_SOFT_LOGP_RIDGE = (
    STAN_FUNCTIONS
    +
r"""
data {
  int<lower=1> G;
  matrix[G,5] M;
  vector[G] bag;
  vector<lower=0>[G] obs_weight;
  real alpha_logp;
  real beta_logp;
  real<lower=0> tau_logp;
}
parameters {
  simplex[5] mets;
  real<lower=0> k;
  real m;
  real a;
  real z;
  real<lower=0> sigma;
}
transformed parameters {
  real log_p =
    alpha_logp
    + beta_logp*a
    + tau_logp*z;
  real p = exp(log_p);
  vector[G] x = M * mets;
  vector[G] mu;

  for (g in 1:G)
    mu[g] =
      m
      + k*normalized_exp_power(x[g], p, a);
}
model {
  mets ~ dirichlet(rep_vector(1.0, 5));
  m ~ normal(0, 3);
  a ~ normal(0, 1);
  z ~ normal(0, 1);
  k ~ normal(0, 5);
  sigma ~ student_t(4, 0, 1);

  for (g in 1:G)
    target +=
      obs_weight[g]
      * normal_lpdf(bag[g] | mu[g], sigma);
}
"""
)


MODEL_CODE = {
    "Linear": STAN_LINEAR,
    "Power": STAN_POWER,
    "Normalized_exponential": STAN_EXP,
    "Power_exponential_hybrid": STAN_HYBRID,
    "Direct_p_ridge": STAN_DIRECT_RIDGE,
    "Hard_logp_ridge": STAN_HARD_LOGP_RIDGE,
    "Soft_logp_ridge": STAN_SOFT_LOGP_RIDGE,
}

MODEL_TYPE = {
    "Linear": "linear",
    "Power": "power",
    "Normalized_exponential": "exponential",
    "Power_exponential_hybrid": "hybrid",
    "Direct_p_ridge": "direct_ridge",
    "Hard_logp_ridge": "hard_logp_ridge",
    "Soft_logp_ridge": "soft_logp_ridge",
}

for name, code in MODEL_CODE.items():
    (STAN_DIR / f"{name.lower()}.stan").write_text(
        code,
        encoding="utf-8",
    )


# =====================================================================
# 8. STAN DATA + SAMPLING HELPERS
# =====================================================================

def make_data(
    indices: np.ndarray,
    *,
    extra: dict[str, float] | None = None,
) -> dict[str, Any]:
    idx = np.asarray(indices, dtype=int)

    out: dict[str, Any] = {
        "G": int(idx.size),
        "M": M[idx].astype(float).tolist(),
        "bag": BAG[idx].astype(float).tolist(),
        "obs_weight": participant_weights(idx).tolist(),
    }

    if extra:
        out.update(extra)

    return out


def fit_stan(
    model_name: str,
    data: dict[str, Any],
    *,
    seed: int,
    chains: int,
    samples: int,
    warmup: int,
    quiet: bool,
) -> Any:
    if not quiet:
        print("\n" + "=" * 74)
        print(f"BUILD/FIT: {model_name}")
        print("=" * 74)

    model = stan.build(
        MODEL_CODE[model_name],
        data=data,
        random_seed=seed,
    )

    fit = model.sample(
        num_chains=chains,
        num_samples=samples,
        num_warmup=warmup,
        delta=ADAPT_DELTA,
        max_depth=MAX_TREEDEPTH,
        refresh=(
            0
            if quiet
            else max(
                1,
                (samples + warmup)//10,
            )
        ),
    )

    return fit


def scalar_draws(
    fit: Any,
    name: str,
) -> np.ndarray:
    return np.asarray(
        fit[name],
        dtype=float,
    ).reshape(-1)


def mets_draws(
    fit: Any,
) -> np.ndarray:
    raw = np.asarray(
        fit["mets"],
        dtype=float,
    )

    if raw.ndim != 2 or raw.shape[0] != 5:
        raise RuntimeError(
            f"Unexpected mets shape: {raw.shape}"
        )

    return raw.T.copy()


def extract_draws(
    fit: Any,
    model_name: str,
) -> dict[str, Any]:
    t = MODEL_TYPE[model_name]

    out: dict[str, Any] = {
        "model": model_name,
        "type": t,
        "mets": mets_draws(fit),
        "m": scalar_draws(fit, "m"),
        "k": scalar_draws(fit, "k"),
        "sigma": scalar_draws(fit, "sigma"),
    }

    if t == "power":
        out["log_p"] = scalar_draws(fit, "log_p")
        out["p"] = scalar_draws(fit, "p")

    elif t == "exponential":
        out["a"] = scalar_draws(fit, "a")
        out["p"] = np.ones_like(out["a"])

    elif t in {
        "hybrid",
        "direct_ridge",
        "hard_logp_ridge",
        "soft_logp_ridge",
    }:
        out["log_p"] = scalar_draws(fit, "log_p")
        out["p"] = scalar_draws(fit, "p")
        out["a"] = scalar_draws(fit, "a")

        if t == "soft_logp_ridge":
            out["z"] = scalar_draws(fit, "z")

    return out


# =====================================================================
# 9. MCMC DIAGNOSTICS
# =====================================================================

def fit_event_dims(
    fit: Any,
    name: str,
) -> tuple[int, ...]:
    if name in fit.param_names:
        i = fit.param_names.index(name)
        return tuple(
            int(v)
            for v in fit.dims[i]
        )
    return ()


def chain_draw_array(
    fit: Any,
    name: str,
) -> np.ndarray:
    raw = np.asarray(fit[name])
    chains = int(fit.num_chains)
    draws = int(
        math.ceil(
            fit.num_samples
            / fit.num_thin
        )
    )

    event_dims = fit_event_dims(
        fit,
        name,
    )

    if event_dims:
        arr = raw.reshape(
            *event_dims,
            -1,
        )
        arr = arr.reshape(
            *event_dims,
            draws,
            chains,
        )

        n_event = len(event_dims)
        axes = (
            n_event + 1,
            n_event,
            *range(n_event),
        )

        return np.transpose(
            arr,
            axes,
        ).copy()

    return raw.reshape(
        draws,
        chains,
    ).T.copy()


def diagnostic_names(
    model_name: str,
) -> list[str]:
    t = MODEL_TYPE[model_name]

    names = [
        "mets",
        "m",
        "k",
        "sigma",
    ]

    if t == "power":
        names += [
            "log_p",
            "p",
        ]

    elif t == "exponential":
        names += ["a"]

    elif t in {
        "hybrid",
        "direct_ridge",
        "hard_logp_ridge",
    }:
        names += [
            "log_p",
            "p",
            "a",
        ]

    elif t == "soft_logp_ridge":
        names += [
            "log_p",
            "p",
            "a",
            "z",
        ]

    return names


def fit_diagnostics(
    fit: Any,
    model_name: str,
) -> dict[str, float | int]:
    names = diagnostic_names(
        model_name
    )

    posterior = {
        name: chain_draw_array(
            fit,
            name,
        )
        for name in names
    }

    idata = az.from_dict(
        {
            "posterior":
                posterior
        },
        sample_dims=[
            "chain",
            "draw",
        ],
        coords={
            "component":
                COMPONENT_NAMES
        },
        dims={
            "mets":
                ["component"]
        },
    )

    sm = az.summary(
        idata,
        var_names=names,
        kind="diagnostics",
        round_to=None,
    )

    available = set(
        getattr(
            fit,
            "sample_and_sampler_param_names",
            (),
        )
    )

    divergences = -1
    treedepth_hits = -1

    if "divergent__" in available:
        divergences = int(
            np.sum(
                np.asarray(
                    fit["divergent__"]
                )
            )
        )

    if "treedepth__" in available:
        treedepth_hits = int(
            np.sum(
                np.asarray(
                    fit["treedepth__"]
                )
                >= MAX_TREEDEPTH
            )
        )

    return {
        "max_Rhat":
            float(
                sm["r_hat"].max()
            ),

        "min_ESS_bulk":
            float(
                sm["ess_bulk"].min()
            ),

        "min_ESS_tail":
            float(
                sm["ess_tail"].min()
            ),

        "divergences":
            divergences,

        "max_treedepth_hits":
            treedepth_hits,
    }


# =====================================================================
# 10. RIDGE REGRESSION
# =====================================================================

def regression(
    y: np.ndarray,
    x: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(
        y,
        dtype=float,
    )
    x = np.asarray(
        x,
        dtype=float,
    )

    X = np.column_stack(
        [
            np.ones_like(x),
            x,
        ]
    )

    coef, _, _, _ = (
        np.linalg.lstsq(
            X,
            y,
            rcond=None,
        )
    )

    fitted = X @ coef
    residual = y - fitted

    sse = float(
        np.sum(
            residual**2
        )
    )

    sst = float(
        np.sum(
            (y-y.mean())**2
        )
    )

    return {
        "alpha":
            float(coef[0]),

        "beta":
            float(coef[1]),

        "tau":
            float(
                np.std(
                    residual,
                    ddof=2,
                )
            ),

        "r2":
            (
                float(1-sse/sst)
                if sst > 0
                else np.nan
            ),

        "corr":
            float(
                np.corrcoef(
                    x,
                    y,
                )[0,1]
            ),
    }


def estimate_ridge(
    hybrid_draws: dict[str, Any],
) -> dict[str, float]:
    direct = regression(
        hybrid_draws["p"],
        hybrid_draws["a"],
    )

    logrel = regression(
        hybrid_draws["log_p"],
        hybrid_draws["a"],
    )

    if abs(direct["beta"]) < 1e-6:
        raise RuntimeError(
            "Direct p-ridge beta is too close "
            "to zero for stable reparameterization."
        )

    return {
        "alpha_p":
            direct["alpha"],

        "beta_p":
            direct["beta"],

        "tau_p":
            direct["tau"],

        "r2_p":
            direct["r2"],

        "corr_p_a":
            direct["corr"],

        "alpha_logp":
            logrel["alpha"],

        "beta_logp":
            logrel["beta"],

        "tau_logp":
            max(
                logrel["tau"],
                np.finfo(float).tiny,
            ),

        "r2_logp":
            logrel["r2"],

        "corr_logp_a":
            logrel["corr"],
    }


def ridge_extra_data(
    model_name: str,
    relation: dict[str, float],
) -> dict[str, float]:
    if model_name == "Direct_p_ridge":
        return {
            "alpha_p":
                relation["alpha_p"],

            "beta_p":
                relation["beta_p"],
        }

    if model_name == "Hard_logp_ridge":
        return {
            "alpha_logp":
                relation["alpha_logp"],

            "beta_logp":
                relation["beta_logp"],
        }

    if model_name == "Soft_logp_ridge":
        return {
            "alpha_logp":
                relation["alpha_logp"],

            "beta_logp":
                relation["beta_logp"],

            "tau_logp":
                relation["tau_logp"],
        }

    return {}


# =====================================================================
# 11. NUMPY RESPONSE FUNCTION
# =====================================================================

def normalized_exp_power_numpy(
    x: np.ndarray,
    p: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    x = np.asarray(
        x,
        dtype=float,
    )
    p = np.asarray(
        p,
        dtype=float,
    )
    a = np.asarray(
        a,
        dtype=float,
    )

    xp = np.power(
        x,
        p,
    )

    b = np.sqrt(
        a*a
        + SMOOTH_A_EPS**2
    )

    expo = (
        -0.5
        * (a+b)
        * (1.0-xp)
    )

    return (
        np.exp(expo)
        * np.expm1(-b*xp)
        / np.expm1(-b)
    )


# =====================================================================
# 12. DRAW-BY-DRAW PREDICTIONS
# =====================================================================

def mu_for_patterns(
    draws: dict[str, Any],
    patterns: np.ndarray,
) -> np.ndarray:
    """
    Returns array shape:
        posterior draws x patterns
    """
    patterns = np.asarray(
        patterns,
        dtype=float,
    )

    x = (
        draws["mets"]
        @ patterns.T
    )

    t = draws["type"]

    if t == "linear":
        return (
            draws["m"][:,None]
            + draws["k"][:,None]*x
        )

    if t == "power":
        return (
            draws["m"][:,None]
            + draws["k"][:,None]
            * np.power(
                x,
                draws["p"][:,None],
            )
        )

    shape = normalized_exp_power_numpy(
        x,
        draws["p"][:,None],
        draws["a"][:,None],
    )

    return (
        draws["m"][:,None]
        + draws["k"][:,None]*shape
    )


def normal_logpdf(
    y: float,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    return (
        -0.5*math.log(2*math.pi)
        - np.log(sigma)
        - 0.5*((y-mu)/sigma)**2
    )


def log_mean_exp(
    values: np.ndarray,
) -> float:
    values = np.asarray(
        values,
        dtype=float,
    )

    return float(
        logsumexp(values)
        - math.log(values.size)
    )


def heldout_prediction(
    draws: dict[str, Any],
    hold_idx: int,
) -> dict[str, float]:
    mu = mu_for_patterns(
        draws,
        M[[hold_idx]],
    )[:,0]

    sigma = draws["sigma"]

    lpd = log_mean_exp(
        normal_logpdf(
            float(BAG[hold_idx]),
            mu,
            sigma,
        )
    )

    eval_weight = float(
        FULL_WEIGHTS[hold_idx]
    )

    return {
        "exact_lpd_unweighted":
            lpd,

        "evaluation_participant_weight":
            eval_weight,

        "exact_lpd_participant_weighted":
            eval_weight*lpd,

        "pred_mean":
            float(
                np.mean(mu)
            ),

        "pred_median":
            float(
                np.median(mu)
            ),

        "pred_q2.5":
            float(
                np.quantile(
                    mu,
                    0.025,
                )
            ),

        "pred_q97.5":
            float(
                np.quantile(
                    mu,
                    0.975,
                )
            ),
    }


# =====================================================================
# 13. FULL-DATA FITS
# =====================================================================

FULL_DRAWS: dict[
    str,
    dict[str, Any],
] = {}

FULL_DIAG_ROWS: list[
    dict[str, Any]
] = []


for j, name in enumerate(
    [
        "Linear",
        "Power",
        "Normalized_exponential",
        "Power_exponential_hybrid",
    ]
):
    fit = fit_stan(
        name,
        make_data(ALL_IDX),
        seed=SEED_BASE+j,
        chains=FULL_CHAINS,
        samples=FULL_NUM_SAMPLES,
        warmup=FULL_NUM_WARMUP,
        quiet=False,
    )

    draws = extract_draws(
        fit,
        name,
    )

    diag = fit_diagnostics(
        fit,
        name,
    )

    FULL_DRAWS[name] = draws

    FULL_DIAG_ROWS.append(
        {
            "model":
                name,

            **diag,
        }
    )

    np.savez_compressed(
        FULL_DRAW_DIR
        / f"{name.lower()}.npz",
        **{
            key: value
            for key, value in draws.items()
            if isinstance(
                value,
                np.ndarray,
            )
        },
    )

    del fit


FULL_RIDGE_RELATION = (
    estimate_ridge(
        FULL_DRAWS[
            "Power_exponential_hybrid"
        ]
    )
)

pd.DataFrame(
    [FULL_RIDGE_RELATION]
).to_csv(
    OUTPUT_DIR
    / "full_data_ridge_relation.csv",
    index=False,
)


for j, name in enumerate(
    [
        "Direct_p_ridge",
        "Hard_logp_ridge",
        "Soft_logp_ridge",
    ]
):
    fit = fit_stan(
        name,
        make_data(
            ALL_IDX,
            extra=ridge_extra_data(
                name,
                FULL_RIDGE_RELATION,
            ),
        ),
        seed=SEED_BASE+100+j,
        chains=FULL_CHAINS,
        samples=FULL_NUM_SAMPLES,
        warmup=FULL_NUM_WARMUP,
        quiet=False,
    )

    draws = extract_draws(
        fit,
        name,
    )

    diag = fit_diagnostics(
        fit,
        name,
    )

    FULL_DRAWS[name] = draws

    FULL_DIAG_ROWS.append(
        {
            "model":
                name,

            **diag,
        }
    )

    np.savez_compressed(
        FULL_DRAW_DIR
        / f"{name.lower()}.npz",
        **{
            key: value
            for key, value in draws.items()
            if isinstance(
                value,
                np.ndarray,
            )
        },
    )

    del fit


FULL_DIAGNOSTICS = (
    pd.DataFrame(
        FULL_DIAG_ROWS
    )
)

FULL_DIAGNOSTICS.to_csv(
    OUTPUT_DIR
    / "full_data_mcmc_diagnostics.csv",
    index=False,
)


# =====================================================================
# 14. FULL-DATA PARAMETER SUMMARIES
# =====================================================================

def summarize(
    values: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean":
            float(
                np.mean(values)
            ),

        "sd":
            float(
                np.std(
                    values,
                    ddof=1,
                )
            ),

        "q2.5":
            float(
                np.quantile(
                    values,
                    0.025,
                )
            ),

        "median":
            float(
                np.median(values)
            ),

        "q97.5":
            float(
                np.quantile(
                    values,
                    0.975,
                )
            ),
    }


weight_rows = []
parameter_rows = []

for name in MODEL_NAMES:
    d = FULL_DRAWS[name]

    for j, factor in enumerate(
        COMPONENT_NAMES
    ):
        weight_rows.append(
            {
                "model":
                    name,

                "factor":
                    factor,

                **summarize(
                    d["mets"][:,j]
                ),
            }
        )

    row: dict[str, Any] = {
        "model":
            name
    }

    for par in [
        "m",
        "k",
        "sigma",
        "p",
        "a",
        "log_p",
        "z",
    ]:
        if par in d:
            for stat, value in (
                summarize(
                    d[par]
                ).items()
            ):
                row[
                    f"{par}_{stat}"
                ] = value

    if "p" in d:
        row["P_p_gt_1"] = float(
            np.mean(
                d["p"] > 1
            )
        )

    if "a" in d:
        row["P_a_gt_0"] = float(
            np.mean(
                d["a"] > 0
            )
        )

    parameter_rows.append(
        row
    )


FULL_WEIGHT_SUMMARY = (
    pd.DataFrame(
        weight_rows
    )
)

FULL_PARAMETER_SUMMARY = (
    pd.DataFrame(
        parameter_rows
    )
)

FULL_WEIGHT_SUMMARY.to_csv(
    OUTPUT_DIR
    / "full_data_weight_summary.csv",
    index=False,
)

FULL_PARAMETER_SUMMARY.to_csv(
    OUTPUT_DIR
    / "full_data_parameter_summary.csv",
    index=False,
)


# =====================================================================
# 15. BP-ONLY POSTERIOR PREDICTIONS
# =====================================================================

BP_ONLY = np.asarray(
    [[0,1,0,0,0]],
    dtype=int,
)

bp_rows = []

for name in MODEL_NAMES:
    d = FULL_DRAWS[name]

    mu = mu_for_patterns(
        d,
        BP_ONLY,
    )[:,0]

    x = d["mets"][:,1]

    bp_rows.append(
        {
            "model":
                name,

            "x_mean":
                float(np.mean(x)),

            "x_q2.5":
                float(
                    np.quantile(
                        x,
                        0.025,
                    )
                ),

            "x_median":
                float(
                    np.median(x)
                ),

            "x_q97.5":
                float(
                    np.quantile(
                        x,
                        0.975,
                    )
                ),

            "BAG_mean":
                float(
                    np.mean(mu)
                ),

            "BAG_q2.5":
                float(
                    np.quantile(
                        mu,
                        0.025,
                    )
                ),

            "BAG_median":
                float(
                    np.median(mu)
                ),

            "BAG_q97.5":
                float(
                    np.quantile(
                        mu,
                        0.975,
                    )
                ),
        }
    )


pd.DataFrame(
    bp_rows
).to_csv(
    OUTPUT_DIR
    / "blood_pressure_only_predictions.csv",
    index=False,
)


# =====================================================================
# 16. POSTERIOR CURVES
# =====================================================================

curve_rows = []

x_grid = np.linspace(
    0,
    1,
    X_GRID_N,
)

for name in MODEL_NAMES:
    d = FULL_DRAWS[name]
    S = d["m"].size

    for x_value in x_grid:
        x = np.full(
            S,
            x_value,
            dtype=float,
        )

        if d["type"] == "linear":
            y = (
                d["m"]
                + d["k"]*x
            )

        elif d["type"] == "power":
            y = (
                d["m"]
                + d["k"]
                * np.power(
                    x,
                    d["p"],
                )
            )

        else:
            y = (
                d["m"]
                + d["k"]
                * normalized_exp_power_numpy(
                    x,
                    d["p"],
                    d["a"],
                )
            )

        curve_rows.append(
            {
                "model":
                    name,

                "x":
                    x_value,

                "mean":
                    float(
                        np.mean(y)
                    ),

                "q2.5":
                    float(
                        np.quantile(
                            y,
                            0.025,
                        )
                    ),

                "median":
                    float(
                        np.median(y)
                    ),

                "q97.5":
                    float(
                        np.quantile(
                            y,
                            0.975,
                        )
                    ),
            }
        )


POSTERIOR_CURVES = (
    pd.DataFrame(
        curve_rows
    )
)

POSTERIOR_CURVES.to_csv(
    OUTPUT_DIR
    / "posterior_curves.csv",
    index=False,
)


# =====================================================================
# 17. CACHE HELPERS
# =====================================================================

def settings_hash() -> str:
    payload = {
        "program_version":
            1,

        "weight_rule":
            "G*n_i/sum_training_n",

        "evaluation_weight_rule":
            "32*n_i/27375",

        "full":
            [
                FULL_CHAINS,
                FULL_NUM_SAMPLES,
                FULL_NUM_WARMUP,
            ],

        "cv":
            [
                CV_CHAINS,
                CV_NUM_SAMPLES,
                CV_NUM_WARMUP,
            ],

        "adapt_delta":
            ADAPT_DELTA,

        "max_treedepth":
            MAX_TREEDEPTH,

        "smooth_a_eps":
            SMOOTH_A_EPS,

        "codes": {
            name:
                hashlib.sha256(
                    code.encode(
                        "utf-8"
                    )
                ).hexdigest()
            for name, code
            in MODEL_CODE.items()
        },
    }

    return (
        hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
            ).encode(
                "utf-8"
            )
        )
        .hexdigest()[:16]
    )


RUN_HASH = settings_hash()


def fold_cache_path(
    model_name: str,
    hold_idx: int,
) -> Path:
    return (
        CACHE_DIR
        /
        (
            f"{model_name.lower()}"
            f"_fold{hold_idx+1:02d}"
            f"_{RUN_HASH}.npz"
        )
    )


def ridge_cache_path(
    hold_idx: int,
) -> Path:
    return (
        CACHE_DIR
        /
        (
            "ridge_relation"
            f"_fold{hold_idx+1:02d}"
            f"_{RUN_HASH}.npz"
        )
    )


def save_scalar_npz(
    path: Path,
    values: dict[str, Any],
) -> None:
    np.savez_compressed(
        path,
        **{
            key:
                np.asarray(value)
            for key, value
            in values.items()
        },
    )


def load_scalar_npz(
    path: Path,
) -> dict[str, Any]:
    z = np.load(
        path,
        allow_pickle=False,
    )

    return {
        key:
            (
                z[key].item()
                if z[key].ndim == 0
                else z[key]
            )
        for key in z.files
    }


# =====================================================================
# 18. EXACT LOO
# =====================================================================

LOO_ROWS: list[
    dict[str, Any]
] = []

RIDGE_FOLD_ROWS: list[
    dict[str, Any]
] = []


def fit_one_loo(
    model_name: str,
    hold_idx: int,
    train_idx: np.ndarray,
    relation: dict[str, float] | None,
) -> tuple[
    dict[str, Any],
    dict[str, float] | None,
]:
    cache = fold_cache_path(
        model_name,
        hold_idx,
    )

    # FRESH_RUN deletes these before fitting.
    # This branch is useful if a user later changes FRESH_RUN=False
    # to resume an interrupted run of THIS exact program.
    if (
        cache.exists()
        and
        not FRESH_RUN
    ):
        row = load_scalar_npz(
            cache
        )

        learned = None

        if (
            model_name
            ==
            "Power_exponential_hybrid"
        ):
            rp = ridge_cache_path(
                hold_idx
            )

            if rp.exists():
                raw = load_scalar_npz(
                    rp
                )

                learned = {
                    key:
                        float(value)
                    for key, value
                    in raw.items()
                    if key
                    != "constellation_id"
                }

        return row, learned

    extra = None

    if model_name in RIDGE_MODEL_NAMES:
        if relation is None:
            raise RuntimeError(
                f"{model_name} requires "
                "training-only ridge relation."
            )

        extra = ridge_extra_data(
            model_name,
            relation,
        )

    fit = fit_stan(
        model_name,
        make_data(
            train_idx,
            extra=extra,
        ),
        seed=(
            SEED_BASE
            + 1_000_000
            + 10_000
            * MODEL_NAMES.index(
                model_name
            )
            + hold_idx
        ),
        chains=CV_CHAINS,
        samples=CV_NUM_SAMPLES,
        warmup=CV_NUM_WARMUP,
        quiet=True,
    )

    draws = extract_draws(
        fit,
        model_name,
    )

    diag = fit_diagnostics(
        fit,
        model_name,
    )

    pred = heldout_prediction(
        draws,
        hold_idx,
    )

    train_weights = (
        participant_weights(
            train_idx
        )
    )

    row: dict[str, Any] = {
        "model":
            model_name,

        "constellation_id":
            int(
                hold_idx+1
            ),

        "heldout_BAG":
            float(
                BAG[hold_idx]
            ),

        "participant_count":
            int(
                N_COUNTS[
                    hold_idx
                ]
            ),

        "train_participant_total":
            int(
                N_COUNTS[
                    train_idx
                ].sum()
            ),

        "train_weight_sum":
            float(
                train_weights.sum()
            ),

        "heldout_full_weight":
            float(
                FULL_WEIGHTS[
                    hold_idx
                ]
            ),

        "number_of_components":
            int(
                M[
                    hold_idx
                ].sum()
            ),

        **{
            COMPONENT_NAMES[j]:
                int(
                    M[
                        hold_idx,
                        j,
                    ]
                )
            for j in range(5)
        },

        **pred,
        **diag,
    }

    learned = None

    if (
        model_name
        ==
        "Power_exponential_hybrid"
    ):
        learned = estimate_ridge(
            draws
        )

        save_scalar_npz(
            ridge_cache_path(
                hold_idx
            ),
            {
                "constellation_id":
                    int(
                        hold_idx+1
                    ),

                **learned,
            },
        )

    save_scalar_npz(
        cache,
        row,
    )

    del fit
    del draws

    return row, learned


print(
    "\n"
    + "="*76
    + "\nPARTICIPANT-WEIGHTED EXACT LOO\n"
    + "="*76
)

for hold_idx in range(32):
    print(
        f"\nOuter fold "
        f"{hold_idx+1:02d}/32"
    )

    train_idx = np.asarray(
        [
            i
            for i in range(32)
            if i != hold_idx
        ],
        dtype=int,
    )

    fold_relation = None

    for name in [
        "Linear",
        "Power",
        "Normalized_exponential",
        "Power_exponential_hybrid",
    ]:
        row, learned = (
            fit_one_loo(
                name,
                hold_idx,
                train_idx,
                relation=None,
            )
        )

        LOO_ROWS.append(
            row
        )

        print(
            f"  {name:28s}"
            f" lpd={row['exact_lpd_unweighted']:+.4f}"
            f" weighted={row['exact_lpd_participant_weighted']:+.4f}"
        )

        if (
            name
            ==
            "Power_exponential_hybrid"
        ):
            if learned is None:
                rp = ridge_cache_path(
                    hold_idx
                )

                if not rp.exists():
                    raise RuntimeError(
                        "Missing fold ridge relation."
                    )

                raw = load_scalar_npz(
                    rp
                )

                learned = {
                    key:
                        float(value)
                    for key, value
                    in raw.items()
                    if key
                    != "constellation_id"
                }

            fold_relation = learned

            RIDGE_FOLD_ROWS.append(
                {
                    "constellation_id":
                        int(
                            hold_idx+1
                        ),

                    **fold_relation,
                }
            )

    assert fold_relation is not None

    for name in [
        "Direct_p_ridge",
        "Hard_logp_ridge",
        "Soft_logp_ridge",
    ]:
        row, _ = fit_one_loo(
            name,
            hold_idx,
            train_idx,
            relation=fold_relation,
        )

        LOO_ROWS.append(
            row
        )

        print(
            f"  {name:28s}"
            f" lpd={row['exact_lpd_unweighted']:+.4f}"
            f" weighted={row['exact_lpd_participant_weighted']:+.4f}"
        )


EXACT_POINTWISE = (
    pd.DataFrame(
        LOO_ROWS
    )
    .sort_values(
        [
            "model",
            "constellation_id",
        ]
    )
    .reset_index(
        drop=True
    )
)

EXACT_POINTWISE.to_csv(
    OUTPUT_DIR
    / "exact_loo_pointwise.csv",
    index=False,
)

pd.DataFrame(
    RIDGE_FOLD_ROWS
).to_csv(
    OUTPUT_DIR
    / "ridge_relation_by_fold.csv",
    index=False,
)


# =====================================================================
# 19. MATRICES OF POINTWISE LOO SCORES
# =====================================================================

def pointwise_vector(
    model_name: str,
    column: str,
) -> np.ndarray:
    sub = (
        EXACT_POINTWISE[
            EXACT_POINTWISE[
                "model"
            ]
            ==
            model_name
        ]
        .sort_values(
            "constellation_id"
        )
    )

    if sub.shape[0] != 32:
        raise RuntimeError(
            f"{model_name}: "
            f"{sub.shape[0]} folds, expected 32."
        )

    return sub[
        column
    ].to_numpy(
        dtype=float,
    )


LPD_UNWEIGHTED = (
    np.column_stack(
        [
            pointwise_vector(
                name,
                "exact_lpd_unweighted",
            )
            for name
            in MODEL_NAMES
        ]
    )
)

LPD_WEIGHTED = (
    np.column_stack(
        [
            pointwise_vector(
                name,
                "exact_lpd_participant_weighted",
            )
            for name
            in MODEL_NAMES
        ]
    )
)

PRED_MATRIX = (
    np.column_stack(
        [
            pointwise_vector(
                name,
                "pred_mean",
            )
            for name
            in MODEL_NAMES
        ]
    )
)


# =====================================================================
# 20. MODEL COMPARISON
# =====================================================================

def total_se(
    pointwise: np.ndarray,
) -> float:
    pointwise = np.asarray(
        pointwise,
        dtype=float,
    )

    return float(
        math.sqrt(
            pointwise.size
            * np.var(
                pointwise,
                ddof=1,
            )
        )
    )


def make_comparison(
    L: np.ndarray,
    *,
    weighted: bool,
) -> pd.DataFrame:
    total = L.sum(axis=0)
    best_idx = int(
        np.argmax(total)
    )

    rows = []

    for j, name in enumerate(
        MODEL_NAMES
    ):
        diff = (
            L[:,j]
            -
            L[:,best_idx]
        )

        residual = (
            BAG
            -
            PRED_MATRIX[:,j]
        )

        if weighted:
            rmse = math.sqrt(
                float(
                    np.sum(
                        FULL_WEIGHTS
                        * residual**2
                    )
                    /
                    FULL_WEIGHTS.sum()
                )
            )

            mae = float(
                np.sum(
                    FULL_WEIGHTS
                    * np.abs(residual)
                )
                /
                FULL_WEIGHTS.sum()
            )

        else:
            rmse = float(
                np.sqrt(
                    np.mean(
                        residual**2
                    )
                )
            )

            mae = float(
                np.mean(
                    np.abs(residual)
                )
            )

        rows.append(
            {
                "model":
                    name,

                "criterion":
                    (
                        "participant_weighted"
                        if weighted
                        else
                        "equal_constellation"
                    ),

                "elpd":
                    float(
                        total[j]
                    ),

                "se_elpd_pointwise_approx":
                    total_se(
                        L[:,j]
                    ),

                "elpd_diff_vs_best":
                    float(
                        diff.sum()
                    ),

                "se_diff_pointwise_approx":
                    total_se(
                        diff
                    ),

                "rough_2x_se_diff_small_N":
                    2
                    * total_se(
                        diff
                    ),

                "RMSE":
                    rmse,

                "MAE":
                    mae,
            }
        )

    out = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "elpd",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    out.insert(
        0,
        "rank",
        np.arange(
            out.shape[0],
            dtype=int,
        ),
    )

    return out


WEIGHTED_COMPARISON = (
    make_comparison(
        LPD_WEIGHTED,
        weighted=True,
    )
)

UNWEIGHTED_COMPARISON = (
    make_comparison(
        LPD_UNWEIGHTED,
        weighted=False,
    )
)

WEIGHTED_COMPARISON.to_csv(
    OUTPUT_DIR
    / "exact_loo_model_comparison_PARTICIPANT_WEIGHTED_PRIMARY.csv",
    index=False,
)

UNWEIGHTED_COMPARISON.to_csv(
    OUTPUT_DIR
    / "exact_loo_model_comparison_equal_constellation_sensitivity.csv",
    index=False,
)


# =====================================================================
# 21. MODEL WEIGHTS
# =====================================================================

def stacking_weights(
    unweighted_log_lpd: np.ndarray,
    observation_weight: np.ndarray,
) -> np.ndarray:
    """
    Weighted stacking objective:

        maximize sum_i observation_weight[i]
                 log sum_k w_k p_k(y_i | y_-i).

    observation_weight is normalized to mean 1.
    """
    L = np.asarray(
        unweighted_log_lpd,
        dtype=float,
    )

    obs_w = np.asarray(
        observation_weight,
        dtype=float,
    )

    K = L.shape[1]

    def objective(
        model_w: np.ndarray,
    ) -> float:
        model_w = np.clip(
            model_w,
            1e-300,
            1,
        )

        row_log_mix = logsumexp(
            L
            + np.log(
                model_w
            )[None,:],
            axis=1,
        )

        return -float(
            np.sum(
                obs_w
                * row_log_mix
            )
        )

    x0 = np.full(
        K,
        1/K,
    )

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[
            (0,1)
        ]*K,
        constraints=[
            {
                "type":
                    "eq",

                "fun":
                    lambda w:
                    np.sum(w)-1,
            }
        ],
        options={
            "ftol":
                1e-12,

            "maxiter":
                10000,
        },
    )

    if not result.success:
        warnings.warn(
            "Stacking optimizer: "
            + str(
                result.message
            )
        )

    w = np.clip(
        result.x,
        0,
        1,
    )

    w /= w.sum()

    return w


def bb_pseudobma_weights(
    unweighted_log_lpd: np.ndarray,
    observation_weight: np.ndarray,
    *,
    B: int,
    seed: int,
) -> np.ndarray:
    """
    Bayesian bootstrap over constellations, retaining fixed participant
    evaluation weights.

    E[n*r_i] = 1, so the bootstrap is centered on
        sum_i observation_weight[i] * lpd_i.
    """
    rng = np.random.default_rng(
        seed
    )

    L = np.asarray(
        unweighted_log_lpd,
        dtype=float,
    )

    obs_w = np.asarray(
        observation_weight,
        dtype=float,
    )

    n, K = L.shape
    acc = np.zeros(K)

    for _ in range(B):
        r = rng.dirichlet(
            np.ones(n)
        )

        elpd_b = (
            n
            * np.sum(
                (
                    r
                    * obs_w
                )[:,None]
                * L,
                axis=0,
            )
        )

        acc += np.exp(
            elpd_b
            - logsumexp(
                elpd_b
            )
        )

    return acc/B


WEIGHTED_STACKING = stacking_weights(
    LPD_UNWEIGHTED,
    FULL_WEIGHTS,
)

WEIGHTED_BB = bb_pseudobma_weights(
    LPD_UNWEIGHTED,
    FULL_WEIGHTS,
    B=N_BB_MODEL_WEIGHTS,
    seed=SEED_BASE+8000,
)

UNWEIGHTED_STACKING = stacking_weights(
    LPD_UNWEIGHTED,
    np.ones(32),
)

UNWEIGHTED_BB = bb_pseudobma_weights(
    LPD_UNWEIGHTED,
    np.ones(32),
    B=N_BB_MODEL_WEIGHTS,
    seed=SEED_BASE+8001,
)


WEIGHTED_MODEL_WEIGHTS = pd.DataFrame(
    {
        "model":
            MODEL_NAMES,

        "stacking_weight":
            WEIGHTED_STACKING,

        "BB_pseudo_BMA_plus_weight":
            WEIGHTED_BB,
    }
)

UNWEIGHTED_MODEL_WEIGHTS = pd.DataFrame(
    {
        "model":
            MODEL_NAMES,

        "stacking_weight":
            UNWEIGHTED_STACKING,

        "BB_pseudo_BMA_plus_weight":
            UNWEIGHTED_BB,
    }
)

WEIGHTED_MODEL_WEIGHTS.to_csv(
    OUTPUT_DIR
    / "model_weights_PARTICIPANT_WEIGHTED_PRIMARY.csv",
    index=False,
)

UNWEIGHTED_MODEL_WEIGHTS.to_csv(
    OUTPUT_DIR
    / "model_weights_equal_constellation_sensitivity.csv",
    index=False,
)


# =====================================================================
# 22. BAYESIAN-BOOTSTRAP RANKING STABILITY
# =====================================================================

def bootstrap_ranking(
    L_unweighted: np.ndarray,
    obs_weight: np.ndarray,
    *,
    B: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    rng = np.random.default_rng(
        seed
    )

    L = np.asarray(
        L_unweighted,
        dtype=float,
    )

    ow = np.asarray(
        obs_weight,
        dtype=float,
    )

    n, K = L.shape

    winner_count = np.zeros(
        K,
        dtype=int,
    )

    elpd_boot = np.empty(
        (B,K),
        dtype=float,
    )

    for b in range(B):
        r = rng.dirichlet(
            np.ones(n)
        )

        e = (
            n
            * np.sum(
                (
                    r*ow
                )[:,None]
                * L,
                axis=0,
            )
        )

        elpd_boot[b] = e

        winner_count[
            int(
                np.argmax(e)
            )
        ] += 1

    winner = (
        pd.DataFrame(
            {
                "model":
                    MODEL_NAMES,

                "bootstrap_winner_frequency":
                    winner_count/B,
            }
        )
        .sort_values(
            "bootstrap_winner_frequency",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    ref = MODEL_NAMES.index(
        PRESPECIFIED_PRIMARY_MODEL
    )

    diff_rows = []

    for j, name in enumerate(
        MODEL_NAMES
    ):
        diff = (
            elpd_boot[:,j]
            -
            elpd_boot[:,ref]
        )

        diff_rows.append(
            {
                "model":
                    name,

                "reference_model":
                    PRESPECIFIED_PRIMARY_MODEL,

                "mean_diff":
                    float(
                        np.mean(diff)
                    ),

                "q2.5":
                    float(
                        np.quantile(
                            diff,
                            0.025,
                        )
                    ),

                "median":
                    float(
                        np.median(diff)
                    ),

                "q97.5":
                    float(
                        np.quantile(
                            diff,
                            0.975,
                        )
                    ),

                "P_model_gt_reference":
                    float(
                        np.mean(
                            diff > 0
                        )
                    ),
            }
        )

    return (
        winner,
        pd.DataFrame(
            diff_rows
        ),
    )


WEIGHTED_WINNER_STABILITY, WEIGHTED_BOOT_DIFF = (
    bootstrap_ranking(
        LPD_UNWEIGHTED,
        FULL_WEIGHTS,
        B=N_BOOTSTRAP_RANKING,
        seed=SEED_BASE+9000,
    )
)

UNWEIGHTED_WINNER_STABILITY, UNWEIGHTED_BOOT_DIFF = (
    bootstrap_ranking(
        LPD_UNWEIGHTED,
        np.ones(32),
        B=N_BOOTSTRAP_RANKING,
        seed=SEED_BASE+9001,
    )
)


WEIGHTED_WINNER_STABILITY.to_csv(
    OUTPUT_DIR
    / "bootstrap_ranking_PARTICIPANT_WEIGHTED_PRIMARY.csv",
    index=False,
)

WEIGHTED_BOOT_DIFF.to_csv(
    OUTPUT_DIR
    / "bootstrap_differences_PARTICIPANT_WEIGHTED_PRIMARY.csv",
    index=False,
)

UNWEIGHTED_WINNER_STABILITY.to_csv(
    OUTPUT_DIR
    / "bootstrap_ranking_equal_constellation_sensitivity.csv",
    index=False,
)

UNWEIGHTED_BOOT_DIFF.to_csv(
    OUTPUT_DIR
    / "bootstrap_differences_equal_constellation_sensitivity.csv",
    index=False,
)


# =====================================================================
# 23. EXACT-LOO MCMC DIAGNOSTICS
# =====================================================================

EXACT_DIAGNOSTICS = (
    EXACT_POINTWISE
    .groupby(
        "model",
        as_index=False,
    )
    .agg(
        folds=(
            "constellation_id",
            "count",
        ),

        worst_Rhat=(
            "max_Rhat",
            "max",
        ),

        min_ESS_bulk=(
            "min_ESS_bulk",
            "min",
        ),

        min_ESS_tail=(
            "min_ESS_tail",
            "min",
        ),

        total_divergences=(
            "divergences",
            "sum",
        ),

        total_max_treedepth_hits=(
            "max_treedepth_hits",
            "sum",
        ),
    )
)

EXACT_DIAGNOSTICS.to_csv(
    OUTPUT_DIR
    / "exact_loo_mcmc_diagnostics.csv",
    index=False,
)


# =====================================================================
# 24. PLOTS
# =====================================================================

fig, ax = plt.subplots(
    figsize=(10.5,7),
)

for name in MODEL_NAMES:
    sub = (
        POSTERIOR_CURVES[
            POSTERIOR_CURVES[
                "model"
            ]
            ==
            name
        ]
    )

    ax.plot(
        sub["x"],
        sub["mean"],
        label=name,
    )

ax.set_xlabel(
    "Weighted MetS score x"
)

ax.set_ylabel(
    "BAG (years)"
)

ax.set_title(
    "Participant-weighted posterior mean curves"
)

ax.legend(
    fontsize=8
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "posterior_mean_curves.png",
    dpi=170,
)

plt.close(fig)


fig, ax = plt.subplots(
    figsize=(10.5,7),
)

plot_df = (
    WEIGHTED_COMPARISON.copy()
)

ypos = np.arange(
    plot_df.shape[0]
)

ax.errorbar(
    plot_df["elpd"],
    ypos,
    xerr=plot_df[
        "se_elpd_pointwise_approx"
    ],
    fmt="o",
    capsize=4,
)

ax.set_yticks(
    ypos
)

ax.set_yticklabels(
    plot_df["model"]
)

ax.invert_yaxis()

ax.set_xlabel(
    "Participant-weighted exact-LOO score "
    "(higher is better)"
)

ax.set_title(
    "Primary participant-weighted exact-LOO comparison"
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "exact_loo_PARTICIPANT_WEIGHTED_PRIMARY.png",
    dpi=170,
)

plt.close(fig)


# =====================================================================
# 25. MANIFEST
# =====================================================================

manifest = {
    "program":
        "mets_brain_age_participant_weighted_reference_pystan_v3_table_s7.py",

    "fresh_run":
        FRESH_RUN,

    "data_source":
        "Exact two-decimal BAG LS Mean values and participant counts from Table S7",

    "bag_field_used":
        "BAG LS Mean (years), not beta coefficient",

    "primary_estimand":
        "participant-weighted constellation prediction",

    "full_fit_weight_formula":
        "w_i = 32*n_i/27375",

    "fold_fit_weight_formula":
        "w_i = G*n_i/sum_training(n), with G=31 in each LOO fold",

    "primary_evaluation_formula":
        "weighted_lpd_i = (32*n_i/27375) * heldout_lpd_i",

    "sensitivity_analysis":
        "equal-constellation unweighted exact LOO",

    "prespecified_primary_model":
        PRESPECIFIED_PRIMARY_MODEL,

    "models":
        MODEL_NAMES,

    "limitations": [
        "The 32 BAG values are adjusted aggregate estimates, not individual-level BAG observations.",
        "Participant-count weighting is an intentional pseudo-likelihood weighting rule, not replication of 27,375 BAG outcomes.",
        "The covariance matrix among the 32 adjusted BAG estimates is not modeled because it is unavailable.",
        "Historical adaptive model development on these same 32 data points cannot be undone by code.",
        "Independent validation requires new external data.",
    ],

    "full_chains":
        FULL_CHAINS,

    "full_num_samples":
        FULL_NUM_SAMPLES,

    "full_num_warmup":
        FULL_NUM_WARMUP,

    "cv_chains":
        CV_CHAINS,

    "cv_num_samples":
        CV_NUM_SAMPLES,

    "cv_num_warmup":
        CV_NUM_WARMUP,

    "adapt_delta":
        ADAPT_DELTA,

    "max_treedepth":
        MAX_TREEDEPTH,

    "smooth_a_eps":
        SMOOTH_A_EPS,
}

(
    OUTPUT_DIR
    / "analysis_manifest.json"
).write_text(
    json.dumps(
        manifest,
        indent=2,
    ),
    encoding="utf-8",
)


# =====================================================================
# 26. CONSOLE SUMMARY
# =====================================================================

print(
    "\n"
    + "="*76
    + "\nPARTICIPANT-WEIGHTED ANALYSIS COMPLETE\n"
    + "="*76
)

print(
    "\nPRIMARY participant-weighted exact-LOO comparison:"
)

print(
    WEIGHTED_COMPARISON.to_string(
        index=False,
        float_format=
            lambda x:
            f"{x:.6f}",
    )
)

print(
    "\nEqual-constellation sensitivity comparison:"
)

print(
    UNWEIGHTED_COMPARISON.to_string(
        index=False,
        float_format=
            lambda x:
            f"{x:.6f}",
    )
)

print(
    "\nPRIMARY participant-weighted model weights:"
)

print(
    WEIGHTED_MODEL_WEIGHTS.to_string(
        index=False,
        float_format=
            lambda x:
            f"{x:.6f}",
    )
)

print(
    "\nParticipant-weighted bootstrap ranking stability:"
)

print(
    WEIGHTED_WINNER_STABILITY.to_string(
        index=False,
        float_format=
            lambda x:
            f"{x:.6f}",
    )
)

print(
    "\nFull-data ridge relation:"
)

print(
    pd.DataFrame(
        [FULL_RIDGE_RELATION]
    ).to_string(
        index=False,
        float_format=
            lambda x:
            f"{x:.6f}",
    )
)

print(
    "\nFull-data MCMC diagnostics:"
)

print(
    FULL_DIAGNOSTICS.to_string(
        index=False,
        float_format=
            lambda x:
            f"{x:.5f}",
    )
)

print(
    "\nExact-LOO MCMC diagnostics:"
)

print(
    EXACT_DIAGNOSTICS.to_string(
        index=False,
        float_format=
            lambda x:
            f"{x:.5f}",
    )
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR.resolve()
)

print(
    "\nPrimary weighting rule:"
)

print(
    "  full fit/evaluation: "
    "w_i = 32*n_i/27375"
)

print(
    "  each LOO training fit: "
    "w_i = 31*n_i/sum_training(n)"
)

print(
    "\nThe equal-constellation exact-LOO results are retained "
    "as a sensitivity analysis."
)


# =====================================================================
# 27. SESSION INFO
# =====================================================================

def package_version(
    name: str,
) -> str:
    try:
        return metadata.version(
            name
        )
    except metadata.PackageNotFoundError:
        return "unknown"


print("\nSESSION INFO")
print("------------")
print(
    "Python:",
    sys.version.splitlines()[0],
)
print(
    "pystan:",
    package_version("pystan"),
)
print(
    "arviz:",
    package_version("arviz"),
)
print(
    "numpy:",
    np.__version__,
)
print(
    "pandas:",
    pd.__version__,
)
print(
    "scipy:",
    package_version("scipy"),
)
print(
    "matplotlib:",
    package_version("matplotlib"),
)
print(
    "run hash:",
    RUN_HASH,
)
