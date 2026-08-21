# =====================================================================
# Participant-weighted seven-model RStan analysis of MetS and BAG
# using exact Table S7 BAG LS means
#
# RStan translation/refactor of:
#   mets_brain_age_participant_weighted_reference_pystan_v3_table_s7.py
#
# Data source:
#   Dove et al. (2026), Alzheimer's & Dementia
#   DOI: 10.1002/alz.71563
#
# This R version intentionally relies on base/recommended R, RStan, and loo
# functions wherever a standard implementation exists:
#
#   * rstan::stan_model() compiles each Stan model ONCE per run.
#   * rstan::sampling() reuses the compiled stanmodel objects for all 224
#     exact-LOO refits.
#   * rstan::extract(), rstan::monitor(), rstan::check_hmc_diagnostics(),
#     rstan::get_num_divergent(), and rstan::get_num_max_treedepth()
#     replace hand-written posterior/diagnostic machinery.
#   * stats::lm() estimates the posterior p-a ridge relationships.
#   * loo::stacking_weights() and loo::pseudobma_weights() are used
#     directly for ordinary exact-CV model weights.
#   * loo::pseudobma_weights() is also used for participant-weighted
#     pseudo-BMA+ after multiplying each pointwise exact-LOO contribution
#     by its fixed participant evaluation weight.
#   * loo::stacking_weights() is used for participant-weighted stacking too:
#     the 32 exact-LOO rows are replicated by their integer Table S7
#     participant counts ONLY inside the deterministic stacking optimizer.
#     This is algebraically identical to maximizing the fixed participant-
#     weighted stacking objective, up to a positive constant multiplier.
#   * base R saveRDS()/readRDS(), write.csv(), plotting, quantile(), dnorm(),
#     sweep(), crossprod(), and matrix algebra are used throughout.
#
# Participant-weighted STACKING no longer uses a custom optimizer. Because
# the requested weights are exactly proportional to integer Table S7 counts,
# row replication gives the same stacking maximizer and lets the program use
# loo::stacking_weights() directly. The bootstrap ranking analysis remains
# custom because it is a project-specific diagnostic rather than a standard
# loo model-weight output.
#
# ---------------------------------------------------------------------
# PRIMARY WEIGHTING RULE
# ---------------------------------------------------------------------
# For any fitted dataset containing G constellations:
#
#     weight_i = G * n_i / sum_j(n_j)
#
# so:
#
#     sum_i weight_i = G
#     mean(weight_i) = 1
#
# Full fit:
#
#     weight_i = 32*n_i/27375
#
# Exact LOO training fold:
#
#     weight_i = 31*n_i/sum_training(n)
#
# The primary held-out score is:
#
#     weighted_lpd_i = (32*n_i/27375) * lpd_i
#
# ---------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------
# 1. Linear
# 2. Power
# 3. Normalized exponential
# 4. Unrestricted normalized exponential-power hybrid
# 5. Direct p-ridge
# 6. Hard log-p ridge
# 7. Soft log-p ridge
#
# Ridge relations are re-estimated inside every exact-LOO training fold.
#
# ---------------------------------------------------------------------
# IMPORTANT INTERPRETATION
# ---------------------------------------------------------------------
# These are 32 adjusted aggregate BAG LS means, not 27,375 individual BAG
# observations. Participant-count weighting is an intentional power-
# likelihood / relative-influence rule. It does not recover participant-
# level information or the covariance matrix among the 32 adjusted means.
#
# ---------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------
# In R/RStudio:
#
#   source("~/mets_brain_age_participant_weighted_reference_rstan_table_s7_final_fixed.R")
#
# Required non-base packages:
#
#   install.packages(c("rstan", "loo"))
#
# =====================================================================


# FINAL POST-PROCESSING REPAIR NOTE (2026-08-21)
# ------------------------------------------------
# Two participant-weighted stacking bugs were found in earlier R translations:
#
#   1. constrOptim(method = "BFGS", grad = NULL) failed because constrOptim
#      requires an explicit gradient for BFGS.
#   2. Switching that call to Nelder-Mead removed the runtime error but the
#      barrier optimizer stopped away from the true boundary optimum.
#
# This version removes the custom participant-weighted stacking optimizer
# entirely. For stacking only, each exact-LOO row is repeated n_i times and
# loo::stacking_weights() is called on the resulting 27,375 x 7 matrix. Since
#
#   sum_i n_i * log(mix_i)
#     = (27375 / 32) * sum_i (32*n_i/27375) * log(mix_i),
#
# the maximizer is EXACTLY the requested participant-weighted stacking
# maximizer. No participant replication is used in Stan fitting, ELPD standard
# errors, or bootstrap uncertainty calculations. FRESH_RUN remains FALSE so
# the already-completed exact-LOO cache can be reused.
#
# =====================================================================
# 0. CONFIGURATION AND DEPENDENCIES
# =====================================================================

# FALSE reuses the existing exact-LOO cache for this repaired/resumed run.
# Set TRUE only when you intentionally want to delete the output directory
# and recompute every fit from scratch.
FRESH_RUN <- FALSE

OUTPUT_DIR <- "mets_brain_age_participant_weighted_table_s7_results_rstan"
CACHE_DIR <- file.path(OUTPUT_DIR, "exact_loo_cache")
STAN_DIR <- file.path(OUTPUT_DIR, "stan_models")
FULL_FIT_DIR <- file.path(OUTPUT_DIR, "full_fits")
FULL_DRAW_DIR <- file.path(OUTPUT_DIR, "full_fit_draws")

PRESPECIFIED_PRIMARY_MODEL <- "Normalized_exponential"

FULL_CHAINS <- 4L
FULL_NUM_SAMPLES <- 3000L
FULL_NUM_WARMUP <- 3000L

CV_CHAINS <- 4L
CV_NUM_SAMPLES <- 2000L
CV_NUM_WARMUP <- 1500L

ADAPT_DELTA <- 0.99
MAX_TREEDEPTH <- 15L

SEED_BASE <- 2026081902

SMOOTH_A_EPS <- 1e-12

N_BB_MODEL_WEIGHTS <- 10000L
N_BOOTSTRAP_RANKING <- 50000L
X_GRID_N <- 401L

COMPONENT_NAMES <- c(
  "central_adiposity",
  "elevated_blood_pressure",
  "hyperglycemia",
  "elevated_triglycerides",
  "low_HDL_cholesterol"
)

MODEL_NAMES <- c(
  "Linear",
  "Power",
  "Normalized_exponential",
  "Power_exponential_hybrid",
  "Direct_p_ridge",
  "Hard_logp_ridge",
  "Soft_logp_ridge"
)

RIDGE_MODEL_NAMES <- c(
  "Direct_p_ridge",
  "Hard_logp_ridge",
  "Soft_logp_ridge"
)

MODEL_TYPE <- c(
  Linear = "linear",
  Power = "power",
  Normalized_exponential = "exponential",
  Power_exponential_hybrid = "hybrid",
  Direct_p_ridge = "direct_ridge",
  Hard_logp_ridge = "hard_logp_ridge",
  Soft_logp_ridge = "soft_logp_ridge"
)

options(warnPartialMatchDollar = TRUE)

required_packages <- c("rstan", "loo")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0L) {
  stop(
    "Missing required package(s): ",
    paste(missing_packages, collapse = ", "),
    "\nInstall them with install.packages(c(",
    paste(sprintf('"%s"', missing_packages), collapse = ", "),
    "))."
  )
}

rstan::rstan_options(auto_write = TRUE)

detected_cores <- parallel::detectCores(logical = TRUE)
if (is.na(detected_cores) || detected_cores < 1L) {
  detected_cores <- 1L
}

SAMPLING_CORES <- min(
  max(FULL_CHAINS, CV_CHAINS),
  detected_cores
)

options(mc.cores = SAMPLING_CORES)


# =====================================================================
# 1. EXACT TABLE S7 DATA
# =====================================================================

# Columns:
# 1 central adiposity
# 2 elevated blood pressure
# 3 hyperglycemia
# 4 elevated triglycerides
# 5 low HDL cholesterol

M <- rbind(
  c(0,0,0,0,0),
  c(0,0,0,0,1),
  c(1,0,0,0,1),
  c(0,0,1,0,0),
  c(0,0,0,1,0),
  c(0,0,0,1,1),
  c(1,0,0,0,0),
  c(1,0,1,0,0),
  c(1,0,1,1,0),
  c(0,1,0,0,0),
  c(0,0,1,1,0),
  c(1,0,0,1,1),
  c(1,0,1,0,1),
  c(1,0,0,1,0),
  c(0,1,0,1,0),
  c(0,1,1,0,0),
  c(1,0,1,1,1),
  c(1,1,0,0,0),
  c(0,1,0,0,1),
  c(0,1,0,1,1),
  c(1,1,1,0,0),
  c(1,1,0,1,0),
  c(0,1,1,1,0),
  c(0,0,1,1,1),
  c(1,1,0,0,1),
  c(1,1,0,1,1),
  c(0,0,1,0,1),
  c(1,1,1,1,0),
  c(0,1,1,1,1),
  c(0,1,1,0,1),
  c(1,1,1,1,1),
  c(1,1,1,0,1)
)

storage.mode(M) <- "double"
colnames(M) <- COMPONENT_NAMES

# Exact two-decimal BAG LS Means from Table S7, rearranged into the
# established program row order. These are NOT beta coefficients.
BAG <- c(
  -0.45,
  -0.39,
  -0.22,
  -0.18,
  -0.19,
   0.09,
   0.10,
   0.14,
   0.34,
   0.42,
   0.54,
   0.69,
   0.73,
   0.75,
   0.73,
   0.86,
   0.93,
   0.94,
   0.85,
   0.91,
   1.19,
   1.25,
   1.30,
   1.35,
   1.35,
   1.58,
   1.62,
   2.04,
   2.18,
   2.19,
   2.47,
   2.77
)

N_COUNTS <- c(
  4850,
  834, 172, 288, 948, 569, 497, 51, 31,
  5388, 84, 226, 47, 244, 1987, 437, 63,
  1094, 2034, 1845, 186, 857, 252, 75,
  740, 1257, 98, 204, 465, 451, 727, 374
)

CONSTELLATION_ID <- seq_len(nrow(M))
N_TOTAL <- sum(N_COUNTS)

TABLE_S7_N_COUNTS_REFERENCE <- c(
  4850,
  834, 172, 288, 948, 569, 497, 51, 31,
  5388, 84, 226, 47, 244, 1987, 437, 63,
  1094, 2034, 1845, 186, 857, 252, 75,
  740, 1257, 98, 204, 465, 451, 727, 374
)

TABLE_S7_BAG_LS_MEAN_REFERENCE <- c(
  -0.45, -0.39, -0.22, -0.18, -0.19, 0.09, 0.10, 0.14,
   0.34,  0.42,  0.54,  0.69,  0.73, 0.75, 0.73, 0.86,
   0.93,  0.94,  0.85,  0.91,  1.19, 1.25, 1.30, 1.35,
   1.35,  1.58,  1.62,  2.04,  2.18, 2.19, 2.47, 2.77
)

TABLE_S7_CONSTELLATION_LABELS <- c(
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
  "Adip+BP+Gluc+HDL"
)


# =====================================================================
# 2. OUTPUT DIRECTORIES
# =====================================================================

if (FRESH_RUN && dir.exists(OUTPUT_DIR)) {
  unlink(OUTPUT_DIR, recursive = TRUE, force = TRUE)
}

for (path in c(
  OUTPUT_DIR,
  CACHE_DIR,
  STAN_DIR,
  FULL_FIT_DIR,
  FULL_DRAW_DIR
)) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
}


# =====================================================================
# 3. DATA VALIDATION / AUDIT
# =====================================================================

stopifnot(
  nrow(M) == 32L,
  ncol(M) == 5L,
  length(BAG) == 32L,
  length(N_COUNTS) == 32L,
  all(M %in% c(0, 1)),
  all(is.finite(BAG)),
  all(N_COUNTS > 0),
  N_TOTAL == 27375,
  all(M[1, ] == 0),
  identical(as.numeric(N_COUNTS), as.numeric(TABLE_S7_N_COUNTS_REFERENCE)),
  identical(as.numeric(BAG), as.numeric(TABLE_S7_BAG_LS_MEAN_REFERENCE))
)

expected_component_counts <- c(
  central_adiposity = 6770,
  elevated_blood_pressure = 18298,
  hyperglycemia = 3833,
  elevated_triglycerides = 9834,
  low_HDL_cholesterol = 9977
)

observed_component_counts <- as.numeric(crossprod(N_COUNTS, M))
names(observed_component_counts) <- COMPONENT_NAMES

stopifnot(
  identical(
    as.numeric(observed_component_counts),
    as.numeric(expected_component_counts)
  )
)

expected_by_number <- c(
  `0` = 4850,
  `1` = 7955,
  `2` = 6770,
  `3` = 4710,
  `4` = 2363,
  `5` = 727
)

number_of_components <- rowSums(M)
observed_by_number <- tapply(
  N_COUNTS,
  number_of_components,
  sum
)

stopifnot(
  identical(
    as.numeric(observed_by_number),
    as.numeric(expected_by_number)
  )
)

TABLE_S7_EMBEDDED_DATA <- data.frame(
  constellation_id = CONSTELLATION_ID,
  constellation = TABLE_S7_CONSTELLATION_LABELS,
  participant_count = N_COUNTS,
  BAG_LS_mean_years = BAG,
  M,
  check.names = FALSE
)

utils::write.csv(
  TABLE_S7_EMBEDDED_DATA,
  file.path(OUTPUT_DIR, "table_s7_embedded_data_exact.csv"),
  row.names = FALSE
)

cat(
  "\nExact Table S7 data validation: PASS\n",
  "  32/32 participant counts match embedded Table S7 reference.\n",
  "  32/32 BAG LS Mean values match embedded Table S7 reference.\n",
  "  BAG values are the Table S7 LS means, not beta coefficients.\n",
  sep = ""
)


# =====================================================================
# 4. PARTICIPANT WEIGHTS
# =====================================================================

participant_weights <- function(indices) {
  idx <- as.integer(indices)
  counts <- as.numeric(N_COUNTS[idx])
  G <- length(idx)

  w <- G * counts / sum(counts)

  if (!isTRUE(all.equal(sum(w), as.numeric(G), tolerance = 1e-12))) {
    stop("Participant weights do not sum to G.")
  }

  w
}

ALL_IDX <- CONSTELLATION_ID
FULL_WEIGHTS <- participant_weights(ALL_IDX)

PARTICIPANT_WEIGHT_TABLE <- data.frame(
  constellation_id = CONSTELLATION_ID,
  participant_count = N_COUNTS,
  full_fit_weight = FULL_WEIGHTS,
  weight_fraction_of_total = FULL_WEIGHTS / sum(FULL_WEIGHTS),
  BAG = BAG,
  M,
  check.names = FALSE
)

utils::write.csv(
  PARTICIPANT_WEIGHT_TABLE,
  file.path(OUTPUT_DIR, "participant_weights.csv"),
  row.names = FALSE
)

cat(
  "\nParticipant-count weighting:\n",
  "  total participants = ", N_TOTAL, "\n",
  "  sum of full-data weights = ", sprintf("%.12f", sum(FULL_WEIGHTS)), "\n",
  "  mean full-data weight = ", sprintf("%.12f", mean(FULL_WEIGHTS)), "\n",
  "  minimum weight = ", sprintf("%.6f", min(FULL_WEIGHTS)), "\n",
  "  maximum weight = ", sprintf("%.6f", max(FULL_WEIGHTS)), "\n",
  "  max/min influence ratio = ",
  sprintf("%.2f", max(FULL_WEIGHTS) / min(FULL_WEIGHTS)),
  "\n",
  sep = ""
)


# =====================================================================
# 5. STAN MODEL DEFINITIONS
# =====================================================================

# Branchless smooth approximation to:
#
#                    exp(a*x^p)-1
#     H(x,p,a) =     ------------
#                       exp(a)-1
#
# b = sqrt(a^2 + eps^2) approximates |a| smoothly:
#
# H ≈ exp[-(a+b)(1-x^p)/2] * expm1(-b*x^p)/expm1(-b)
#
# Every exponential argument is non-positive. The expression is exact at
# x=0 and x=1 and differs from the exact family only in an infinitesimal
# neighborhood around a=0 controlled by eps=1e-12.

STAN_FUNCTIONS <- sprintf(
'functions {
  real normalized_exp_power(real x, real p, real a) {
    real xp = pow(x, p);
    real b = sqrt(square(a) + %.17g * %.17g);
    real expo = -0.5 * (a + b) * (1.0 - xp);
    return exp(expo) * expm1(-b * xp) / expm1(-b);
  }
}
',
  SMOOTH_A_EPS,
  SMOOTH_A_EPS
)

STAN_DATA <- '
data {
  int<lower=1> G;
  matrix[G,5] M;
  vector[G] bag;
  vector<lower=0>[G] obs_weight;
}
'

STAN_LINEAR <- paste0(
  STAN_DATA,
'
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
    target += obs_weight[g] * normal_lpdf(bag[g] | mu[g], sigma);
}
'
)

STAN_POWER <- paste0(
  STAN_DATA,
'
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
    target += obs_weight[g] * normal_lpdf(bag[g] | mu[g], sigma);
}
'
)

STAN_EXP <- paste0(
  STAN_FUNCTIONS,
  STAN_DATA,
'
parameters {
  simplex[5] mets;
  real<lower=0> k;
  real m;
  real a;
  real<lower=0> sigma;
}
transformed parameters {
  vector[G] x = M * mets;
  vector[G] mu;

  for (g in 1:G)
    mu[g] = m + k*normalized_exp_power(x[g], 1.0, a);
}
model {
  mets ~ dirichlet(rep_vector(1.0, 5));
  m ~ normal(0, 3);
  a ~ normal(0, 1);
  k ~ normal(0, 5);
  sigma ~ student_t(4, 0, 1);

  for (g in 1:G)
    target += obs_weight[g] * normal_lpdf(bag[g] | mu[g], sigma);
}
'
)

STAN_HYBRID <- paste0(
  STAN_FUNCTIONS,
  STAN_DATA,
'
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
    mu[g] = m + k*normalized_exp_power(x[g], p, a);
}
model {
  mets ~ dirichlet(rep_vector(1.0, 5));
  m ~ normal(0, 3);
  log_p ~ normal(0, 0.5);
  a ~ normal(0, 1);
  k ~ normal(0, 5);
  sigma ~ student_t(4, 0, 1);

  for (g in 1:G)
    target += obs_weight[g] * normal_lpdf(bag[g] | mu[g], sigma);
}
'
)

# Direct p-ridge:
#
#     p = alpha_p + beta_p*a
#
# The PyStan version sampled p>0, derived a, and placed N(0,1) on a.
# Algebraically, that induces:
#
#     p ~ Normal(alpha_p, |beta_p|)
#
# up to a constant independent of the sampled parameters. Writing the
# equivalent normal prior directly on p is simpler, uses Stan's built-in
# distribution machinery, and avoids the compiler's false-positive
# "p has no prior" warning.
STAN_DIRECT_RIDGE <- paste0(
  STAN_FUNCTIONS,
'
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
  real a = (p - alpha_p) / beta_p;
  vector[G] x = M * mets;
  vector[G] mu;

  for (g in 1:G)
    mu[g] = m + k*normalized_exp_power(x[g], p, a);
}
model {
  mets ~ dirichlet(rep_vector(1.0, 5));
  m ~ normal(0, 3);
  p ~ normal(alpha_p, abs(beta_p));
  k ~ normal(0, 5);
  sigma ~ student_t(4, 0, 1);

  for (g in 1:G)
    target += obs_weight[g] * normal_lpdf(bag[g] | mu[g], sigma);
}
'
)

STAN_HARD_LOGP_RIDGE <- paste0(
  STAN_FUNCTIONS,
'
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
    mu[g] = m + k*normalized_exp_power(x[g], p, a);
}
model {
  mets ~ dirichlet(rep_vector(1.0, 5));
  m ~ normal(0, 3);
  a ~ normal(0, 1);
  k ~ normal(0, 5);
  sigma ~ student_t(4, 0, 1);

  for (g in 1:G)
    target += obs_weight[g] * normal_lpdf(bag[g] | mu[g], sigma);
}
'
)

STAN_SOFT_LOGP_RIDGE <- paste0(
  STAN_FUNCTIONS,
'
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
  real log_p = alpha_logp + beta_logp*a + tau_logp*z;
  real p = exp(log_p);
  vector[G] x = M * mets;
  vector[G] mu;

  for (g in 1:G)
    mu[g] = m + k*normalized_exp_power(x[g], p, a);
}
model {
  mets ~ dirichlet(rep_vector(1.0, 5));
  m ~ normal(0, 3);
  a ~ normal(0, 1);
  z ~ normal(0, 1);
  k ~ normal(0, 5);
  sigma ~ student_t(4, 0, 1);

  for (g in 1:G)
    target += obs_weight[g] * normal_lpdf(bag[g] | mu[g], sigma);
}
'
)

MODEL_CODE <- list(
  Linear = STAN_LINEAR,
  Power = STAN_POWER,
  Normalized_exponential = STAN_EXP,
  Power_exponential_hybrid = STAN_HYBRID,
  Direct_p_ridge = STAN_DIRECT_RIDGE,
  Hard_logp_ridge = STAN_HARD_LOGP_RIDGE,
  Soft_logp_ridge = STAN_SOFT_LOGP_RIDGE
)

STAN_FILES <- setNames(
  file.path(STAN_DIR, paste0(tolower(MODEL_NAMES), ".stan")),
  MODEL_NAMES
)

for (name in MODEL_NAMES) {
  writeLines(
    MODEL_CODE[[name]],
    STAN_FILES[[name]],
    useBytes = TRUE
  )
}


# =====================================================================
# 6. RUN SIGNATURE / CACHE VERSIONING
# =====================================================================

make_run_hash <- function() {
  settings <- c(
    "program_version=rstan_table_s7_v1",
    "weight_rule=G*n_i/sum_training_n",
    "evaluation_weight_rule=32*n_i/27375",
    paste0("full_chains=", FULL_CHAINS),
    paste0("full_samples=", FULL_NUM_SAMPLES),
    paste0("full_warmup=", FULL_NUM_WARMUP),
    paste0("cv_chains=", CV_CHAINS),
    paste0("cv_samples=", CV_NUM_SAMPLES),
    paste0("cv_warmup=", CV_NUM_WARMUP),
    paste0("adapt_delta=", format(ADAPT_DELTA, digits = 17)),
    paste0("max_treedepth=", MAX_TREEDEPTH),
    paste0("smooth_a_eps=", format(SMOOTH_A_EPS, scientific = TRUE, digits = 17)),
    paste0(
      names(STAN_FILES),
      "=",
      unname(tools::md5sum(STAN_FILES))
    )
  )

  tmp <- tempfile(fileext = ".txt")
  on.exit(unlink(tmp), add = TRUE)

  writeLines(settings, tmp, useBytes = TRUE)

  substr(
    unname(tools::md5sum(tmp)),
    1L,
    16L
  )
}

RUN_HASH <- make_run_hash()


# =====================================================================
# 7. COMPILE EACH STAN MODEL ONCE
# =====================================================================

cat(
  "\nCompiling seven Stan models once for reuse across all fits...\n"
)

STAN_MODELS <- setNames(
  lapply(
    MODEL_NAMES,
    function(name) {
      cat("  Compiling ", name, "...\n", sep = "")

      rstan::stan_model(
        file = STAN_FILES[[name]],
        model_name = paste0("mets_", tolower(name)),
        verbose = FALSE
      )
    }
  ),
  MODEL_NAMES
)


# =====================================================================
# 8. DATA / SAMPLING / EXTRACTION HELPERS
# =====================================================================

make_data <- function(indices, extra = NULL) {
  idx <- as.integer(indices)

  out <- list(
    G = length(idx),
    M = M[idx, , drop = FALSE],
    bag = BAG[idx],
    obs_weight = participant_weights(idx)
  )

  if (!is.null(extra)) {
    out <- utils::modifyList(out, extra)
  }

  out
}

saved_pars <- function(model_name) {
  type <- unname(MODEL_TYPE[[model_name]])

  switch(
    type,
    linear = c("mets", "m", "k", "sigma"),
    power = c("mets", "m", "k", "sigma", "log_p", "p"),
    exponential = c("mets", "m", "k", "sigma", "a"),
    hybrid = c("mets", "m", "k", "sigma", "log_p", "p", "a"),
    direct_ridge = c("mets", "m", "k", "sigma", "p", "a"),
    hard_logp_ridge = c("mets", "m", "k", "sigma", "a", "log_p", "p"),
    soft_logp_ridge = c("mets", "m", "k", "sigma", "a", "z", "log_p", "p"),
    stop("Unknown model type: ", type)
  )
}

diagnostic_pars <- function(model_name) {
  type <- unname(MODEL_TYPE[[model_name]])

  switch(
    type,
    linear = c("mets", "m", "k", "sigma"),
    power = c("mets", "m", "k", "sigma", "log_p", "p"),
    exponential = c("mets", "m", "k", "sigma", "a"),
    hybrid = c("mets", "m", "k", "sigma", "log_p", "p", "a"),
    direct_ridge = c("mets", "m", "k", "sigma", "p", "a"),
    hard_logp_ridge = c("mets", "m", "k", "sigma", "a", "log_p", "p"),
    soft_logp_ridge = c("mets", "m", "k", "sigma", "a", "z", "log_p", "p"),
    stop("Unknown model type: ", type)
  )
}

fit_stan <- function(
  model_name,
  data,
  seed,
  chains,
  samples,
  warmup,
  quiet = FALSE
) {
  if (!quiet) {
    cat(
      "\n",
      paste(rep("=", 74L), collapse = ""),
      "\nFIT: ", model_name, "\n",
      paste(rep("=", 74L), collapse = ""),
      "\n",
      sep = ""
    )
  }

  total_iter <- as.integer(samples + warmup)

  fit <- rstan::sampling(
    object = STAN_MODELS[[model_name]],
    data = data,
    pars = saved_pars(model_name),
    include = TRUE,
    chains = as.integer(chains),
    iter = total_iter,
    warmup = as.integer(warmup),
    thin = 1L,
    seed = as.integer(seed),
    cores = min(as.integer(chains), SAMPLING_CORES),
    refresh = if (quiet) 0L else max(1L, total_iter %/% 10L),
    control = list(
      adapt_delta = ADAPT_DELTA,
      max_treedepth = MAX_TREEDEPTH
    )
  )

  if (!quiet) {
    cat("\nRStan HMC diagnostics for ", model_name, ":\n", sep = "")
    rstan::check_hmc_diagnostics(fit)
  }

  fit
}

extract_draws <- function(fit, model_name) {
  type <- unname(MODEL_TYPE[[model_name]])

  post <- rstan::extract(
    fit,
    pars = saved_pars(model_name),
    permuted = TRUE,
    inc_warmup = FALSE
  )

  out <- list(
    model = model_name,
    type = type,
    mets = as.matrix(post[["mets"]]),
    m = as.numeric(post[["m"]]),
    k = as.numeric(post[["k"]]),
    sigma = as.numeric(post[["sigma"]])
  )

  if (identical(type, "power")) {
    out[["log_p"]] <- as.numeric(post[["log_p"]])
    out[["p"]] <- as.numeric(post[["p"]])

  } else if (identical(type, "exponential")) {
    out[["a"]] <- as.numeric(post[["a"]])
    out[["p"]] <- rep(1, length(out[["a"]]))

  } else if (identical(type, "hybrid")) {
    out[["log_p"]] <- as.numeric(post[["log_p"]])
    out[["p"]] <- as.numeric(post[["p"]])
    out[["a"]] <- as.numeric(post[["a"]])

  } else if (identical(type, "direct_ridge")) {
    out[["p"]] <- as.numeric(post[["p"]])
    out[["a"]] <- as.numeric(post[["a"]])
    out[["log_p"]] <- log(out[["p"]])

  } else if (identical(type, "hard_logp_ridge")) {
    out[["a"]] <- as.numeric(post[["a"]])
    out[["log_p"]] <- as.numeric(post[["log_p"]])
    out[["p"]] <- as.numeric(post[["p"]])

  } else if (identical(type, "soft_logp_ridge")) {
    out[["a"]] <- as.numeric(post[["a"]])
    out[["z"]] <- as.numeric(post[["z"]])
    out[["log_p"]] <- as.numeric(post[["log_p"]])
    out[["p"]] <- as.numeric(post[["p"]])
  }

  out
}

fit_diagnostics <- function(fit, model_name) {
  arr <- as.array(
    fit,
    pars = diagnostic_pars(model_name)
  )

  mon <- rstan::monitor(
    arr,
    print = FALSE,
    probs = c(0.025, 0.5, 0.975)
  )

  required_cols <- c("Rhat", "Bulk_ESS", "Tail_ESS")

  if (!all(required_cols %in% colnames(mon))) {
    stop(
      "Unexpected rstan::monitor() columns: ",
      paste(colnames(mon), collapse = ", ")
    )
  }

  rhat <- mon[, "Rhat"]
  bulk <- mon[, "Bulk_ESS"]
  tail <- mon[, "Tail_ESS"]

  list(
    max_Rhat = max(rhat[is.finite(rhat)], na.rm = TRUE),
    min_ESS_bulk = min(bulk[is.finite(bulk)], na.rm = TRUE),
    min_ESS_tail = min(tail[is.finite(tail)], na.rm = TRUE),
    divergences = as.integer(rstan::get_num_divergent(fit)),
    max_treedepth_hits = as.integer(rstan::get_num_max_treedepth(fit))
  )
}


# =====================================================================
# 9. RIDGE REGRESSION USING stats::lm()
# =====================================================================

regression_summary <- function(y, x) {
  df <- data.frame(y = as.numeric(y), x = as.numeric(x))
  fit <- stats::lm(y ~ x, data = df)

  coefs <- stats::coef(fit)
  fit_summary <- summary(fit)

  list(
    alpha = unname(coefs[[1L]]),
    beta = unname(coefs[[2L]]),
    tau = unname(fit_summary[["sigma"]]),
    r2 = unname(fit_summary[["r.squared"]]),
    corr = stats::cor(df[["x"]], df[["y"]])
  )
}

estimate_ridge <- function(hybrid_draws) {
  direct <- regression_summary(
    hybrid_draws[["p"]],
    hybrid_draws[["a"]]
  )

  logrel <- regression_summary(
    hybrid_draws[["log_p"]],
    hybrid_draws[["a"]]
  )

  if (abs(direct[["beta"]]) < 1e-6) {
    stop(
      "Direct p-ridge beta is too close to zero for stable ",
      "reparameterization."
    )
  }

  c(
    alpha_p = direct[["alpha"]],
    beta_p = direct[["beta"]],
    tau_p = direct[["tau"]],
    r2_p = direct[["r2"]],
    corr_p_a = direct[["corr"]],
    alpha_logp = logrel[["alpha"]],
    beta_logp = logrel[["beta"]],
    tau_logp = max(logrel[["tau"]], .Machine$double.xmin),
    r2_logp = logrel[["r2"]],
    corr_logp_a = logrel[["corr"]]
  )
}

ridge_extra_data <- function(model_name, relation) {
  if (identical(model_name, "Direct_p_ridge")) {
    return(list(
      alpha_p = unname(relation[["alpha_p"]]),
      beta_p = unname(relation[["beta_p"]])
    ))
  }

  if (identical(model_name, "Hard_logp_ridge")) {
    return(list(
      alpha_logp = unname(relation[["alpha_logp"]]),
      beta_logp = unname(relation[["beta_logp"]])
    ))
  }

  if (identical(model_name, "Soft_logp_ridge")) {
    return(list(
      alpha_logp = unname(relation[["alpha_logp"]]),
      beta_logp = unname(relation[["beta_logp"]]),
      tau_logp = unname(relation[["tau_logp"]])
    ))
  }

  list()
}


# =====================================================================
# 10. R RESPONSE / POSTERIOR-PREDICTION FUNCTIONS
# =====================================================================

normalized_exp_power_R <- function(
  x,
  p,
  a,
  eps = SMOOTH_A_EPS
) {
  xp <- x^p
  b <- sqrt(a*a + eps*eps)
  expo <- -0.5 * (a + b) * (1 - xp)

  exp(expo) * expm1(-b*xp) / expm1(-b)
}

mu_for_patterns <- function(draws, patterns) {
  patterns <- as.matrix(patterns)

  if (ncol(patterns) != 5L) {
    stop("patterns must have five columns.")
  }

  x <- draws[["mets"]] %*% t(patterns)
  S <- nrow(x)
  P <- ncol(x)

  type <- draws[["type"]]

  if (identical(type, "linear")) {
    out <- sweep(x, 1L, draws[["k"]], `*`)
    return(sweep(out, 1L, draws[["m"]], `+`))
  }

  p_mat <- matrix(
    draws[["p"]],
    nrow = S,
    ncol = P
  )

  if (identical(type, "power")) {
    shape <- x^p_mat
  } else {
    a_mat <- matrix(
      draws[["a"]],
      nrow = S,
      ncol = P
    )

    shape <- normalized_exp_power_R(
      x,
      p_mat,
      a_mat
    )
  }

  out <- sweep(shape, 1L, draws[["k"]], `*`)
  sweep(out, 1L, draws[["m"]], `+`)
}

log_mean_exp <- function(x) {
  x <- as.numeric(x)
  m <- max(x)

  if (!is.finite(m)) {
    return(m)
  }

  m + log(mean(exp(x - m)))
}

heldout_prediction <- function(draws, hold_idx) {
  mu <- mu_for_patterns(
    draws,
    M[hold_idx, , drop = FALSE]
  )[, 1L]

  sigma <- draws[["sigma"]]

  lpd <- log_mean_exp(
    stats::dnorm(
      BAG[[hold_idx]],
      mean = mu,
      sd = sigma,
      log = TRUE
    )
  )

  eval_weight <- FULL_WEIGHTS[[hold_idx]]

  c(
    exact_lpd_unweighted = lpd,
    evaluation_participant_weight = eval_weight,
    exact_lpd_participant_weighted = eval_weight*lpd,
    pred_mean = mean(mu),
    pred_median = stats::median(mu),
    pred_q2.5 = unname(stats::quantile(mu, 0.025, names = FALSE)),
    pred_q97.5 = unname(stats::quantile(mu, 0.975, names = FALSE))
  )
}

summarize_vector <- function(x) {
  q <- stats::quantile(
    x,
    probs = c(0.025, 0.5, 0.975),
    names = FALSE
  )

  c(
    mean = mean(x),
    sd = stats::sd(x),
    q2.5 = q[[1L]],
    median = q[[2L]],
    q97.5 = q[[3L]]
  )
}


# =====================================================================
# 11. FULL-DATA FITS
# =====================================================================

FULL_DRAWS <- setNames(vector("list", length(MODEL_NAMES)), MODEL_NAMES)
FULL_DIAG_ROWS <- vector("list", length(MODEL_NAMES))

core_models <- c(
  "Linear",
  "Power",
  "Normalized_exponential",
  "Power_exponential_hybrid"
)

for (j in seq_along(core_models)) {
  name <- core_models[[j]]

  fit <- fit_stan(
    model_name = name,
    data = make_data(ALL_IDX),
    seed = SEED_BASE + (j - 1L),
    chains = FULL_CHAINS,
    samples = FULL_NUM_SAMPLES,
    warmup = FULL_NUM_WARMUP,
    quiet = FALSE
  )

  draws <- extract_draws(fit, name)
  diag <- fit_diagnostics(fit, name)

  FULL_DRAWS[[name]] <- draws
  FULL_DIAG_ROWS[[match(name, MODEL_NAMES)]] <- data.frame(
    model = name,
    as.data.frame(diag, check.names = FALSE),
    check.names = FALSE
  )

  # RStan documents save()/load() as the supported way to persist fitted
  # objects together with their DSO state across R sessions.
  save(
    fit,
    file = file.path(
      FULL_FIT_DIR,
      paste0(tolower(name), "_stanfit_", RUN_HASH, ".RData")
    ),
    compress = "xz"
  )

  saveRDS(
    draws,
    file.path(
      FULL_DRAW_DIR,
      paste0(tolower(name), "_draws_", RUN_HASH, ".rds")
    ),
    compress = "xz"
  )

  rm(fit)
  gc(verbose = FALSE)
}

FULL_RIDGE_RELATION <- estimate_ridge(
  FULL_DRAWS[["Power_exponential_hybrid"]]
)

utils::write.csv(
  as.data.frame(as.list(FULL_RIDGE_RELATION), check.names = FALSE),
  file.path(OUTPUT_DIR, "full_data_ridge_relation.csv"),
  row.names = FALSE
)

ridge_models <- c(
  "Direct_p_ridge",
  "Hard_logp_ridge",
  "Soft_logp_ridge"
)

for (j in seq_along(ridge_models)) {
  name <- ridge_models[[j]]

  fit <- fit_stan(
    model_name = name,
    data = make_data(
      ALL_IDX,
      extra = ridge_extra_data(
        name,
        FULL_RIDGE_RELATION
      )
    ),
    seed = SEED_BASE + 100L + (j - 1L),
    chains = FULL_CHAINS,
    samples = FULL_NUM_SAMPLES,
    warmup = FULL_NUM_WARMUP,
    quiet = FALSE
  )

  draws <- extract_draws(fit, name)
  diag <- fit_diagnostics(fit, name)

  FULL_DRAWS[[name]] <- draws
  FULL_DIAG_ROWS[[match(name, MODEL_NAMES)]] <- data.frame(
    model = name,
    as.data.frame(diag, check.names = FALSE),
    check.names = FALSE
  )

  # RStan documents save()/load() as the supported way to persist fitted
  # objects together with their DSO state across R sessions.
  save(
    fit,
    file = file.path(
      FULL_FIT_DIR,
      paste0(tolower(name), "_stanfit_", RUN_HASH, ".RData")
    ),
    compress = "xz"
  )

  saveRDS(
    draws,
    file.path(
      FULL_DRAW_DIR,
      paste0(tolower(name), "_draws_", RUN_HASH, ".rds")
    ),
    compress = "xz"
  )

  rm(fit)
  gc(verbose = FALSE)
}

FULL_DIAGNOSTICS <- do.call(rbind, FULL_DIAG_ROWS)
rownames(FULL_DIAGNOSTICS) <- NULL

utils::write.csv(
  FULL_DIAGNOSTICS,
  file.path(OUTPUT_DIR, "full_data_mcmc_diagnostics.csv"),
  row.names = FALSE
)


# =====================================================================
# 12. FULL-DATA PARAMETER / WEIGHT SUMMARIES
# =====================================================================

weight_rows <- vector(
  "list",
  length(MODEL_NAMES) * length(COMPONENT_NAMES)
)

wi <- 1L

for (name in MODEL_NAMES) {
  d <- FULL_DRAWS[[name]]

  for (j in seq_along(COMPONENT_NAMES)) {
    s <- summarize_vector(d[["mets"]][, j])

    weight_rows[[wi]] <- data.frame(
      model = name,
      factor = COMPONENT_NAMES[[j]],
      mean = unname(s[["mean"]]),
      sd = unname(s[["sd"]]),
      q2.5 = unname(s[["q2.5"]]),
      median = unname(s[["median"]]),
      q97.5 = unname(s[["q97.5"]]),
      check.names = FALSE
    )

    wi <- wi + 1L
  }
}

FULL_WEIGHT_SUMMARY <- do.call(rbind, weight_rows)
rownames(FULL_WEIGHT_SUMMARY) <- NULL

PARAMETER_NAMES <- c(
  "m",
  "k",
  "sigma",
  "p",
  "a",
  "log_p",
  "z"
)

STAT_NAMES <- c(
  "mean",
  "sd",
  "q2.5",
  "median",
  "q97.5"
)

parameter_numeric_cols <- as.vector(
  outer(
    PARAMETER_NAMES,
    STAT_NAMES,
    paste,
    sep = "_"
  )
)

parameter_rows <- lapply(
  MODEL_NAMES,
  function(name) {
    d <- FULL_DRAWS[[name]]

    vals <- setNames(
      as.list(rep(NA_real_, length(parameter_numeric_cols))),
      parameter_numeric_cols
    )

    for (par in PARAMETER_NAMES) {
      if (!is.null(d[[par]])) {
        s <- summarize_vector(d[[par]])

        for (stat_name in STAT_NAMES) {
          vals[[paste0(par, "_", stat_name)]] <- unname(s[[stat_name]])
        }
      }
    }

    data.frame(
      model = name,
      as.data.frame(vals, check.names = FALSE),
      P_p_gt_1 = if (!is.null(d[["p"]])) mean(d[["p"]] > 1) else NA_real_,
      P_a_gt_0 = if (!is.null(d[["a"]])) mean(d[["a"]] > 0) else NA_real_,
      check.names = FALSE
    )
  }
)

FULL_PARAMETER_SUMMARY <- do.call(rbind, parameter_rows)
rownames(FULL_PARAMETER_SUMMARY) <- NULL

utils::write.csv(
  FULL_WEIGHT_SUMMARY,
  file.path(OUTPUT_DIR, "full_data_weight_summary.csv"),
  row.names = FALSE
)

utils::write.csv(
  FULL_PARAMETER_SUMMARY,
  file.path(OUTPUT_DIR, "full_data_parameter_summary.csv"),
  row.names = FALSE
)


# =====================================================================
# 13. BP-ONLY POSTERIOR PREDICTIONS
# =====================================================================

BP_ONLY <- matrix(
  c(0, 1, 0, 0, 0),
  nrow = 1L
)

bp_rows <- lapply(
  MODEL_NAMES,
  function(name) {
    d <- FULL_DRAWS[[name]]

    mu <- mu_for_patterns(
      d,
      BP_ONLY
    )[, 1L]

    x <- d[["mets"]][, 2L]

    xq <- stats::quantile(
      x,
      c(0.025, 0.5, 0.975),
      names = FALSE
    )

    yq <- stats::quantile(
      mu,
      c(0.025, 0.5, 0.975),
      names = FALSE
    )

    data.frame(
      model = name,
      x_mean = mean(x),
      x_q2.5 = xq[[1L]],
      x_median = xq[[2L]],
      x_q97.5 = xq[[3L]],
      BAG_mean = mean(mu),
      BAG_q2.5 = yq[[1L]],
      BAG_median = yq[[2L]],
      BAG_q97.5 = yq[[3L]],
      check.names = FALSE
    )
  }
)

BP_PREDICTIONS <- do.call(rbind, bp_rows)
rownames(BP_PREDICTIONS) <- NULL

utils::write.csv(
  BP_PREDICTIONS,
  file.path(OUTPUT_DIR, "blood_pressure_only_predictions.csv"),
  row.names = FALSE
)


# =====================================================================
# 14. POSTERIOR CURVES
# =====================================================================

X_GRID <- seq(0, 1, length.out = X_GRID_N)

curve_rows <- vector(
  "list",
  length(MODEL_NAMES) * X_GRID_N
)

ci <- 1L

for (name in MODEL_NAMES) {
  d <- FULL_DRAWS[[name]]
  S <- length(d[["m"]])

  for (x_value in X_GRID) {
    x <- rep(x_value, S)

    if (identical(d[["type"]], "linear")) {
      y <- d[["m"]] + d[["k"]]*x

    } else if (identical(d[["type"]], "power")) {
      y <- d[["m"]] + d[["k"]]*(x^d[["p"]])

    } else {
      y <- d[["m"]] +
        d[["k"]] *
        normalized_exp_power_R(
          x,
          d[["p"]],
          d[["a"]]
        )
    }

    q <- stats::quantile(
      y,
      c(0.025, 0.5, 0.975),
      names = FALSE
    )

    curve_rows[[ci]] <- data.frame(
      model = name,
      x = x_value,
      mean = mean(y),
      q2.5 = q[[1L]],
      median = q[[2L]],
      q97.5 = q[[3L]],
      check.names = FALSE
    )

    ci <- ci + 1L
  }
}

POSTERIOR_CURVES <- do.call(rbind, curve_rows)
rownames(POSTERIOR_CURVES) <- NULL

utils::write.csv(
  POSTERIOR_CURVES,
  file.path(OUTPUT_DIR, "posterior_curves.csv"),
  row.names = FALSE
)


# =====================================================================
# 15. EXACT-LOO CACHE HELPERS
# =====================================================================

fold_cache_path <- function(model_name, hold_idx) {
  file.path(
    CACHE_DIR,
    sprintf(
      "%s_fold%02d_%s.rds",
      tolower(model_name),
      hold_idx,
      RUN_HASH
    )
  )
}

ridge_cache_path <- function(hold_idx) {
  file.path(
    CACHE_DIR,
    sprintf(
      "ridge_relation_fold%02d_%s.rds",
      hold_idx,
      RUN_HASH
    )
  )
}


# =====================================================================
# 16. ONE EXACT-LOO REFIT
# =====================================================================

fit_one_loo <- function(
  model_name,
  hold_idx,
  train_idx,
  relation = NULL
) {
  cache <- fold_cache_path(
    model_name,
    hold_idx
  )

  if (file.exists(cache) && !FRESH_RUN) {
    row <- readRDS(cache)
    learned <- NULL

    if (identical(model_name, "Power_exponential_hybrid")) {
      rp <- ridge_cache_path(hold_idx)

      if (file.exists(rp)) {
        learned <- readRDS(rp)
      }
    }

    return(list(
      row = row,
      learned = learned
    ))
  }

  extra <- NULL

  if (model_name %in% RIDGE_MODEL_NAMES) {
    if (is.null(relation)) {
      stop(
        model_name,
        " requires a training-only ridge relation."
      )
    }

    extra <- ridge_extra_data(
      model_name,
      relation
    )
  }

  model_index_zero_based <- match(
    model_name,
    MODEL_NAMES
  ) - 1L

  fit <- fit_stan(
    model_name = model_name,
    data = make_data(
      train_idx,
      extra = extra
    ),
    seed = SEED_BASE +
      1000000 +
      10000 * model_index_zero_based +
      (hold_idx - 1L),
    chains = CV_CHAINS,
    samples = CV_NUM_SAMPLES,
    warmup = CV_NUM_WARMUP,
    quiet = TRUE
  )

  draws <- extract_draws(
    fit,
    model_name
  )

  diag <- fit_diagnostics(
    fit,
    model_name
  )

  pred <- heldout_prediction(
    draws,
    hold_idx
  )

  train_weights <- participant_weights(
    train_idx
  )

  row <- data.frame(
    model = model_name,
    constellation_id = hold_idx,
    heldout_BAG = BAG[[hold_idx]],
    participant_count = N_COUNTS[[hold_idx]],
    train_participant_total = sum(N_COUNTS[train_idx]),
    train_weight_sum = sum(train_weights),
    heldout_full_weight = FULL_WEIGHTS[[hold_idx]],
    number_of_components = sum(M[hold_idx, ]),
    central_adiposity = M[hold_idx, 1L],
    elevated_blood_pressure = M[hold_idx, 2L],
    hyperglycemia = M[hold_idx, 3L],
    elevated_triglycerides = M[hold_idx, 4L],
    low_HDL_cholesterol = M[hold_idx, 5L],
    as.data.frame(as.list(pred), check.names = FALSE),
    as.data.frame(diag, check.names = FALSE),
    check.names = FALSE
  )

  learned <- NULL

  if (identical(model_name, "Power_exponential_hybrid")) {
    learned <- estimate_ridge(draws)
    saveRDS(
      learned,
      ridge_cache_path(hold_idx),
      compress = "xz"
    )
  }

  saveRDS(
    row,
    cache,
    compress = "xz"
  )

  rm(fit, draws)
  gc(verbose = FALSE)

  list(
    row = row,
    learned = learned
  )
}


# =====================================================================
# 17. PARTICIPANT-WEIGHTED EXACT LOO
# =====================================================================

LOO_ROWS <- list()
RIDGE_FOLD_ROWS <- list()

cat(
  "\n",
  paste(rep("=", 76L), collapse = ""),
  "\nPARTICIPANT-WEIGHTED EXACT LOO\n",
  paste(rep("=", 76L), collapse = ""),
  "\n",
  sep = ""
)

for (hold_idx in ALL_IDX) {
  cat(
    "\nOuter fold ",
    sprintf("%02d", hold_idx),
    "/32\n",
    sep = ""
  )

  train_idx <- setdiff(
    ALL_IDX,
    hold_idx
  )

  fold_relation <- NULL

  for (name in core_models) {
    ans <- fit_one_loo(
      model_name = name,
      hold_idx = hold_idx,
      train_idx = train_idx,
      relation = NULL
    )

    row <- ans[["row"]]
    learned <- ans[["learned"]]

    LOO_ROWS[[length(LOO_ROWS) + 1L]] <- row

    cat(
      "  ",
      sprintf("%-28s", name),
      " lpd=",
      sprintf("%+.4f", row[["exact_lpd_unweighted"]]),
      " weighted=",
      sprintf("%+.4f", row[["exact_lpd_participant_weighted"]]),
      "\n",
      sep = ""
    )

    if (identical(name, "Power_exponential_hybrid")) {
      if (is.null(learned)) {
        rp <- ridge_cache_path(hold_idx)

        if (!file.exists(rp)) {
          stop(
            "Missing fold ridge relation for fold ",
            hold_idx,
            "."
          )
        }

        learned <- readRDS(rp)
      }

      fold_relation <- learned

      RIDGE_FOLD_ROWS[[length(RIDGE_FOLD_ROWS) + 1L]] <-
        data.frame(
          constellation_id = hold_idx,
          as.data.frame(
            as.list(fold_relation),
            check.names = FALSE
          ),
          check.names = FALSE
        )
    }
  }

  if (is.null(fold_relation)) {
    stop(
      "Fold ",
      hold_idx,
      " did not produce a ridge relation."
    )
  }

  for (name in ridge_models) {
    ans <- fit_one_loo(
      model_name = name,
      hold_idx = hold_idx,
      train_idx = train_idx,
      relation = fold_relation
    )

    row <- ans[["row"]]

    LOO_ROWS[[length(LOO_ROWS) + 1L]] <- row

    cat(
      "  ",
      sprintf("%-28s", name),
      " lpd=",
      sprintf("%+.4f", row[["exact_lpd_unweighted"]]),
      " weighted=",
      sprintf("%+.4f", row[["exact_lpd_participant_weighted"]]),
      "\n",
      sep = ""
    )
  }
}

EXACT_POINTWISE <- do.call(
  rbind,
  LOO_ROWS
)

EXACT_POINTWISE <- EXACT_POINTWISE[
  order(
    EXACT_POINTWISE[["model"]],
    EXACT_POINTWISE[["constellation_id"]]
  ),
  ,
  drop = FALSE
]

rownames(EXACT_POINTWISE) <- NULL

utils::write.csv(
  EXACT_POINTWISE,
  file.path(OUTPUT_DIR, "exact_loo_pointwise.csv"),
  row.names = FALSE
)

RIDGE_RELATION_BY_FOLD <- do.call(
  rbind,
  RIDGE_FOLD_ROWS
)

rownames(RIDGE_RELATION_BY_FOLD) <- NULL

utils::write.csv(
  RIDGE_RELATION_BY_FOLD,
  file.path(OUTPUT_DIR, "ridge_relation_by_fold.csv"),
  row.names = FALSE
)


# =====================================================================
# 18. POINTWISE MATRICES
# =====================================================================

pointwise_vector <- function(model_name, column) {
  sub <- EXACT_POINTWISE[
    EXACT_POINTWISE[["model"]] == model_name,
    ,
    drop = FALSE
  ]

  sub <- sub[
    order(sub[["constellation_id"]]),
    ,
    drop = FALSE
  ]

  if (nrow(sub) != 32L) {
    stop(
      model_name,
      ": ",
      nrow(sub),
      " folds found; expected 32."
    )
  }

  as.numeric(sub[[column]])
}

LPD_UNWEIGHTED <- vapply(
  MODEL_NAMES,
  function(name) {
    pointwise_vector(
      name,
      "exact_lpd_unweighted"
    )
  },
  numeric(32L)
)

LPD_WEIGHTED <- vapply(
  MODEL_NAMES,
  function(name) {
    pointwise_vector(
      name,
      "exact_lpd_participant_weighted"
    )
  },
  numeric(32L)
)

PRED_MATRIX <- vapply(
  MODEL_NAMES,
  function(name) {
    pointwise_vector(
      name,
      "pred_mean"
    )
  },
  numeric(32L)
)

colnames(LPD_UNWEIGHTED) <- MODEL_NAMES
colnames(LPD_WEIGHTED) <- MODEL_NAMES
colnames(PRED_MATRIX) <- MODEL_NAMES


# =====================================================================
# 19. MODEL COMPARISON
# =====================================================================

total_se <- function(pointwise) {
  pointwise <- as.numeric(pointwise)

  sqrt(
    length(pointwise) *
      stats::var(pointwise)
  )
}

make_comparison <- function(L, weighted = FALSE) {
  total <- colSums(L)
  best_idx <- which.max(total)

  rows <- lapply(
    seq_along(MODEL_NAMES),
    function(j) {
      name <- MODEL_NAMES[[j]]
      diff <- L[, j] - L[, best_idx]
      residual <- BAG - PRED_MATRIX[, j]

      if (weighted) {
        rmse <- sqrt(
          sum(FULL_WEIGHTS * residual^2) /
            sum(FULL_WEIGHTS)
        )

        mae <- sum(
          FULL_WEIGHTS * abs(residual)
        ) / sum(FULL_WEIGHTS)

      } else {
        rmse <- sqrt(mean(residual^2))
        mae <- mean(abs(residual))
      }

      data.frame(
        model = name,
        criterion = if (weighted) {
          "participant_weighted"
        } else {
          "equal_constellation"
        },
        elpd = total[[j]],
        se_elpd_pointwise_approx = total_se(L[, j]),
        elpd_diff_vs_best = sum(diff),
        se_diff_pointwise_approx = total_se(diff),
        rough_2x_se_diff_small_N = 2 * total_se(diff),
        RMSE = rmse,
        MAE = mae,
        check.names = FALSE
      )
    }
  )

  out <- do.call(rbind, rows)

  out <- out[
    order(-out[["elpd"]]),
    ,
    drop = FALSE
  ]

  out <- data.frame(
    rank = seq_len(nrow(out)) - 1L,
    out,
    check.names = FALSE
  )

  rownames(out) <- NULL
  out
}

WEIGHTED_COMPARISON <- make_comparison(
  LPD_WEIGHTED,
  weighted = TRUE
)

UNWEIGHTED_COMPARISON <- make_comparison(
  LPD_UNWEIGHTED,
  weighted = FALSE
)

utils::write.csv(
  WEIGHTED_COMPARISON,
  file.path(
    OUTPUT_DIR,
    "exact_loo_model_comparison_PARTICIPANT_WEIGHTED_PRIMARY.csv"
  ),
  row.names = FALSE
)

utils::write.csv(
  UNWEIGHTED_COMPARISON,
  file.path(
    OUTPUT_DIR,
    "exact_loo_model_comparison_equal_constellation_sensitivity.csv"
  ),
  row.names = FALSE
)


# =====================================================================
# 20. EXACT-CV MODEL WEIGHTS
# =====================================================================

# Ordinary exact-CV stacking and pseudo-BMA+ use loo's standard functions
# directly. loo explicitly supports N x K pointwise exact-LOO/K-fold lpd
# matrices for stacking_weights() and pseudobma_weights().

set.seed(SEED_BASE + 8001L)

UNWEIGHTED_STACKING <- as.numeric(
  loo::stacking_weights(
    LPD_UNWEIGHTED,
    optim_method = "BFGS",
    optim_control = list(
      reltol = 1e-12,
      maxit = 10000L
    )
  )
)

names(UNWEIGHTED_STACKING) <- MODEL_NAMES

set.seed(SEED_BASE + 8002L)

UNWEIGHTED_BB <- as.numeric(
  loo::pseudobma_weights(
    LPD_UNWEIGHTED,
    BB = TRUE,
    BB_n = N_BB_MODEL_WEIGHTS,
    alpha = 1
  )
)

names(UNWEIGHTED_BB) <- MODEL_NAMES


# Participant-weighted pseudo-BMA+ can also use loo directly because the
# fixed observation weights enter linearly in each pointwise ELPD term.
# LPD_WEIGHTED already equals FULL_WEIGHTS[i] * LPD_UNWEIGHTED[i,model].

set.seed(SEED_BASE + 8000L)

WEIGHTED_BB <- as.numeric(
  loo::pseudobma_weights(
    LPD_WEIGHTED,
    BB = TRUE,
    BB_n = N_BB_MODEL_WEIGHTS,
    alpha = 1
  )
)

names(WEIGHTED_BB) <- MODEL_NAMES


# Participant-weighted stacking
# -----------------------------
# loo::stacking_weights() has no observation-weight argument. Here the fixed
# participant weights are proportional to the integer Table S7 counts:
#
#   FULL_WEIGHTS[i] = 32 * N_COUNTS[i] / 27375.
#
# Therefore maximizing
#
#   sum_i FULL_WEIGHTS[i] * log(mix_i)
#
# is equivalent (up to the positive constant 32/27375) to maximizing the
# ordinary stacking objective after repeating row i exactly N_COUNTS[i] times.
# This lets us use loo's tested stacking implementation directly and avoids a
# second hand-written optimizer. This replication is ONLY an optimization
# identity; it is not used as a participant-level likelihood or as an
# uncertainty/bootstrap sample size.

PARTICIPANT_STACKING_INDEX <- rep(
  seq_len(nrow(LPD_UNWEIGHTED)),
  times = as.integer(N_COUNTS)
)

stopifnot(length(PARTICIPANT_STACKING_INDEX) == N_TOTAL)

LPD_STACKING_PARTICIPANT_REPLICATED <- LPD_UNWEIGHTED[
  PARTICIPANT_STACKING_INDEX,
  ,
  drop = FALSE
]

WEIGHTED_STACKING <- as.numeric(
  loo::stacking_weights(
    LPD_STACKING_PARTICIPANT_REPLICATED,
    optim_method = "BFGS",
    optim_control = list(
      reltol = 1e-12,
      maxit = 10000L
    )
  )
)

names(WEIGHTED_STACKING) <- MODEL_NAMES

# Fail-fast validation. A legitimate stacking optimum cannot score worse than
# the best pure model because every simplex vertex is an admissible candidate.
participant_weighted_stacking_score <- function(model_weight) {
  w <- as.numeric(model_weight)

  if (length(w) != ncol(LPD_UNWEIGHTED) ||
      any(!is.finite(w)) ||
      any(w < -1e-10) ||
      abs(sum(w) - 1) > 1e-8) {
    stop("Invalid participant-weighted stacking weights returned by loo.")
  }

  w[w < 0] <- 0
  w <- w / sum(w)

  row_max <- apply(LPD_UNWEIGHTED, 1L, max)
  scaled_density <- exp(
    sweep(LPD_UNWEIGHTED, 1L, row_max, `-`)
  )
  mix_scaled <- as.numeric(scaled_density %*% w)

  if (any(!is.finite(mix_scaled)) || any(mix_scaled <= 0)) {
    stop("Non-finite predictive mixture in stacking validation.")
  }

  sum(
    FULL_WEIGHTS *
      (row_max + log(mix_scaled))
  )
}

WEIGHTED_STACKING_SCORE <- participant_weighted_stacking_score(
  WEIGHTED_STACKING
)

SINGLE_MODEL_WEIGHTED_SCORES <- colSums(
  sweep(
    LPD_UNWEIGHTED,
    1L,
    FULL_WEIGHTS,
    `*`
  )
)

BEST_SINGLE_WEIGHTED_SCORE <- max(SINGLE_MODEL_WEIGHTED_SCORES)
BEST_SINGLE_WEIGHTED_MODEL <- MODEL_NAMES[
  which.max(SINGLE_MODEL_WEIGHTED_SCORES)
]

if (WEIGHTED_STACKING_SCORE < BEST_SINGLE_WEIGHTED_SCORE - 1e-8) {
  stop(
    sprintf(
      paste0(
        "Participant-weighted stacking self-check failed: ",
        "mixture score %.12f is below best single-model score %.12f (%s)."
      ),
      WEIGHTED_STACKING_SCORE,
      BEST_SINGLE_WEIGHTED_SCORE,
      BEST_SINGLE_WEIGHTED_MODEL
    )
  )
}

WEIGHTED_STACKING_DIAGNOSTICS <- data.frame(
  stacking_score = WEIGHTED_STACKING_SCORE,
  best_single_model = BEST_SINGLE_WEIGHTED_MODEL,
  best_single_score = BEST_SINGLE_WEIGHTED_SCORE,
  stacking_improvement =
    WEIGHTED_STACKING_SCORE - BEST_SINGLE_WEIGHTED_SCORE,
  replicated_rows_for_optimization =
    length(PARTICIPANT_STACKING_INDEX),
  check.names = FALSE
)

utils::write.csv(
  WEIGHTED_STACKING_DIAGNOSTICS,
  file.path(
    OUTPUT_DIR,
    "participant_weighted_stacking_diagnostics.csv"
  ),
  row.names = FALSE
)

cat(
  "\nParticipant-weighted stacking self-check: PASS\n",
  sprintf(
    "  stacking score = %.12f\n",
    WEIGHTED_STACKING_SCORE
  ),
  sprintf(
    "  best single-model score = %.12f (%s)\n",
    BEST_SINGLE_WEIGHTED_SCORE,
    BEST_SINGLE_WEIGHTED_MODEL
  ),
  sprintf(
    "  stacking improvement = %+.12f\n",
    WEIGHTED_STACKING_SCORE - BEST_SINGLE_WEIGHTED_SCORE
  ),
  sep = ""
)

WEIGHTED_MODEL_WEIGHTS <- data.frame(
  model = MODEL_NAMES,
  stacking_weight = as.numeric(WEIGHTED_STACKING[MODEL_NAMES]),
  BB_pseudo_BMA_plus_weight = as.numeric(WEIGHTED_BB[MODEL_NAMES]),
  check.names = FALSE
)

UNWEIGHTED_MODEL_WEIGHTS <- data.frame(
  model = MODEL_NAMES,
  stacking_weight = as.numeric(UNWEIGHTED_STACKING[MODEL_NAMES]),
  BB_pseudo_BMA_plus_weight = as.numeric(UNWEIGHTED_BB[MODEL_NAMES]),
  check.names = FALSE
)

utils::write.csv(
  WEIGHTED_MODEL_WEIGHTS,
  file.path(
    OUTPUT_DIR,
    "model_weights_PARTICIPANT_WEIGHTED_PRIMARY.csv"
  ),
  row.names = FALSE
)

utils::write.csv(
  UNWEIGHTED_MODEL_WEIGHTS,
  file.path(
    OUTPUT_DIR,
    "model_weights_equal_constellation_sensitivity.csv"
  ),
  row.names = FALSE
)


# =====================================================================
# 21. BAYESIAN-BOOTSTRAP RANKING STABILITY
# =====================================================================

# Base R vectorized Bayesian bootstrap:
# Gamma(1,1) / rowSum is Dirichlet(1,...,1), so no extra package is needed.

bootstrap_ranking <- function(
  L_unweighted,
  obs_weight,
  B,
  seed
) {
  set.seed(seed)

  L <- as.matrix(L_unweighted)
  ow <- as.numeric(obs_weight)

  n <- nrow(L)
  K <- ncol(L)

  if (length(ow) != n) {
    stop("obs_weight length does not match L rows.")
  }

  raw <- matrix(
    stats::rgamma(
      B*n,
      shape = 1,
      rate = 1
    ),
    nrow = B,
    ncol = n
  )

  boot_w <- raw / rowSums(raw)

  weighted_L <- sweep(
    L,
    1L,
    ow,
    `*`
  )

  elpd_boot <- n * (boot_w %*% weighted_L)

  winner <- max.col(
    elpd_boot,
    ties.method = "first"
  )

  winner_frequency <- tabulate(
    winner,
    nbins = K
  ) / B

  winner_df <- data.frame(
    model = MODEL_NAMES,
    bootstrap_winner_frequency = winner_frequency,
    check.names = FALSE
  )

  winner_df <- winner_df[
    order(-winner_df[["bootstrap_winner_frequency"]]),
    ,
    drop = FALSE
  ]

  rownames(winner_df) <- NULL

  ref <- match(
    PRESPECIFIED_PRIMARY_MODEL,
    MODEL_NAMES
  )

  diff_rows <- lapply(
    seq_along(MODEL_NAMES),
    function(j) {
      diff <- elpd_boot[, j] - elpd_boot[, ref]

      q <- stats::quantile(
        diff,
        probs = c(0.025, 0.5, 0.975),
        names = FALSE
      )

      data.frame(
        model = MODEL_NAMES[[j]],
        reference_model = PRESPECIFIED_PRIMARY_MODEL,
        mean_diff = mean(diff),
        q2.5 = q[[1L]],
        median = q[[2L]],
        q97.5 = q[[3L]],
        P_model_gt_reference = mean(diff > 0),
        check.names = FALSE
      )
    }
  )

  list(
    winner = winner_df,
    differences = do.call(rbind, diff_rows)
  )
}

WEIGHTED_BOOT <- bootstrap_ranking(
  LPD_UNWEIGHTED,
  FULL_WEIGHTS,
  B = N_BOOTSTRAP_RANKING,
  seed = SEED_BASE + 9000L
)

UNWEIGHTED_BOOT <- bootstrap_ranking(
  LPD_UNWEIGHTED,
  rep(1, 32L),
  B = N_BOOTSTRAP_RANKING,
  seed = SEED_BASE + 9001L
)

WEIGHTED_WINNER_STABILITY <- WEIGHTED_BOOT[["winner"]]
WEIGHTED_BOOT_DIFF <- WEIGHTED_BOOT[["differences"]]

UNWEIGHTED_WINNER_STABILITY <- UNWEIGHTED_BOOT[["winner"]]
UNWEIGHTED_BOOT_DIFF <- UNWEIGHTED_BOOT[["differences"]]

utils::write.csv(
  WEIGHTED_WINNER_STABILITY,
  file.path(
    OUTPUT_DIR,
    "bootstrap_ranking_PARTICIPANT_WEIGHTED_PRIMARY.csv"
  ),
  row.names = FALSE
)

utils::write.csv(
  WEIGHTED_BOOT_DIFF,
  file.path(
    OUTPUT_DIR,
    "bootstrap_differences_PARTICIPANT_WEIGHTED_PRIMARY.csv"
  ),
  row.names = FALSE
)

utils::write.csv(
  UNWEIGHTED_WINNER_STABILITY,
  file.path(
    OUTPUT_DIR,
    "bootstrap_ranking_equal_constellation_sensitivity.csv"
  ),
  row.names = FALSE
)

utils::write.csv(
  UNWEIGHTED_BOOT_DIFF,
  file.path(
    OUTPUT_DIR,
    "bootstrap_differences_equal_constellation_sensitivity.csv"
  ),
  row.names = FALSE
)


# =====================================================================
# 22. EXACT-LOO MCMC DIAGNOSTICS
# =====================================================================

exact_diag_rows <- lapply(
  MODEL_NAMES,
  function(name) {
    sub <- EXACT_POINTWISE[
      EXACT_POINTWISE[["model"]] == name,
      ,
      drop = FALSE
    ]

    data.frame(
      model = name,
      folds = nrow(sub),
      worst_Rhat = max(sub[["max_Rhat"]], na.rm = TRUE),
      min_ESS_bulk = min(sub[["min_ESS_bulk"]], na.rm = TRUE),
      min_ESS_tail = min(sub[["min_ESS_tail"]], na.rm = TRUE),
      total_divergences = sum(sub[["divergences"]], na.rm = TRUE),
      total_max_treedepth_hits =
        sum(sub[["max_treedepth_hits"]], na.rm = TRUE),
      check.names = FALSE
    )
  }
)

EXACT_DIAGNOSTICS <- do.call(
  rbind,
  exact_diag_rows
)

rownames(EXACT_DIAGNOSTICS) <- NULL

utils::write.csv(
  EXACT_DIAGNOSTICS,
  file.path(OUTPUT_DIR, "exact_loo_mcmc_diagnostics.csv"),
  row.names = FALSE
)


# =====================================================================
# 23. BASE-R PLOTS
# =====================================================================

grDevices::png(
  file.path(
    OUTPUT_DIR,
    "posterior_mean_curves.png"
  ),
  width = 1500,
  height = 1000,
  res = 150
)

plot(
  NA,
  xlim = c(0, 1),
  ylim = range(
    POSTERIOR_CURVES[["mean"]],
    finite = TRUE
  ),
  xlab = "Weighted MetS score x",
  ylab = "BAG (years)",
  main = "Participant-weighted posterior mean curves"
)

for (j in seq_along(MODEL_NAMES)) {
  name <- MODEL_NAMES[[j]]

  sub <- POSTERIOR_CURVES[
    POSTERIOR_CURVES[["model"]] == name,
    ,
    drop = FALSE
  ]

  graphics::lines(
    sub[["x"]],
    sub[["mean"]],
    lty = j,
    lwd = 2
  )
}

graphics::legend(
  "topleft",
  legend = MODEL_NAMES,
  lty = seq_along(MODEL_NAMES),
  lwd = 2,
  cex = 0.75,
  bty = "n"
)

grDevices::dev.off()


plot_df <- WEIGHTED_COMPARISON
y <- rev(seq_len(nrow(plot_df)))
elpd <- plot_df[["elpd"]]
se <- plot_df[["se_elpd_pointwise_approx"]]

grDevices::png(
  file.path(
    OUTPUT_DIR,
    "exact_loo_PARTICIPANT_WEIGHTED_PRIMARY.png"
  ),
  width = 1600,
  height = 1000,
  res = 150
)

graphics::plot(
  elpd,
  y,
  xlim = range(
    c(elpd - se, elpd + se),
    finite = TRUE
  ),
  ylim = c(0.5, length(y) + 0.5),
  yaxt = "n",
  xlab = "Participant-weighted exact-LOO score (higher is better)",
  ylab = "",
  main = "Primary participant-weighted exact-LOO comparison",
  pch = 19
)

graphics::arrows(
  x0 = elpd - se,
  y0 = y,
  x1 = elpd + se,
  y1 = y,
  angle = 90,
  code = 3,
  length = 0.04
)

graphics::axis(
  side = 2,
  at = y,
  labels = plot_df[["model"]],
  las = 1,
  tick = FALSE
)

graphics::abline(
  v = max(elpd),
  lty = 3
)

grDevices::dev.off()


# =====================================================================
# 24. ANALYSIS MANIFEST / SESSION INFO
# =====================================================================

manifest <- list(
  program =
    "mets_brain_age_participant_weighted_reference_rstan_table_s7_final_fixed.R",
  run_hash = RUN_HASH,
  fresh_run = FRESH_RUN,
  data_source =
    "Exact two-decimal BAG LS Mean values and participant counts from Table S7",
  bag_field_used =
    "BAG LS Mean (years), not beta coefficient",
  primary_estimand =
    "participant-weighted constellation prediction",
  full_fit_weight_formula =
    "w_i = 32*n_i/27375",
  fold_fit_weight_formula =
    "w_i = G*n_i/sum_training(n), with G=31 in each LOO fold",
  primary_evaluation_formula =
    "weighted_lpd_i = (32*n_i/27375) * heldout_lpd_i",
  sensitivity_analysis =
    paste(
      "equal-constellation evaluation of predictions from",
      "participant-weighted training fits"
    ),
  prespecified_primary_model =
    PRESPECIFIED_PRIMARY_MODEL,
  models = MODEL_NAMES,
  implementation = c(
    "RStan stan_model() compilation once per candidate model",
    "RStan sampling() reuses compiled stanmodel objects for all exact-LOO refits",
    "RStan extract()/monitor()/HMC diagnostic helpers",
    "stats::lm() for ridge regressions",
    "loo::stacking_weights()/pseudobma_weights() where directly applicable",
    "loo::stacking_weights() for participant-weighted stacking via exact count-proportional row replication"
  ),
  limitations = c(
    paste(
      "The 32 BAG values are adjusted aggregate estimates,",
      "not individual-level BAG observations."
    ),
    paste(
      "Participant-count weighting is an intentional pseudo-likelihood",
      "weighting rule, not replication of 27,375 BAG outcomes."
    ),
    paste(
      "The covariance matrix among the 32 adjusted BAG estimates",
      "is not modeled because it is unavailable."
    ),
    paste(
      "Historical adaptive model development on these same 32 data",
      "points cannot be undone by code."
    ),
    "Independent validation requires new external data."
  ),
  full_chains = FULL_CHAINS,
  full_num_samples = FULL_NUM_SAMPLES,
  full_num_warmup = FULL_NUM_WARMUP,
  cv_chains = CV_CHAINS,
  cv_num_samples = CV_NUM_SAMPLES,
  cv_num_warmup = CV_NUM_WARMUP,
  adapt_delta = ADAPT_DELTA,
  max_treedepth = MAX_TREEDEPTH,
  smooth_a_eps = SMOOTH_A_EPS
)

saveRDS(
  manifest,
  file.path(OUTPUT_DIR, "analysis_manifest.rds"),
  compress = "xz"
)

capture.output(
  dput(manifest),
  file = file.path(
    OUTPUT_DIR,
    "analysis_manifest.txt"
  )
)

capture.output(
  sessionInfo(),
  file = file.path(
    OUTPUT_DIR,
    "sessionInfo.txt"
  )
)


# =====================================================================
# 25. CONSOLE SUMMARY
# =====================================================================

cat(
  "\n",
  paste(rep("=", 76L), collapse = ""),
  "\nPARTICIPANT-WEIGHTED RSTAN ANALYSIS COMPLETE\n",
  paste(rep("=", 76L), collapse = ""),
  "\n",
  sep = ""
)

cat(
  "\nPRIMARY participant-weighted exact-LOO comparison:\n"
)

print(
  WEIGHTED_COMPARISON,
  row.names = FALSE,
  digits = 6
)

cat(
  "\nEqual-constellation evaluation sensitivity comparison:\n"
)

print(
  UNWEIGHTED_COMPARISON,
  row.names = FALSE,
  digits = 6
)

cat(
  "\nPRIMARY participant-weighted model weights:\n"
)

print(
  WEIGHTED_MODEL_WEIGHTS,
  row.names = FALSE,
  digits = 6
)

cat(
  "\nParticipant-weighted bootstrap ranking stability:\n"
)

print(
  WEIGHTED_WINNER_STABILITY,
  row.names = FALSE,
  digits = 6
)

cat(
  "\nFull-data ridge relation:\n"
)

print(
  as.data.frame(
    as.list(FULL_RIDGE_RELATION),
    check.names = FALSE
  ),
  row.names = FALSE,
  digits = 6
)

cat(
  "\nFull-data MCMC diagnostics:\n"
)

print(
  FULL_DIAGNOSTICS,
  row.names = FALSE,
  digits = 6
)

cat(
  "\nExact-LOO MCMC diagnostics:\n"
)

print(
  EXACT_DIAGNOSTICS,
  row.names = FALSE,
  digits = 6
)

cat(
  "\nOutput directory:\n  ",
  normalizePath(
    OUTPUT_DIR,
    winslash = "/",
    mustWork = FALSE
  ),
  "\n",
  sep = ""
)

cat(
  "\nPrimary weighting rule:\n",
  "  full fit/evaluation: w_i = 32*n_i/27375\n",
  "  each LOO training fit: w_i = 31*n_i/sum_training(n)\n",
  sep = ""
)

cat(
  "\nImportant sensitivity-label note:\n",
  "  The equal-constellation table uses equal weights only at evaluation.\n",
  "  Its models were still trained with participant-count weighting.\n",
  sep = ""
)

cat(
  "\nR / package versions:\n",
  "  R: ", R.version.string, "\n",
  "  rstan: ", as.character(utils::packageVersion("rstan")), "\n",
  "  loo: ", as.character(utils::packageVersion("loo")), "\n",
  "  run hash: ", RUN_HASH, "\n",
  sep = ""
)
