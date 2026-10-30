# MetS Table S7: exact-LOO RStan extension for two-exponential response curves
# Repository: https://github.com/AndersH3/MetS
# Source: Dove et al. (2026), DOI: 10.1002/alz.71563, Table S7.
#
# This is an incremental three-model RStan run. It does NOT rerun or rewrite
# the seven previously published models or their cached output.
#
# Three separate interpretations of the proposed
#     A*exp(B*x+C) + (1-A)*exp(D*x+E)
# are tested:
#  1. Unrestricted A (the LITERAL user equation, A is any real number).
#  2. Convex mixture (0<A<1); this model is always strictly positive.
#  3. Convex mixture plus intercept m; can predict negative BAG values.
#
# A is weakly identifiable jointly with C and E (only A*exp(C) and
# (1-A)*exp(E) enter the likelihood). Proper priors regularize, but do
# not eliminate, the weak identification. Slopes are ordered B>D to
# remove permutation symmetry. Inspect HMC diagnostics carefully.
#
# As in the seven-model reference analysis, this uses 32 adjusted aggregate
# BAG LS means and normalized participant weights -- NOT individual data.
# Evaluation is exact leave-one-CONSTELLATION-out, not PSIS-LOO.
#
# Run:
#   install.packages("rstan")    # if needed; requires C++ toolchain
#   source("mets_double_exponential_rstan_comparison.R")
#
# To resume an interrupted run, leave FRESH_RUN <- FALSE.
# Original fixed 7-model results are retained as a separate baseline table.

FRESH_RUN <- FALSE
OUTPUT_DIR <- "mets_double_exponential_table_s7_rstan_results"
CACHE_DIR <- file.path(OUTPUT_DIR, "exact_loo_cache")
STAN_DIR <- file.path(OUTPUT_DIR, "stan_models")
FULL_CHAINS <- 4L
FULL_SAMPLES <- 3000L
FULL_WARMUP <- 3000L
CV_CHAINS <- 4L
CV_SAMPLES <- 2000L
CV_WARMUP <- 1500L
ADAPT_DELTA <- 0.99
MAX_TREEDEPTH <- 15L
SEED_BASE <- 2026081902L

MODELS <- c("Two_exp_unrestricted_A", "Two_exp_convex_A", "Two_exp_convex_A_offset")

M <- rbind(
  c(0,0,0,0,0), c(0,0,0,0,1), c(1,0,0,0,1), c(0,0,1,0,0),
  c(0,0,0,1,0), c(0,0,0,1,1), c(1,0,0,0,0), c(1,0,1,0,0),
  c(1,0,1,1,0), c(0,1,0,0,0), c(0,0,1,1,0), c(1,0,0,1,1),
  c(1,0,1,0,1), c(1,0,0,1,0), c(0,1,0,1,0), c(0,1,1,0,0),
  c(1,0,1,1,1), c(1,1,0,0,0), c(0,1,0,0,1), c(0,1,0,1,1),
  c(1,1,1,0,0), c(1,1,0,1,0), c(0,1,1,1,0), c(0,0,1,1,1),
  c(1,1,0,0,1), c(1,1,0,1,1), c(0,0,1,0,1), c(1,1,1,1,0),
  c(0,1,1,1,1), c(0,1,1,0,1), c(1,1,1,1,1), c(1,1,1,0,1)
)
storage.mode(M) <- "double"
colnames(M) <- c("central_adiposity", "blood_pressure", "hyperglycemia",
                 "triglycerides", "low_HDL")
BAG <- c(
 -0.45,-0.39,-0.22,-0.18,-0.19,0.09,0.10,0.14,
  0.34,0.42,0.54,0.69,0.73,0.75,0.73,0.86,
  0.93,0.94,0.85,0.91,1.19,1.25,1.30,1.35,
  1.35,1.58,1.62,2.04,2.18,2.19,2.47,2.77
)
N_COUNTS <- c(
  4850,834,172,288,948,569,497,51,31,5388,84,226,47,244,1987,437,
  63,1094,2034,1845,186,857,252,75,740,1257,98,204,465,451,727,374
)
stopifnot(nrow(M)==32L, ncol(M)==5L, length(BAG)==32L,
          length(N_COUNTS)==32L, sum(N_COUNTS)==27375L,
          all(M %in% c(0,1)), all(is.finite(BAG)))
ALL_IDX <- seq_len(32L)
FULL_WEIGHTS <- 32*N_COUNTS/sum(N_COUNTS)
participant_weights <- function(idx) {
  w <- length(idx)*N_COUNTS[idx]/sum(N_COUNTS[idx])
  stopifnot(isTRUE(all.equal(sum(w),as.numeric(length(idx)))))
  as.numeric(w)
}
make_data <- function(idx) list(
  G=length(idx), M=M[idx,,drop=FALSE], bag=BAG[idx],
  obs_weight=participant_weights(idx)
)

if (!requireNamespace("rstan",quietly=TRUE))
  stop("Install rstan and configure an RStan-compatible C++ toolchain first.")
rstan::rstan_options(auto_write=TRUE)
options(mc.cores=min(4L,parallel::detectCores()))
if (FRESH_RUN && dir.exists(OUTPUT_DIR))
  unlink(OUTPUT_DIR,recursive=TRUE,force=TRUE)
for (path in c(OUTPUT_DIR,CACHE_DIR,STAN_DIR))
  dir.create(path,recursive=TRUE,showWarnings=FALSE)

# Priors:
#   mets ~ Dirichlet(1,1,1,1,1) (same as reference)
#   sigma ~ half-Student-t(4,0,1) (same as reference)
#   A ~ Normal(1,2), if unrestricted; Beta(2,2), if convex
#   B,D ~ Normal(0,2) with B>D imposed
#   C,E ~ Normal(-1,1.5)
#   m ~ Normal(0,3), only for offset model
# The new-model priors must be reported with comparisons; they are not
# specified by the original seven-model family.
stan_code <- function(name) {
  unrestricted <- identical(name,"Two_exp_unrestricted_A")
  offset <- identical(name,"Two_exp_convex_A_offset")
  a_decl <- if (unrestricted) "real A;" else "real<lower=0,upper=1> A;"
  a_prior <- if (unrestricted) "A ~ normal(1, 2);" else "A ~ beta(2, 2);"
  m_decl <- if (offset) "real m;" else ""
  m_prior <- if (offset) "m ~ normal(0, 3);" else ""
  m_piece <- if (offset) "m + " else ""
  paste0(
'data {
  int<lower=1> G;
  matrix[G,5] M;
  vector[G] bag;
  vector<lower=0>[G] obs_weight;
}
parameters {
  simplex[5] mets;
  ',a_decl,'
  ordered[2] slopes;
  real C;
  real E;
  ',m_decl,'
  real<lower=0> sigma;
}
transformed parameters {
  real B = slopes[2];
  real D = slopes[1];
  vector[G] x = M * mets;
  vector[G] mu;
  for (g in 1:G)
    mu[g] = ',m_piece,'A*exp(B*x[g]+C) + (1-A)*exp(D*x[g]+E);
}
model {
  mets ~ dirichlet(rep_vector(1.0,5));
  ',a_prior,'
  slopes ~ normal(0,2);
  C ~ normal(-1,1.5);
  E ~ normal(-1,1.5);
  ',m_prior,'
  sigma ~ student_t(4,0,1);
  for (g in 1:G)
    target += obs_weight[g]*normal_lpdf(bag[g] | mu[g],sigma);
}
')
}
MODEL_CODE <- setNames(lapply(MODELS,stan_code),MODELS)
MODEL_FILES <- setNames(file.path(STAN_DIR,paste0(MODELS,".stan")),MODELS)
for (nm in MODELS) writeLines(MODEL_CODE[[nm]],MODEL_FILES[[nm]],useBytes=TRUE)
settings <- c(paste0(MODELS,"=",unname(tools::md5sum(MODEL_FILES))),
  paste(FULL_CHAINS,FULL_SAMPLES,FULL_WARMUP,CV_CHAINS,CV_SAMPLES,
        CV_WARMUP,ADAPT_DELTA,MAX_TREEDEPTH,SEED_BASE,collapse=","),
  "Table-S7-exact-2026-08-21", "priors-A-N1,2-Beta2,2-slope-N0,2-C-E-N-1,1.5")
sigfile <- tempfile()
writeLines(settings,sigfile)
RUN_HASH <- substr(unname(tools::md5sum(sigfile)),1,16)
unlink(sigfile)
cat("Model-run hash:",RUN_HASH,"\n")
cat("Compiling",length(MODELS),"Stan models...\n")
STAN_MODELS <- setNames(lapply(MODELS,function(nm)
  rstan::stan_model(file=MODEL_FILES[[nm]],
                    model_name=paste0("mets_",tolower(nm)))),MODELS)

saved_pars <- function(name)
  c("mets","A","B","C","D","E","sigma",
    if (identical(name,"Two_exp_convex_A_offset")) "m")
fit_model <- function(name,idx,seed,full=FALSE) {
  chains <- if (full) FULL_CHAINS else CV_CHAINS
  warmup <- if (full) FULL_WARMUP else CV_WARMUP
  samples <- if (full) FULL_SAMPLES else CV_SAMPLES
  rstan::sampling(
    STAN_MODELS[[name]],data=make_data(idx),pars=saved_pars(name),
    chains=chains,iter=warmup+samples,warmup=warmup,thin=1L,
    seed=as.integer(seed),cores=chains,
    refresh=if (full) 500L else 0L,
    control=list(adapt_delta=ADAPT_DELTA,max_treedepth=MAX_TREEDEPTH)
  )
}
extract_post <- function(fit,name) {
  post <- rstan::extract(fit,pars=saved_pars(name),permuted=TRUE,
                         inc_warmup=FALSE)
  if (is.null(post[["m"]])) post[["m"]] <- rep(0,length(post[["A"]]))
  post
}
pred_mu <- function(post,patterns) {
  x <- post[["mets"]] %*% t(as.matrix(patterns))
  out <- matrix(post[["m"]],nrow=nrow(x),ncol=ncol(x)) +
    matrix(post[["A"]],nrow=nrow(x),ncol=ncol(x))*
      exp(matrix(post[["B"]],nrow=nrow(x),ncol=ncol(x))*x +
          matrix(post[["C"]],nrow=nrow(x),ncol=ncol(x))) +
    (1-matrix(post[["A"]],nrow=nrow(x),ncol=ncol(x)))*
      exp(matrix(post[["D"]],nrow=nrow(x),ncol=ncol(x))*x +
          matrix(post[["E"]],nrow=nrow(x),ncol=ncol(x)))
  out
}
log_mean_exp <- function(z) {
  v <- max(z)
  if (!is.finite(v)) return(v)
  v+log(mean(exp(z-v)))
}
score_holdout <- function(post,i) {
  p <- as.vector(pred_mu(post,M[i,,drop=FALSE]))
  list(lpd=log_mean_exp(stats::dnorm(BAG[i],mean=p,
                                      sd=post[["sigma"]],log=TRUE)),
       prediction=mean(p))
}
diag_fit <- function(fit,name,fold=NA_integer_) {
  s <- rstan::summary(fit,pars=saved_pars(name))$summary
  data.frame(model=name,fold=fold,
    max_Rhat=max(s[,"Rhat"],na.rm=TRUE),
    min_n_eff=min(s[,"n_eff"],na.rm=TRUE),
    divergences=sum(rstan::get_num_divergent(fit)),
    max_treedepth_hits=sum(rstan::get_num_max_treedepth(fit)),
    check.names=FALSE)
}
full_diag <- list()
fold_diag <- list()
full_param <- list()
full_curves <- list()
folds <- list()
for (j in seq_along(MODELS)) {
  nm <- MODELS[[j]]
  cat("\nFULL FIT:",nm,"\n")
  fit_path <- file.path(OUTPUT_DIR,paste0(nm,"_full_draws_",RUN_HASH,".rds"))
  diag_path <- file.path(OUTPUT_DIR,paste0(nm,"_full_diag_",RUN_HASH,".rds"))
  if (!FRESH_RUN && file.exists(fit_path) && file.exists(diag_path)) {
    post <- readRDS(fit_path)
    fdiag <- readRDS(diag_path)
  } else {
    fit <- fit_model(nm,ALL_IDX,SEED_BASE+10L*j,full=TRUE)
    post <- extract_post(fit,nm)
    fdiag <- diag_fit(fit,nm)
    saveRDS(post,fit_path,compress="xz")
    saveRDS(fdiag,diag_path)
    rm(fit);gc(verbose=FALSE)
  }
  full_diag[[j]] <- fdiag
  parvecs <- c("A","B","C","D","E","m","sigma")
  full_param[[j]] <- do.call(rbind,lapply(parvecs,function(p)
    data.frame(model=nm,parameter=p,
       mean=mean(post[[p]]),sd=stats::sd(post[[p]]),
       q2.5=unname(stats::quantile(post[[p]],0.025)),
       q97.5=unname(stats::quantile(post[[p]],0.975)))))
  xx <- seq(0,1,length.out=201)
  pm <- post[["m"]]; A <- post[["A"]]
  full_curves[[j]] <- do.call(rbind,lapply(xx,function(x) {
    yy <- pm + A*exp(post[["B"]]*x+post[["C"]]) +
      (1-A)*exp(post[["D"]]*x+post[["E"]])
    data.frame(model=nm,x=x,mean=mean(yy),
      lower=unname(stats::quantile(yy,0.025)),
      upper=unname(stats::quantile(yy,0.975)))
  }))
  rm(post);gc(verbose=FALSE)
  for (i in ALL_IDX) {
    cache <- file.path(CACHE_DIR,
      sprintf("%s_fold_%02d_%s.rds",nm,i,RUN_HASH))
    if (!FRESH_RUN && file.exists(cache)) {
      record <- readRDS(cache)
    } else {
      cat("  fold",sprintf("%02d/32",i),"\n")
      train <- setdiff(ALL_IDX,i)
      fit <- fit_model(nm,train,SEED_BASE+1000000L+10000L*j+i)
      post <- extract_post(fit,nm)
      held <- score_holdout(post,i)
      record <- list(
        row=data.frame(model=nm,constellation_id=i,
          participant_count=N_COUNTS[i],BAG=BAG[i],
          lpd=held$lpd,weighted_lpd=FULL_WEIGHTS[i]*held$lpd,
          heldout_prediction=held$prediction,
          residual=BAG[i]-held$prediction,
          check.names=FALSE),
        diag=diag_fit(fit,nm,fold=i)
      )
      saveRDS(record,cache,compress="xz")
      rm(fit,post);gc(verbose=FALSE)
    }
    folds[[length(folds)+1L]] <- record$row
    fold_diag[[length(fold_diag)+1L]] <- record$diag
  }
}
POINTWISE <- do.call(rbind,folds)
FULL_DIAG <- do.call(rbind,full_diag)
FOLD_DIAG <- do.call(rbind,fold_diag)
FULL_PARAMS <- do.call(rbind,full_param)
CURVES <- do.call(rbind,full_curves)
for (a in list(
  list(POINTWISE,"exact_loo_pointwise_new_models.csv"),
  list(FULL_DIAG,"full_data_mcmc_diagnostics.csv"),
  list(FOLD_DIAG,"exact_loo_mcmc_diagnostics.csv"),
  list(FULL_PARAMS,"full_data_parameter_summary.csv"),
  list(CURVES,"posterior_curves_new_models.csv")
)) utils::write.csv(a[[1]],file.path(OUTPUT_DIR,a[[2]]),row.names=FALSE)
se_pointwise <- function(q) sqrt(length(q)*stats::var(q))
new_comparison <- do.call(rbind,lapply(MODELS,function(nm) {
  sub <- POINTWISE[POINTWISE$model==nm,]
  data.frame(model=nm,
    ELPD=sum(sub$weighted_lpd),
    SE_ELPD=se_pointwise(sub$weighted_lpd),
    RMSE=sqrt(sum(FULL_WEIGHTS*sub$residual^2)/sum(FULL_WEIGHTS)),
    MAE=sum(FULL_WEIGHTS*abs(sub$residual))/sum(FULL_WEIGHTS))
}))
new_equal <- do.call(rbind,lapply(MODELS,function(nm) {
  sub <- POINTWISE[POINTWISE$model==nm,]
  data.frame(model=nm,ELPD=sum(sub$lpd),
    SE_ELPD=se_pointwise(sub$lpd),
    RMSE=sqrt(mean(sub$residual^2)),
    MAE=mean(abs(sub$residual)))
}))
# Previously committed 7-model scores; kept as historical reference, not
# refitted in this run. The new-model ELPDs are calculated above using the
# identical weight normalization and predictive log-score definitions.
reference <- data.frame(
  model=c("Normalized_exponential","Direct_p_ridge","Hard_logp_ridge",
          "Linear","Power","Soft_logp_ridge","Power_exponential_hybrid"),
  ELPD=c(2.64180,2.31923,2.30284,1.70159,1.50031,1.48019,1.28052),
  SE_ELPD=rep(NA_real_,7L),
  RMSE=c(.21723,.22165,.22115,.22515,.21734,.22053,.21936),
  MAE=c(.12722,.13807,.13485,.14150,.12839,.13328,.12738))
comparison <- rbind(reference,new_comparison)
comparison$ELPD_diff_vs_reference_exp <- comparison$ELPD-2.64180
comparison <- comparison[order(-comparison$ELPD),]
comparison$rank <- seq_len(nrow(comparison))
comparison <- comparison[,c("rank","model","ELPD","ELPD_diff_vs_reference_exp",
                          "SE_ELPD","RMSE","MAE")]
utils::write.csv(comparison,
 file.path(OUTPUT_DIR,"comparison_with_previous_seven_PARTICIPANT_WEIGHTED.csv"),
 row.names=FALSE)
utils::write.csv(new_equal,
 file.path(OUTPUT_DIR,"new_models_equal_constellation_sensitivity.csv"),
 row.names=FALSE)
writeLines(c(
  paste("Run hash:",RUN_HASH),
  "Existing seven model scores are historical committed RStan results.",
  "New models are refitted using exact LOO on Table S7 BAG LS means.",
  "Weighted held-out score = (32*n_i/27375)*log mean posterior density.",
  "Training weights = G*n_i / sum_training(n_i), G = 31 in LOO folds.",
  "Strictly positive convex literal mixture cannot predict negative BAG.",
  "The unrestricted-A model is the literal user equation without an intercept.",
  "All two-exponential fits have weak A,C,E identification; inspect diagnostics.",
  "SE_ELPD for reference models not recalculated here (NA).",
  "No causal interpretation or individual-level likelihood is justified."),
 file.path(OUTPUT_DIR,"run_notes.txt"))
utils::capture.output(sessionInfo(),file=file.path(OUTPUT_DIR,"sessionInfo.txt"))
print(comparison,row.names=FALSE,digits=6)
cat("\nEqual-constellation sensitivity (new models only):\n")
print(new_equal,row.names=FALSE,digits=6)
cat("\nHMC diagnostics: full fits\n")
print(FULL_DIAG,row.names=FALSE)
cat("\nWorst fold Rhat:",max(FOLD_DIAG$max_Rhat,na.rm=TRUE),
    "; total divergences:",sum(FOLD_DIAG$divergences),"\n")
cat("\nFiles written to:",normalizePath(OUTPUT_DIR),"\n")
