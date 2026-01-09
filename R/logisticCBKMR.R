update_r_delta_joint_distribution_transform <- function(delta, w,  y, Z,  eta, K, tau, a.p0, b.p0, p, N,
                                                        r.params, thres, Acc1, Acc2, i,
                                                        rprior.logdens, rprop.gen1, rprop.logdens1, rprop.gen2, rprop.logdens2){

  n <- length(y)
  r_new <-  r.star <- w
  delta_new <- delta
  move_type_probs <- c(0.5, 0.5)                       # can be altered based on iteration if needed
  move_type <-  sample(1:2, 1, prob = move_type_probs) # decides if should we flip a delta_m and update r_m, or update a random r_m for which delta_m is 1
  move.prob <-  ifelse(all(delta == 0), 1, 1/2)        # flip a delta_m w.p. 1 if all deltas are 0

  probs = exp(eta)/(1+exp(eta))
  F_y <- sapply(1:n, function(i){
    ifelse(y[i] == 0, (1-probs[i])/2, 1-probs[i]/2)
  })
  # F_y <- pbinom(y, size = 1, prob = probs) # NB CDF
  F_y <- pmin(pmax(F_y, 1e-12), 1-1e-12)                     # avoid extremes
  F_y <- qnorm(F_y)                                        # inverse of standard normal CDF

  # --- Move 1: Flip a random delta ---
  if (move_type == 1) {
    comp <- sample(1:p, 1)
    delta.star <- delta
    delta.star[comp] <- 1 - delta[comp]
    move.prob.star <- ifelse(all(delta.star == 0), 1, 1/2)

    # MH steps start to update the r_m, the priors and proposals are functions of (r, delta)
    r.star[comp] <- ifelse(delta.star[comp] == 0, 0, rprop.gen1(r.params = r.params))
    diffpriors <- (lgamma(sum(delta.star) + a.p0) + lgamma(p - sum(delta.star) + b.p0) -
                     lgamma(sum(delta) + a.p0) - lgamma(p - sum(delta) + b.p0)) +
      ifelse(delta[comp] == 1, -1, 1)* rprior.logdens(x = ifelse(delta[comp] == 1,
                                                                 w[comp], r.star[comp]), r.params = r.params)

    negdifflogproposal <- -log(move.prob.star) + log(move.prob) -
      ifelse(delta[comp] == 1, -1, 1)*with(list(r.sel =
                                                  ifelse(delta[comp] == 1, w[comp], r.star[comp])),
                                           rprop.logdens1(x = r.sel, r.params = r.params))

    K_new <- kernel_mat_RBF_rcpp_openmp(Z, r.star)            # new kernel matrix

    ll_new <- logdmvn_arma(F_y, K_new*tau + diag(1 - tau, N)) # these two are comp. intensive steps
    ll_old <- logdmvn_arma(F_y, K*tau + diag(1 - tau, N))

    log_ratio <- (ll_new - ll_old) + diffpriors +  negdifflogproposal
    log_ratio[is.na(log_ratio)] <- 0
    logalpha <- min(0,log_ratio)

    if (log(runif(1)) <  logalpha) {
      delta_new <- delta.star
      r_new <- r.star
      K <- K_new
      Acc1[i, comp] <- 1
    }
  }

  # --- Move 2: Perturb a selected active r ---
  if (move_type == 2 && sum(delta) > 0) {
    active_indices <- which(delta == 1)
    comp <- sample(active_indices, 1)

    # next MH steps are similar as Move 1
    r.star[comp] <- rprop.gen2(current = w[comp], r.params = r.params)

    K_new <- kernel_mat_RBF_rcpp_openmp(Z, r.star)
    ll_new <- logdmvn_arma(F_y, K_new*tau + diag(1 - tau, N))
    ll_old <- logdmvn_arma(F_y, K*tau + diag(1 - tau, N))

    diffpriors <- rprior.logdens(r.star[comp], r.params = r.params) -
      rprior.logdens(w[comp], r.params = r.params)

    negdifflogproposal <- -rprop.logdens2(r.star[comp], w[comp],
                                          r.params = r.params) + rprop.logdens2(w[comp],
                                                                                r.star[comp], r.params = r.params)

    negdifflogproposal[is.na(negdifflogproposal)] <- 0
    log_ratio <- (ll_new - ll_old) + diffpriors + negdifflogproposal
    log_ratio[is.na(log_ratio)] <- 0

    logalpha <- min(0,log_ratio)

    if (log(runif(1)) <  logalpha) {
      r_new <- r.star
      K <- K_new
      Acc2[i, comp] <- 1
    }
  }

  return(list(w = r_new, delta = delta_new, K = K,  Acc1 = Acc1,
              Acc2 = Acc2, F_y = F_y))
}


#' CBKMR for logistic regression (with logit link) without additional covariates
#' @param y An N-dimensional vector of binary responses (0/1)
#' @param Z An N x p matrix of features of interest, must be standardized
#' @param nsim Number of MCMC iterations
#' @param verbose Logical, whether to print progress updates
#' @param thres Threshold for numerical stability in CDF calculations
#' @param beta0_scheme Scheme for updating beta0: 1 = Metropolis-Hastings,
#' 2 = Elliptical Slice Sampling, 3 = Discrete Grid
#' @return A list containing MCMC samples and acceptance rates
#' @examples
#' \dontrun{
#' # Simulate some data
#' set.seed(123)
#' N <- 100
#' p <- 10
#' Z <- matrix(rnorm(N * p), nrow = N, ncol = p)
#' Z_scaled <- scale(Z)
#' true_w <- c(rep(1, 5), rep(0, 5))
#' K <- exp(-as.matrix(dist(Z_scaled))^2 %*% diag(true_w))
#' eta <- rep(0, N)
#' probs <- 1 / (1 + exp(-eta))
#' y <- rbinom(N, size = 1, prob = probs)
#'
#' # Run CBKMR
#' result <- logisticCBKMR(y, Z_scaled, nsim = 1000, beta0_scheme = 1)
#' }
#' @export
logisticCBKMR <- function(y, Z, nsim = 5000,  verbose = TRUE, thres = 10, beta0_scheme, ...){

  extra_args <- list(...)
  if (!is.null(extra_args$seed)) {
    seed <- extra_args$seed
  } else {
    seed <- 1234
  }

  if (!is.null(extra_args$priordist)) {
    priordist <- extra_args$priordist
    if (!(priordist %in% c("uniform", "gamma", "hcauchy", "invunif"))) {
      stop("Invalid prior distribution specified. Choose from 'uniform', 'dgamma', 'dhcauchy', or 'invunif'.")
    }
  } else {
    priordist = "uniform"
  }

  rprior.logdens <- switch(priordist,
                           "uniform" = rprior.logdens.unif,
                           "gamma" = rprior.logdens.dgamma,
                           "hcauchy" = rprior.logdens.dhcauchy,
                           "invunif" = rprior.logdens.invunif)

  rprop.gen1 <- rprop.gen1.unif
  rprop.logdens1 <- rprop.logdens1.unif
  rprop.gen2 <- rprop.gen2.truncnorm
  rprop.logdens2 <- rprop.logdens2.truncnorm

  if (!is.null(extra_args$r.a)) {
    r.a <- extra_args$r.a
  } else {
    r.a <- 0
  }
  if (!is.null(extra_args$r.b)) {
    r.b <- extra_args$r.b
  } else {
    r.b <- 5
  }
  if (!is.null(extra_args$r.jump2)) {
    r.jump2 <- extra_args$r.jump2
  } else {
    r.jump2 <- 0.5
  }
  if(!is.null(extra_args$mu.r)){
    mu.r <- extra_args$mu.r
  }else{
    mu.r <- 2
  }
  if(!is.null(extra_args$sigma.r)){
    sigma.r <- extra_args$sigma.r
  }else{
    sigma.r <- 1
  }

  set.seed(seed)

  r.params <- list(r.a = r.a, r.b = r.b, r.jump2 = r.jump2, mu.r = mu.r, sigma.r = sigma.r)

  if(beta0_scheme == 1){
    # update through random walk Metropolis
    update_beta0_fn <- update_beta0_with_MH_no_X
  }else if(beta0_scheme == 3){
    # update through discrete grid
    update_beta0_fn <- update_beta0_with_discrete_grid_no_X
  }else if(beta0_scheme == 2){
    # update through ESS
    update_beta0_fn <- update_beta0_using_ESS_no_X
  }else{
    stop("Invalid scheme selected for updating beta0.")
  }

  N <- length(y)            # Number of samples
  p <- ncol(Z)              # Number of features inside kernel

  thin<-1				            # Thinning interval
  burn<-nsim/2		          # Burnin
  lastit<-(nsim-burn)/thin	# Last stored value

  # Store
  Beta<-matrix(0, lastit, 1)              # only intercept, can be generalized for covariates
  Acc1 <-  Acc2 <-  matrix(0, nsim, p)    # keeps track of RJ-MCMC acceptance probabilities
  wmat<-delmat<-matrix(0, lastit, p)      # stores inverse length-scales and deltas
  tau_mat <- matrix(0, lastit, 1)
  #Init
  beta0 <- rep(0, 1)
  tau <- 0.5                              # initial value of tau
  rho <- 1
  rho.var <- 5
  accrho <- 0
  h <- h_star <- rep(0, N)
  w <- rep(1, p)                          # initialize inverse lengthscales, r_m's, calling the vector w instead of r
  w[sample(1:p, size = floor(0.6*p))] <- 0  # set 60% of the r_m's to zero at the start]
  z <- rep(1, N)                          # initialize latent factors
  delta <- rep(1, p)                      # initialize spike and slab indicator, all variables are included at the start
  lambda0 <- rep(1, p)
  a.p0 <- b.p0 <- 1                       # prior params for delta


  K <- kernel_mat_RBF_rcpp_openmp(Z, w)   # initial kernel matrix
  if(verbose == T){print("Starting MCMC.")}


  for (i in 1:nsim){

    #update eta
    #########################
    eta <- rep(beta0, N)


    #update inverse lengthscales (r_m's) and delta_m's
    ##################################################
    out <- update_r_delta_joint_distribution_transform(delta, w,  y, Z,  eta, K, tau, a.p0, b.p0, p, N,
                                           r.params, thres, Acc1, Acc2, i,
                                           rprior.logdens = rprior.logdens,
                                           rprop.gen1 = rprop.gen1,
                                           rprop.logdens1 = rprop.logdens1,
                                           rprop.gen2 = rprop.gen2,
                                           rprop.logdens2 = rprop.logdens2)
    w <- out$w
    delta <- out$delta
    K <- out$K
    Acc1 <- out$Acc1
    Acc2 <- out$Acc2
    F_y <- out$F_y

    ###### update tau ########
    #########################
    tau_prop <- rnorm(1, mean = tau, sd = 0.25) # propose a new value for tau using a Normal rw

    # Reject immediately if outside (0, 1)
    if (tau_prop >= 0 && tau_prop <= 1) {
      # Log-priors (currently using Beta(1, 1), uniform — can generalize)
      logprior_prop <- dbeta(tau_prop, shape1 = 1, shape2 = 1, log = TRUE)
      logprior_curr <- dbeta(tau, shape1 = 1, shape2 = 1, log = TRUE)

      # Log-likelihoods with current and proposal Sigma matrices
      Sigma_prop <- K * tau_prop + diag(1 - tau_prop, N)
      Sigma_curr <- K * tau + diag(1 - tau, N)

      loglik_prop_store <- logdmvn_arma_with_U(F_y, Sigma_prop)
      loglik_curr_store <- logdmvn_arma_with_U(F_y, Sigma_curr)
      loglik_prop <- loglik_prop_store$log_density
      loglik_curr <- loglik_curr_store$log_density

      log_alpha <- (loglik_prop + logprior_prop) - (loglik_curr + logprior_curr)
      if (log(runif(1)) < log_alpha) {
        tau <- tau_prop
        Ut <- loglik_prop_store$Ut
      }else{
        Ut <- loglik_curr_store$Ut
      }

    }else{
      if(i == 1){
        Sigma_curr <- K * tau + diag(1 - tau, N)
        loglik_curr_store <- logdmvn_arma_with_U(F_y, Sigma_curr)
        Ut = loglik_curr_store$Ut
      }
    }

    #update beta0
    #########################
    beta0 <- update_beta0_fn(current_beta0 = beta0, y = y, Ut = Ut,
                             beta0_prior_mean = 0, beta0_prior_sd = 10, thres = thres)

    if (i> burn & i%%thin==0) {
      j<-(i-burn)/thin
      Beta[j,]<-beta0
      tau_mat[j,]<-tau
      wmat[j, ]<-w
      delmat[j, ]<-delta
    }
    if(verbose){
      svMisc::progress(i, nsim, progress.bar = FALSE)
    }else{if(i%%500 == 0){print(paste0(i, " / ", nsim))}
    }
  }

  mcmc.setup.details <- list(priordistn = priordist, thin = thin, burn = burn, lastit = lastit,
                             r.params = r.params, seed = seed)

  return(NB = list(Beta = Beta, tau =  tau_mat,  wmat = wmat, delta = delmat,
                   Acc1 = Acc1, Acc2 = Acc2, mcmc.setup.details = mcmc.setup.details))
}
