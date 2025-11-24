#' Find ordered nearest neighbors with weighted locations
#' @param locs An n x p matrix of locations (assumed to be ordered)
#' @param w A vector of length p of weights for each dimension
#' @param m The number of nearest neighbors to find
#' @param wthres A threshold below which weights are considered negligible
#' @return An n x m matrix of nearest neighbor indices (1-based)
#' @examples
#' \dontrun{
#' locs <- matrix(runif(100), nrow=20)  # 20 points in 5D
#' w <- runif(5)  # weights for each dimension
#' m <- 3         # number of neighbors
#' nn_ind <- find_ordered_nn_wvec(locs, w, m)
#' }
find_ordered_nn_wvec <- function(locs, w, m, wthres=1e-4){
  # assume locs are already ordered
  if(all(w < wthres)){w = rep(wthres, length(w))}
  nonz_dim = which(c(w) > wthres)
  locs = locs[, nonz_dim, drop=FALSE]
  nn_ind = find_ordered_nn(t(sqrt(w[nonz_dim])*t(locs)), m = m)[,-1]
  return(nn_ind)
}

update_r_delta_joint_distribution_transform_rnngp <- function(delta, w,  y, Z,  eta, K, tau, a.p0, b.p0, p, N,
                                          r.params, thres, Acc1, Acc2, i, nngp_obj){

  Z_NN1 = nngp_obj$nn_ind

  r_new <-  r.star <- w
  delta_new <- delta
  move_type_probs <- c(0.5, 0.5)                       # can be altered based on iteration if needed
  move_type <-  sample(1:2, 1, prob = move_type_probs) # decides if should we flip a delta_m and update r_m, or update a random r_m for which delta_m is 1
  move.prob <-  ifelse(all(delta == 0), 1, 1/2)        # flip a delta_m w.p. 1 if all deltas are 0

  probs = exp(eta)/(1+exp(eta))
  # F_y <- sapply(1:n, function(i){
  #   ifelse(y[i] == 0, (1-probs[i])/2, 1-probs[i]/2)
  # })
  F_y = ifelse(y == 0, (1 - probs)/2, 1 - probs/2)

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

    Z_NN2 = find_ordered_nn_wvec(locs = Z[nngp_obj$ordering, , drop=FALSE], w = r.star, m = nngp_obj$m)

    ll_new <- rnngp_loglik_tauonly(x=F_y[nngp_obj$ordering], coords=Z[nngp_obj$ordering, , drop=FALSE], w = r.star, neighbor_matrix = Z_NN2, tau = tau)
    ll_old <- rnngp_loglik_tauonly(x=F_y[nngp_obj$ordering], coords=Z[nngp_obj$ordering, , drop=FALSE], w = w, neighbor_matrix = Z_NN1, tau = tau)

    log_ratio <- (ll_new - ll_old) + diffpriors +  negdifflogproposal
    log_ratio[is.na(log_ratio)] <- 0
    logalpha <- min(0,log_ratio)

    if (log(runif(1)) <  logalpha) {
      delta_new <- delta.star
      r_new <- r.star
      Acc1[i, comp] <- 1
      nngp_obj$nn_ind = Z_NN2
    }
  }

  # --- Move 2: Perturb a selected active r ---
  if (move_type == 2 && sum(delta) > 0) {
    active_indices <- which(delta == 1)
    comp <- sample(active_indices, 1)

    # next MH steps are similar as Move 1
    r.star[comp] <- rprop.gen2(current = w[comp], r.params = r.params)

    Z_NN2 = find_ordered_nn_wvec(locs = Z[nngp_obj$ordering, , drop=FALSE], w = r.star, m = nngp_obj$m)

    ll_new <- rnngp_loglik_tauonly(x=F_y[nngp_obj$ordering], coords=Z[nngp_obj$ordering, , drop=FALSE], w = r.star, neighbor_matrix = Z_NN2, tau = tau)
    ll_old <- rnngp_loglik_tauonly(x=F_y[nngp_obj$ordering], coords=Z[nngp_obj$ordering, , drop=FALSE], w = w, neighbor_matrix = Z_NN1, tau = tau)

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
      Acc2[i, comp] <- 1
      nngp_obj$nn_ind = Z_NN2
    }
  }

  return(list(w = r_new, delta = delta_new, K = K,  Acc1 = Acc1,
              Acc2 = Acc2, F_y = F_y, nngp_obj = nngp_obj))
}

logistic_BKMR_dt_correct_beta_rnngp <- function(y, Z = Z, nsim = 5000,  verbose = TRUE, thres = 10, neighbor_size = 20, beta0_scheme = 1){

  Z = scale(Z)  # standardize Z

  extra_args <- list(...)
  if (!is.null(extra_args$seed)) {
    seed <- extra_args$seed
  } else {
    seed <- 1234
  }

  set.seed(seed)

  # set initial ordering using GpGp package
  Zord = order_maxmin(Z)
  m = neighbor_size
  nngp_obj = list(ordering = Zord, m = m)

  if(missing(beta0_scheme)) {
    beta0_scheme = 1
  }
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

  rnngp_loglik_tauonly_use_AD <- rnngp_loglik_tauonly_use_AD

  r.params <- list(r.a = 0, r.b = 5, r.jump2 = 0.5)

  N <- length(y)            # Number of samples
  p <- ncol(Z)              # Number of features inside kernel

  nsim<-nsim	              # Number of MCMC Iterations
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
  tau <- 0.5                             # initialize tau
  rho <- 1
  rho.var <- 5
  accrho <- 0
  h <- h_star <- rep(0, N)
  w <- rep(1, p)                          # initialize inverse lengthscales, r_m's, calling the vector w instead of r
  w[sample.int(p, size = floor(0.6*p))] = 0  # initialize with 50% of the variables excluded]
  w_NB <- rep(1, N)                       # initialize Polya-Gamma (PG) weights
  z <- rep(1, N)                          # initialize latent factors
  delta <- rep(1, p)                      # initialize spike and slab indicator, all variables are included at the start
  lambda0 <- rep(1, p)
  a.p0 <- b.p0 <- 1                       # prior params for delta

  K <- NULL
  nngp_obj$nn_ind = find_ordered_nn_wvec(locs = Z[nngp_obj$ordering, , drop=FALSE], w = w, m = nngp_obj$m)

  if(verbose == T){print("Starting MCMC.")}

  for (i in 1:nsim){

    #update eta
    #########################
    eta <- rep(beta0, N)

    #update inverse lengthscales (r_m's) and delta_m's
    ##################################################
    out <- update_r_delta_joint_distribution_transform_rnngp(delta, w,  y, Z,  eta, K, tau, a.p0, b.p0, p, N,
                                         r.params, thres, Acc1, Acc2, i, nngp_obj = nngp_obj)
    w <- out$w
    delta <- out$delta

    Acc1 <- out$Acc1
    Acc2 <- out$Acc2

    F_y <- out$F_y

    Z_nn = out$nngp_obj$nn_ind

    nngp_obj = out$nngp_obj

    ###### update tau ########
    #########################

    tau_prop <- rnorm(1, mean = tau, sd = 0.25) # propose a new value for tau using a Normal rw
    loglik_curr_storage <- rnngp_loglik_tauonly_use_AD(x=F_y[Zord], coords=Z[Zord, , drop=FALSE], w = w, neighbor_matrix = Z_nn, tau = tau)

    loglik_curr <- loglik_curr_storage$log_density
    Ut <- loglik_curr_storage[[2]]

    # Reject immediately if outside (0, 1)
    if (tau_prop >= 0 && tau_prop <= 1) {

      # Log-priors (currently using Beta(1, 1), uniform — can generalize)
      logprior_prop <- dbeta(tau_prop, shape1 = 1, shape2 = 1, log = TRUE)
      logprior_curr <- dbeta(tau, shape1 = 1, shape2 = 1, log = TRUE)

      loglik_prop_storage <- rnngp_loglik_tauonly_use_AD(x=F_y[Zord], coords=Z[Zord, , drop=FALSE], w = w, neighbor_matrix = Z_nn, tau = tau_prop)
      loglik_prop <- loglik_prop_storage$log_density

      log_alpha <- (loglik_prop + logprior_prop) - (loglik_curr + logprior_curr)

      if (log(runif(1)) < log_alpha) {
        tau <- tau_prop
        Ut <- loglik_prop_storage[[2]]
      }

    }

    ## update beta0
    #########################
    beta0 <- update_beta0_fn(current_beta0 = beta0, y = y[Zord], Ut = Ut,
                             beta0_prior_mean = 0, beta0_prior_sd = 10, thres = thres)


    if (i> burn & i%%thin==0) {
      j<-(i-burn)/thin
      Beta[j,]<-beta0
      tau_mat[j,]<-tau
      wmat[j, ]<-w
      delmat[j, ]<-delta
    }
    if(isTRUE(update_ordering && i%%update_steps == 0)){
      # veccs_obj = vecchia_respecify(locs = Z, rel = w, m = 30, ordering = "maxmin", conditioning = "mra", ic0=TRUE)
      # K = Sig.sel.rel(1, veccs_obj)
      which_nonz = which(delta > 0)
      w_sub = w[which_nonz]
      set.seed(202507)
      Zord = GpGp::order_maxmin(t(w_sub*t(Z[, which_nonz, drop=FALSE])))
      nngp_obj$ordering = Zord
      nngp_obj$nn_ind = find_ordered_nn_wvec(locs = Z[nngp_obj$ordering, , drop=FALSE], w = w, m = nngp_obj$m)
    }
    if(verbose == "TRUE"){
      svMisc::progress(i, nsim, progress.bar = FALSE)
    }else{if(i%%500 == 0){print(paste0(i, " / ", nsim))}
    }
  }

  return(NB = list(Beta = Beta, tau =  tau_mat,  wmat = wmat, delta = delmat,
                   Acc1 = Acc1, Acc2 = Acc2))
}
