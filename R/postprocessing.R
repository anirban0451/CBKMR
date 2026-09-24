cbkmr_postprocess = function(res, ...){
  extra_args <- list(...)

  pip_threshold <- extra_args$pip_t
  if(is.null(pip_threshold)){
    pip_threshold = 0.7
  }

  thin <- extra_args$thin
  if(is.null(thin)){
    thin = 2
  }

  delta1 <- as.numeric(res$delta)
  dim(delta1) <- dim(res$delta)

  non_z_entries = which(colMeans(delta1) > pip_threshold)

  # Safety check to prevent subsetting crashes if no genes pass PIP threshold
  if(length(non_z_entries) == 0){
    warning("No features exceeded the PIP threshold. Returning empty active set.")
    return(list(
      pip = colMeans(delta1),
      selected = integer(0),
      post_rel = numeric(0),
      post_beta = colMeans(res$Beta),
      post_tau = mean(res$tau)
    ))
  }

  wmat = res$wmat[, non_z_entries, drop = FALSE]

  # FIXED: Swapped ncol() for nrow() to sequence over MCMC iterations
  wmat = wmat[seq(1, nrow(res$wmat), by = thin), , drop = FALSE]
  betamat = res$Beta[seq(1, nrow(res$Beta), by = thin), , drop = FALSE]
  tau_thinned = res$tau[seq(1, nrow(res$tau), by = thin), , drop = FALSE]

  # Return the whole relevance vector
  post_rel = rep(0, ncol(delta1))
  post_rel[non_z_entries] = colMeans(as.matrix(wmat))

  return(list(
    pip = colMeans(delta1),
    selected = non_z_entries,
    post_rel = post_rel,
    post_beta = colMeans(betamat),
    post_tau = as.numeric(mean(tau_thinned))
  ))
}

average_precision_score <- function(y_true, y_score) {
  # Input guards
  stopifnot(length(y_true) == length(y_score))
  stopifnot(all(y_true %in% c(0, 1, NA)))

  # Remove NAs
  keep <- !is.na(y_score) & !is.na(y_true)
  y_true  <- y_true[keep]
  y_score <- y_score[keep]

  n_pos <- sum(y_true == 1)

  if (n_pos == 0L) {
    warning("No positive labels: AP is undefined.")
    return(NA_real_)
  }

  if (n_pos == length(y_true)) {
    return(1.0)
  }

  # Break ties pessimistically: negatives before positives at same score
  ord          <- order(y_score, -y_true, decreasing = TRUE)
  y_true_sorted <- y_true[ord]

  tp           <- cumsum(y_true_sorted == 1)
  fp           <- cumsum(y_true_sorted == 0)
  precision    <- tp / (tp + fp)
  recall       <- tp / n_pos
  recall_diff  <- diff(c(0, recall))

  return(sum(recall_diff * precision))
}

cbkmr_predict = function(res, X = NULL, Z, y, new_X = NULL, new_Z, new_y = NULL){

  if(ncol(Z) != ncol(new_Z)){
    stop("Dimensions between training and new data are different.\n")
  }

  if(is.vector(new_Z)){
    new_Z = matrix(new_Z, nrow = 1, byrow = TRUE)
  }

  p <- ncol(Z)
  N <- nrow(Z)       # FIXED: Was ncol(Z)
  N_new <- nrow(new_Z)

  processed_res <- cbkmr_postprocess(res)
  post_beta <- processed_res$post_beta
  post_tau <- processed_res$post_tau

  wvec <- rep(0, p)
  if (length(processed_res$selected) > 0) {
    wvec[processed_res$selected] <- processed_res$post_rel
  }

  # Handle Covariates
  if(!is.null(X)){
    X = cbind(1, X)
  } else {
    X = matrix(1, ncol = 1, nrow = N)
  }

  if(!is.null(new_X)){
    new_X = cbind(1, new_X)
  } else {
    new_X = matrix(1, ncol = 1, nrow = N_new)
  }

  # Scale Z
  Z_trans <- scale(Z)
  new_Z_trans <- scale(new_Z, center = attr(Z_trans, "scaled:center"),
                       scale = attr(Z_trans, "scaled:scale"))

  # Calculate Marginals
  eta <- as.numeric(X %*% post_beta)
  new_eta <- as.numeric(new_X %*% post_beta)

  probs <- 1 / (1 + exp(-eta))
  new_probs <- 1 / (1 + exp(-new_eta))

  # Latent Continuous Transform (Training Data)
  Fy <- ifelse(y == 0, (1 - probs)/2, 1 - probs/2)
  z_observed <- qnorm(Fy)

  # ---------------------------------------------------------
  # KRIGING: Conditional Mean and Variance (Incorporating Tau)
  # ---------------------------------------------------------
  Kmat <- CBKMR:::kernel_mat_RBF_rcpp_openmp(Z_trans, wvec)

  # Sigma_N = tau * K + (1 - tau) * I
  Sigma_N <- post_tau * Kmat + diag(1 - post_tau, N)

  # Sigma_cross = tau * K_cross
  newKmat <- CBKMR:::kernel_cross_RBF_rcpp_openmp(new_Z_trans, Z_trans, wvec)
  Sigma_cross <- post_tau * newKmat

  # Conditional Mean (mu*)
  Sigma_N_inv_z <- solve(Sigma_N, z_observed)
  condmean <- as.numeric(Sigma_cross %*% Sigma_N_inv_z)

  # Conditional Variance (Sigma*) - Extracting only the diagonal for individual predictions
  # Mathematical trick: rowSums(A * B) efficiently gets the diagonal of A %*% t(B)
  cross_inv_cross <- rowSums((Sigma_cross %*% solve(Sigma_N)) * Sigma_cross)

  # The variance of a new observation in this space is exactly 1
  condvar <- 1 - cross_inv_cross
  condsd <- sqrt(pmax(condvar, 1e-8)) # Protect against numerical underflow

  # ---------------------------------------------------------
  # COPULA PREDICTION: Conditional PMF Integration
  # ---------------------------------------------------------

  # For Y = 1 (Success)
  F_y1 <- 1 - new_probs/2
  z_y1 <- qnorm(F_y1)
  # Multiply Conditional Copula Density by the Marginal for Y = 1
  val_1 <- (dnorm(z_y1, mean = condmean, sd = condsd) / dnorm(z_y1)) * new_probs

  # For Y = 0 (Failure)
  F_y0 <- (1 - new_probs)/2
  z_y0 <- qnorm(F_y0)
  # Multiply Conditional Copula Density by the Marginal for Y = 0
  val_0 <- (dnorm(z_y0, mean = condmean, sd = condsd) / dnorm(z_y0)) * (1 - new_probs)

  # Normalize to create proper probabilities
  condprob_1 <- val_1 / (val_1 + val_0)
  predclass <- ifelse(condprob_1 > 0.5, 1, 0)

  if(is.null(new_y)){
    pr_score = NULL
  }else{
    pr_score = average_precision_score(y_true = new_y, y_score = condprob_1)
  }
  return(list(
    pred = predclass,
    pr_score = pr_score,
    postprob = condprob_1
  ))
}
