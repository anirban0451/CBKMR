update_beta0_using_ESS_no_X = function(current_beta0, y, Ut,
                                       beta0_prior_mean = 0, beta0_prior_sd, ...){

  # Capture all ... arguments
  extra_args <- list(...)

  # Extract thres, providing a default if it wasn't passed
  if (!is.null(extra_args$thres)) {
    thres <- extra_args$thres
  } else {
    thres <- 10 # Default value
  }


  # --- 1. Define the LOG-LIKELIHOOD function (Prior is handled by proposal) ---
  log_likelihood = function(b0){
    return(lCBpdf_bernoulli_no_X(y=y, beta0=b0, Ut=Ut))
  }

  # --- 2. Elliptical Slice Sampler ---

  # Step 1: Define the ellipse
  # Draw from the prior: N(mean, sd)
  # enforce bounds
  while(1){
    nu = rnorm(1, mean = beta0_prior_mean, sd = beta0_prior_sd)
    if (isTRUE(nu >= -thres & nu <= thres)){
      break
    }
  }

  # Step 3: Define the slice height (using likelihood ONLY)
  logL_current = log_likelihood(current_beta0)
  log_threshold = logL_current + log(runif(1))

  # Step 4: Propose and Shrink (on the angle phi)
  phi = runif(1, min = 0, max = 2 * pi) # Initial angle proposal
  phi_min = phi - 2 * pi
  phi_max = phi

  while(TRUE) {
    # Propose a new point *on the ellipse*
    proposed_beta0 = (current_beta0 - beta0_prior_mean) * cos(phi) +
      (nu - beta0_prior_mean) * sin(phi) +
      beta0_prior_mean
    # proposed_beta0 = pmax(pmin(proposed_beta0, thres), -thres) # Enforce bounds

    logL_proposed = log_likelihood(proposed_beta0)

    if (logL_proposed > log_threshold) {
      return(proposed_beta0) # Accept
    } else {
      # Shrink the angular bracket
      if (phi > 0) {
        phi_max = phi
      } else {
        phi_min = phi
      }
      phi = runif(1, min = phi_min, max = phi_max)
    }
  }
}

update_beta0_with_discrete_grid_no_X = function(current_beta0, y, Ut,
                                                beta0_prior_mean = 0, beta0_prior_sd = 10, ...){

  beta_grid = current_beta0 + seq(-2, 2, by = 0.1)

  log_likelihoods = sapply(beta_grid, function(b0) {
    lCBpdf_bernoulli_no_X(y=y, beta0=b0, Ut=Ut)
  })

  log_prior = dnorm(beta_grid, mean = beta0_prior_mean, sd = beta0_prior_sd, log = TRUE)
  log_posterior = log_likelihoods + log_prior

  posterior_probs = exp(log_posterior - max(log_posterior))
  posterior_probs = posterior_probs / sum(posterior_probs)

  new_beta0 = sample(beta_grid, size = 1, prob = posterior_probs)
  return(new_beta0)
}

update_beta0_with_MH_no_X = function(current_beta0, y, Ut,
                                     beta0_prior_mean = 0, beta0_prior_sd = 10, proposal_sd = 0.5, ...){

  # get the thres from ... arguments
  extra_args <- list(...)
  if (!is.null(extra_args$thres)) {
    thres <- extra_args$thres
  } else {
    thres <- 10 # Default value
  }

  # Propose a new beta0 using a normal random walk
  proposed_beta0 = rnorm(1, mean = current_beta0, sd = proposal_sd)
  prosed_beta0 = pmax(pmin(proposed_beta0, thres), -thres) # Enforce bounds

  # Calculate log-likelihoods
  loglik_current = lCBpdf_bernoulli_no_X(y=y, beta0=current_beta0, Ut=Ut)
  loglik_proposed = lCBpdf_bernoulli_no_X(y=y, beta0=proposed_beta0, Ut=Ut)

  # Calculate log-priors
  logprior_current = dnorm(current_beta0, mean = beta0_prior_mean, sd = beta0_prior_sd, log = TRUE)
  logprior_proposed = dnorm(proposed_beta0, mean = beta0_prior_mean, sd = beta0_prior_sd, log = TRUE)

  # Calculate acceptance ratio
  log_alpha = (loglik_proposed + logprior_proposed) - (loglik_current + logprior_current)

  # Accept or reject the proposal
  if (log(runif(1)) < log_alpha) {
    return(proposed_beta0) # Accept
  } else {
    return(current_beta0) # Reject
  }
}
