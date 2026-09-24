generate_toy_data <- function(N, p, iter, design, eff = 1, num_nonz = 6, ker_var = 1, ...) {

  # Ensure p is large enough for the hardcoded designs
  if (design == 2 && p < 3) stop("Design 2 requires at least p = 3.")
  if (design == 3 && p < 6) stop("Design 3 requires at least p = 6.")
  if (design == 1 && p < num_nonz) stop("p must be >= num_nonz for Design 1.")

  # Set seed dynamically based on iter
  set.seed(2024 * iter / 2)

  # ============================================================================
  # 1. Generate Feature Matrix Z
  # ============================================================================
  Sigma_Z <- diag(1, N) # feature covariance, assuming independent across samples

  Z <- list()
  dframe <- NULL
  for(i in 1:p){
    Z[[i]] <- c(mvnfast::rmvn(n = 1, mu = rep(0, N), sigma = Sigma_Z))

    # Optional dependency mapping kept intact from original code
    if(i > 1){
      dframe <- cbind(dframe, Z[[i - 1]])
      Z[[i]]  <- lm(Z[[i]] ~ dframe - 1)$residuals
    }
  }

  Z <- do.call(cbind, Z)  # final feature matrix, N x p
  Z_trans <- scale(Z)     # scaling is a must for VS problems

  # ============================================================================
  # 2. Simulate Binary Outcome
  # ============================================================================
  y_logis <- rep(0, N)

  # Loop to ensure the simulated y is not too sparse or entirely 1's/0's
  for(ind in 1:50){
    if(sum(y_logis) <= floor(N*0.05) | sum(y_logis) > floor(N*0.95)){

      # Route based on selected design
      if (design == 1) {

        # Design 1: Latent GP Random Effect Model
        # Note: Assumes kernel_mat_RBF_rcpp_openmp is defined in your environment
        non_z <- rep(eff, num_nonz)
        Kmat <- kernel_mat_RBF_rcpp_openmp(Z_trans, c(non_z, rep(0, p - num_nonz)))

        eta <- 0 + c(mvnfast::rmvn(1, mu = rep(0, N), sigma = ker_var * Kmat))

      } else if (design == 2) {

        # Design 2: Specific non-linear parametric form (3 features)
        extra_args <- list(...)
        sqcoeff = ifelse(is.null(extra_args$b), 0.5, 0.75)
        eta <- 1.5 * Z_trans[,1] +
          sqcoeff * (Z_trans[,2])^2 +
          2 * sin(1.5 * Z_trans[,3])

      } else if (design == 3) {

        # Design 3: Complex non-linear parametric form (6 features)
        eta <- 1.5 * Z_trans[,1] +
          0.75 * (Z_trans[,2])^2 +
          2 * sin(1.5 * Z_trans[,3]) +
          (1 / (Z_trans[,4])^2) +
          3 * sin((Z_trans[,5])^3) +
          exp(-sin((Z_trans[,6])^2))

        # Apply the default caseind == 2 multiplier
        eta <- 2 * eta

      } else {
        stop("Invalid design parameter. Must be 1, 2, or 3.")
      }

      # Convert linear predictor to probabilities and sample binary outcome
      p0 <- exp(eta) / (1 + exp(eta)) # Same as 1 / (1 + exp(-eta))
      y_logis <- c(rbinom(N, size = 1, prob = p0))

    } else {
      # Data is balanced enough, break the rejection loop
      break
    }
  }

  # Return data as a named list
  return(list(
    Z = Z,
    y_logis = y_logis
  ))
}


create_corrmat = function(p, ...){
  set.seed(123)
  if(p %% 5 != 0 && p > 0){
    stop("Function is optimized for 5k features.\n Supply accordingly.")
  }else if(p < 0){
    stop("Please use a positive value of p!!")
  }
  extra_args = list(...)
  if(!is.null(extra_args$block1lim)){
    block1lim = extra_args$block1lim
    if(length(block1lim) < 2){
      stop("You need to supply both upper and lower limit.")
    }
    b1l = block1lim[1]
    b1u = block1lim[2]
  }else{
    b1l = 0.2
    b1u = 0.5
  }

  if(!is.null(extra_args$block2lim)){
    block2lim = extra_args$block2lim
    if(length(block2lim) < 2){
      stop("You need to supply both upper and lower limit.")
    }
    b2l = block2lim[1]
    b2u = block2lim[2]
  }else{
    b2l = -0.1
    b2u = 0.1
  }
  if(!is.null(extra_args$block3lim)){
    block3lim = extra_args$block3lim
    if(length(block3lim) < 2){
      stop("You need to supply both upper and lower limit.")
    }
    b3l = block3lim[1]
    b3u = block3lim[2]
  }else{
    b3l = -0.5
    b3u = -0.2
  }
  # Define the index ranges for the 3 blocks to create the reversed-L shapes
  b1 <- 1:(2*p/5)    # Core positive block (top left)
  b2 <- (2*p/5 + 1):(3*p/5)   # Weak correlation reversed-L (middle band)
  b3 <- (3*p/5 + 1):p   # Strong negative reversed-L (outer band)

  # Initialize the matrix
  R <- matrix(0, p, p)

  # ==============================================================================
  # 1. BLOCK 1: Core
  # ==============================================================================
  R[b1, b1] <- runif(length(b1)^2, b1l, b1u)

  # ==============================================================================
  # 2. BLOCK 2: The inner reversed-L
  # ==============================================================================

  R[b1, b2] <- runif(length(b1) * length(b2), b2l, b2u)
  R[b2, b1] <- t(R[b1, b2])


  R[b2, b2] <- runif(length(b2)^2, b2l, b2u)

  # ==============================================================================
  # 3. BLOCK 3: The outer reversed-L
  # ==============================================================================

  R[c(b1, b2), b3] <- runif((length(b1) + length(b2)) * length(b3), b3l, b3u)
  R[b3, c(b1, b2)] <- t(R[c(b1, b2), b3])


  R[b3, b3] <- runif(length(b3)^2, b3l, b3u)

  # ==============================================================================
  # 4. ENFORCE SYMMETRY & POSITIVE DEFINITENESS
  # ==============================================================================
  # Copy lower triangle to upper to ensure perfect symmetry
  R[upper.tri(R)] <- t(R)[upper.tri(R)]
  diag(R) <- 1

  # Project the raw matrix to the nearest valid Positive Definite correlation matrix
  # Note: This will slightly shrink the extreme negative values to ensure mathematical validity
  R_pd <- as.matrix(Matrix::nearPD(R, corr = TRUE, maxit = 1000)$mat)
  return(R_pd)
}


generate_sparse_nb_data <- function(N, p, iter, eff = 1, ker_var = 1, ...) {

  message("This function assumes 4 nonzero features.\n Please modify the source code for any customization.")

  if(p %% 5 != 0 && p > 0){
    stop("Function is optimized for 5k features.\n Supply accordingly.")
  }else if(p < 0){
    stop("Really?!!!?")
  }
  extra_args = list(...)
  if(!is.null(extra_args$block1lim)){
    block1lim = extra_args$block1lim
    if(length(block1lim) < 2){
      stop("You need to supply both upper and lower limit.")
    }
    b1l = block1lim[1]
    b1u = block1lim[2]
  }else{
    b1l = 0.2
    b1u = 0.4
  }

  if(!is.null(extra_args$block2lim)){
    block2lim = extra_args$block2lim
    if(length(block2lim) < 2){
      stop("You need to supply both upper and lower limit.")
    }
    b2l = block2lim[1]
    b2u = block2lim[2]
  }else{
    b2l = -0.1
    b2u = 0.1
  }
  if(!is.null(extra_args$block3lim)){
    block3lim = extra_args$block3lim
    if(length(block3lim) < 2){
      stop("You need to supply both upper and lower limit.")
    }
    b3l = block3lim[1]
    b3u = block3lim[2]
  }else{
    b3l = -0.5
    b3u = -0.1
  }
  Sigma_gene <- create_corrmat(p = p,
                               block1lim = c(b1l, b1u),
                               block2lim = c(b2l, b2u),
                               block3lim = c(b3l, b3u))

  wvec = rep(0, p)
  non_z_entries = c(1, 3, ((3*p/5)+1), ((3*p/5)+5))
  wvec[non_z_entries] = eff

  # ==============================================================================
  # 1. GENERATE CONTINUOUS LATENT SPACE (Single continuous cloud)
  # ==============================================================================
  # No artificial shifting; all cells come from the same global distribution
  set.seed(2024 * iter / 2)
  U_latent <- mvnfast::rmvn(N, mu = rep(0, p), sigma = Sigma_gene + 1e-6 * diag(1, nrow(Sigma_gene), ncol(Sigma_gene)))

  # ==============================================================================
  # 2. NEGATIVE BINOMIAL TECHNICAL NOISE
  # ==============================================================================
  beta_0 <- 1.0    # Global baseline expression scalar
  theta <- 1.0     # NB dispersion (lower = higher technical noise)

  Z_counts <- matrix(0, N, p)

  for(j in 1:p) {
    mu_j <- exp(beta_0 + U_latent[, j])
    Z_counts[, j] <- rnbinom(N, size = theta, mu = mu_j)
  }

  # ==============================================================================
  # 3. FINAL PREPROCESSING & KERNEL GENERATION
  # ==============================================================================
  Z <- log1p(Z_counts)
  Z_trans <- scale(Z)

  # Construct kernel matrix focusing ONLY on the 3 causal effects
  # non_z <- rep(eff, 3)
  # c(non_z, rep(0, p - length(non_z)))
  # Kmat <- CBKMR:::kernel_mat_RBF_rcpp_openmp(Z_trans, c(non_z, rep(0, p - length(non_z))))
  Kmat <- CBKMR:::kernel_mat_RBF_rcpp_openmp(Z_trans, wvec)

  # ==============================================================================
  # 4. NON-LINEAR OUTCOME GENERATION (Your Latent GP Method)
  # ==============================================================================
  y_logis <- rep(0, N)
  for(ind in 1:50){
    if(sum(y_logis) <= floor(N*0.05) | sum(y_logis) > floor(N*0.95)){

      # Latent random effect model driven by the non-linear RBF kernel
      eta <- c(0 + mvnfast::rmvn(1, mu = rep(0, N), sigma = ker_var*Kmat + 1e-6 * diag(1, nrow(Kmat), ncol(Kmat))))
      #eta <- 1.5*(Z_trans[,1])*(Z_trans[,2])**2 + 2*sin(1.5*Z_trans[,3]) + (1/(Z_trans[,4])^2) + 3*sin((Z_trans[,5])**3) + exp(-sin((Z_trans[,6])**2))
      # eta <- ((Z[,1])+(exp(Z[,61])))+log(1+Z[,65]**2)
      # eta <- -log(1+Z[,1]) + log(1+Z[,3]) + (log(1+(Z[,(3*(p/5)+1)]**2)*Z[,(3*(p/5)+5)]**2))
      p0 <- plogis(eta)
      y_logis <- c(rbinom(N, size = 1, prob = p0))

    } else {
      break
    }
  }

  # Return data as a named list
  return(list(
    Z = Z_counts,
    y_logis = y_logis,
    corrmat = Sigma_gene,
    truth = wvec
  ))
}
