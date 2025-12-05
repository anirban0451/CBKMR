# CBKMR

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Overview

`logisticCBKMR` implements Logistic Composite Binary Kernel Machine Regression. This package allows for variable selection and effect estimation in binary outcome data using kernel machine regression.

## Installation

You can install the development version from GitHub with:

```r
# install.packages("devtools")
devtools::install_github("anirban0451/CBKMR")
```

## Example Usage

The following tutorial demonstrates how to simulate data and run the `logisticCBKMR` model to select features.

### 1. Data Generation

We simulate `N` samples with `p` features. We enforce independence between features using a Gram-Schmidt-like orthogonalization process and scale the data.

```r
library(mvnfast)
library(CBKMR) 

# --- Parameters ---
set.seed(20240)  # Fixed seed for reproducibility
N <- 200        # Number of samples
p <- 10         # Number of features
eff <- 1        # Effect size magnitude
ker_var <- 25   # Kernel variance

# --- Generate Features (Z) ---
Sigma_Z <- diag(1, N) 
Z <- list()
dframe <- NULL

# Generate independent features
for(i in 1:p){
  Z[[i]] <- c(mvnfast::rmvn(n = 1, mu = rep(0, N), sigma = Sigma_Z))
  
  # Orthogonalize to ensure independence
  if(i > 1){
    dframe <- cbind(dframe, Z[[i - 1]])
    Z[[i]] <- lm(Z[[i]] ~ dframe - 1)$residuals 
  }
}
Z <- do.call(cbind, Z) # Final feature matrix (N x p)

# Scale features (Critical for Variable Selection)
Z_trans <- scale(Z)
```

### 2. Simulate Binary Outcome

We construct a true kernel matrix assuming that only the first **3 features** have non-zero effects. We then simulate a binary outcome `y` ensuring the data is not too sparse.

```r
# --- Construct Kernel ---
# Define effect sizes (First 3 features are active, rest are 0)
non_z <- rep(eff, 3)
true_effects <- c(non_z, rep(0, p - 3))

# Generate True Kernel Matrix
Kmat <- CBKMR:::kernel_mat_RBF_rcpp_openmp(Z_trans, true_effects)

# --- Simulate y (Latent Model) ---
y_logis <- rep(0, N)

# Loop to ensure reasonable class balance (5% - 95%)
for(ind in 1:50){
  
  # Latent random effect model
  eta <- mvnfast::rmvn(1, mu = rep(0, N), sigma = ker_var * Kmat + diag(1e-8, N))
  p0 <- exp(eta)/(1 + exp(eta))
  y_logis <- c(rbinom(N, size = 1, prob = p0))
  
  # Check balance
  if(sum(y_logis) > floor(N * 0.05) & sum(y_logis) <= floor(N * 0.95)){
    break
  }
}
```

### 3. Run Analysis

Finally, we run the main function `logisticCBKMR` to estimate effects.

```r
# Run the model
res <- logisticCBKMR(
  y = y_logis, 
  Z = Z_trans, 
  nsim = 4000 * p,  # Total MCMC iterations
  verbose = TRUE,
  beta0_scheme = 1
)

# View results
# str(res)
# print(res$selected_features)
```
