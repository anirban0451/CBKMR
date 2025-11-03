// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <Rcpp.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>
#include "prepare_matrices.h"
using namespace arma;
using namespace Rcpp;

//' Compute log-density of multivariate normal distribution with mean zero
//' and covariance matrix Sigma at vector x
//' This function computes the log-density of a multivariate normal distribution
//' with mean zero and covariance matrix Sigma at point x.
//' @param x A vector representing the point at which to evaluate the density
//' @param Sigma A positive definite covariance matrix
//' @return The log-density value at point x
//' @examples
//' \dontrun{
//' x <- rnorm(5)  # 5-dimensional point
//' Sigma <- diag(5)  # Identity covariance matrix
//' log_density <- logdmvn_arma(x, Sigma)
//' }
// [[Rcpp::export]]
double logdmvn_arma(const arma::vec& x, const arma::mat& Sigma) {
  // add small jitter for numerical stability
  arma::mat Sigma_jittered = Sigma + arma::eye(Sigma.n_rows, Sigma.n_cols) * 1e-8;
  arma::mat L = arma::chol(Sigma_jittered, "lower"); // Compute Cholesky decomposition
  arma::vec y = arma::solve(L, x, arma::solve_opts::fast); // Efficient solve
  double logdet = 2.0 * arma::accu(log(L.diag())); // Faster summation
  double quadform = arma::dot(y, y); // Direct dot product

  return -0.5 * (x.n_elem * log(2.0 * M_PI) + logdet + quadform);
}

//' Compute log-density of multivariate normal distribution with mean zero
//' and covariance matrix Sigma at vector x, returning also U = L^-1
//' This function computes the log-density of a multivariate normal distribution
//' with mean zero and covariance matrix Sigma at point x, and also returns
//' the matrix U = L^-1 where L is the lower triangular matrix from the Cholesky decomposition of Sigma.
//' @param x A vector representing the point at which to evaluate the density
//' @param Sigma A positive definite covariance matrix
//' @return A list containing the log-density value at point x and the matrix U
//' @examples
//' \dontrun{
//' x <- rnorm(5)  # 5-dimensional point
//' Sigma <- diag(5)  # Identity covariance matrix
//' result <- logdmvn_arma_with_U(x, Sigma)
//' log_density <- result$log_density
//' U <- result$Ut
//' }
// [[Rcpp::export]]
Rcpp::List logdmvn_arma_with_U(const arma::vec& x, const arma::mat& Sigma) {

  // Add small jitter for numerical stability
  arma::mat Sigma_jittered = Sigma + arma::eye(Sigma.n_rows, Sigma.n_cols) * 1e-8;

  // Compute Cholesky decomposition
  arma::mat L = arma::chol(Sigma_jittered, "lower");

  // Solve U = L^-1 (using trimatl to be explicit)
  arma::mat Ut = arma::solve(arma::trimatl(L), arma::eye(L.n_rows, L.n_cols), arma::solve_opts::fast);
  arma::vec y = Ut * x;

  // Get log-determinant from L
  double logdet = 2.0 * arma::accu(log(L.diag()));

  // Get quadratic form from y
  double quadform = arma::dot(y, y);

  // Calculate the final log-density value
  double log_density_value = -0.5 * (x.n_elem * log(2.0 * M_PI) + logdet + quadform);

  // --- Return the new list ---
  return Rcpp::List::create(
    Rcpp::Named("log_density") = log_density_value,
    Rcpp::Named("Ut") = Ut
  );
}

//' Compute the log-density of a copula-based Bernoulli model without covariates
//' This function computes the log-density of a copula-based Bernoulli model
//' with only an intercept term (no covariates).
//' @param y A vector of binary response variables (0s and 1s)
//' @param beta0 The intercept parameter
//' @param Ut Either a matrix or a list containing the necessary components for the copula
//' @return The log-density value of the copula-based Bernoulli model
//' @examples
//' \dontrun{
//' y <- c(0, 1, 1, 0, 1)  # Binary response vector
//' beta0 <- 0.5  # Intercept
//' Ut <- matrix(...)  # Some matrix or list for copula, e.g., from NNGP by running build_AD_rcpp
//' log_density <- lCBpdf_bernoulli_no_X(y, beta0, Ut)
//' }
// [[Rcpp::export]]
double lCBpdf_bernoulli_no_X(const arma::vec& y, double beta0, Rcpp::RObject Ut) {

  int n = y.n_elem;
  double linvdet = 0.0;
  double quadterm = 0.0;

  // --- 1. Marginal and Transformation (Common to both paths) ---
  double eta = exp(beta0) / (1.0 + exp(beta0));

  arma::vec fy_log(n);
  arma::vec uprime(n);
  arma::vec qnormuprime(n);

  for(int i = 0; i < n; ++i) {
    fy_log(i) = R::dbinom(y(i), 1, eta, true);
    double Fy = R::pbinom(y(i), 1, eta, true, false);
    double Fyminus1 = R::pbinom(y(i) - 1.0, 1, eta, true, false);
    uprime(i) = (Fyminus1 + Fy) / 2.0;
  }

  uprime = arma::clamp(uprime, 1e-15, 1.0 - 1e-15);

  for(int i = 0; i < n; ++i) {
    qnormuprime(i) = R::qnorm(uprime(i), 0.0, 1.0, true, false);
  }

  // --- 2. Copula Log-Likelihood

  if (Rcpp::is<Rcpp::List>(Ut)) {

    Rcpp::List Ut_list = Rcpp::as<Rcpp::List>(Ut);

    arma::sp_mat ImA = Rcpp::as<arma::sp_mat>(Ut_list["ImA"]);

    arma::mat D_mat = Rcpp::as<arma::mat>(Ut_list["D_inv_sqrt"]);
    arma::vec D_inv_sqrt = D_mat.col(0);

    linvdet = arma::sum(arma::log(D_inv_sqrt));

    arma::vec ImA_z = ImA * qnormuprime;
    arma::vec L_z = D_inv_sqrt % ImA_z;

    quadterm = arma::dot(L_z, L_z) - arma::dot(qnormuprime, qnormuprime);

  } else {

    arma::mat Utcpp = Rcpp::as<arma::mat>(Ut);

    linvdet = arma::accu(arma::log(Utcpp.diag()));

    arma::vec Utx = Utcpp * qnormuprime;
    quadterm = arma::dot(Utx, Utx) - arma::dot(qnormuprime, qnormuprime);
  }

  // --- 3. Combine and Return ---
  double lcpdfcopuls = linvdet - 0.5 * quadterm;
  double finalval = arma::sum(fy_log) + lcpdfcopuls;

  return finalval;
}
