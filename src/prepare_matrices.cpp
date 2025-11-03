// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#ifdef _OPENMP
  #include <omp.h>
#endif
#include "prepare_matrices.h"
using namespace arma;

//' Compute RBF kernel matrix with corresponding weights using OpenMP
//'
//' This function computes the RBF kernel matrix for a given data matrix Z
//' and a vector of weights w, utilizing OpenMP for parallelization.
//' @param Z An N x p matrix where each row is a data point.
//' @param w A vector of length p containing weights for each dimension.
//' @return An N x N RBF kernel matrix.
//' @examples
//' \dontrun{
//' Z <- matrix(rnorm(100), nrow=10)  # 10 data points in 10D
//' w <- runif(10)                    # weights for each dimension
//' K <- kernel_mat_RBF_rcpp_openmp(Z, w)
//' }
// [[Rcpp::export]]
arma::mat kernel_mat_RBF_rcpp_openmp(const arma::mat& Z, const arma::vec& w) {
  int N = Z.n_rows;
  int p = Z.n_cols;

  arma::mat K(N, N, arma::fill::zeros);

  // Pre-scale columns of Z by sqrt of weights
  arma::mat Zw = Z;
  for(int k = 0; k < p; ++k) {
    Zw.col(k) *= std::sqrt(w(k));
  }

  // Parallelize outer loop using OpenMP
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N; ++i) {
    for (int j = i; j < N; ++j) {
      double dist_sq = 0.0;
      for (int k = 0; k < p; ++k) {
        double diff = Zw(i,k) - Zw(j,k);
        dist_sq += diff * diff;
      }
      double val = std::exp(-dist_sq);
      K(i,j) = val;
      if (i != j) {
        K(j,i) = val;
      }
    }
  }
  return K;
}
