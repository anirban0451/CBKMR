#include <RcppArmadillo.h>
#include <vector>
#include <algorithm>

// We need R's C-level math functions
#include <R_ext/Rdynload.h>
#include <Rmath.h>


// Conditionally include the OpenMP header
#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]

inline double gaussian_kernel(double d, double sigma2) {
  return sigma2 * std::exp(- (d * d));
}

//' Build the A and D matrices for the NNGP model such that
//' Cov = (I - A)^T D (I - A)
//' @param coords An n x p matrix of coordinates
//' @param neighbor_matrix An n x
//' m integer matrix of neighbor indices (1-based, with NA for missing neighbors)
//' @param sigma2 The variance parameter for the Gaussian kernel
//' @param nugget_stabilizer A small value added to the diagonal for numerical stability
//' @return A list containing the sparse matrix triplet (A_i, A_j, A_x) for the sparse matrix A
//'  and the vector D
//'  @examples
//' \dontrun{
//' coords <- matrix(runif(100), nrow=20)  # 20 points in 5D
//' neighbor_matrix <- matrix(sample(c(1:20, NA), 60, replace=TRUE), nrow=20)  # 3 neighbors per point
//' sigma2 <- 1.0
//' result <- build_AD_rcpp(coords, neighbor_matrix, sigma2)
//' A_i <- result$A_i
//' A_j <- result$A_j
//' A_x <- result$A_x
//' D <- result$D
//' }
// [[Rcpp::export]]
Rcpp::List build_AD_rcpp(
    const arma::mat& coords,
    const arma::imat& neighbor_matrix,
    double sigma2,
    double nugget_stabilizer = 1e-8
) {
  int n = coords.n_rows;
  int m = neighbor_matrix.n_cols;

  // Initialize storage for results
  std::vector<int> A_i_vec;
  std::vector<int> A_j_vec;
  std::vector<double> A_x_vec;
  arma::vec D(n);

  // Main loop over each point, parallelized with OpenMP
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < n; ++i) {

    // --- 1. Extract valid neighbor indices for point i ---
    std::vector<int> current_neighbors;
    // Note: Rcpp::IntegerMatrix::Row is not thread-safe if modified, but read-only is fine.
    // Rcpp::IntegerVector row = neighbor_matrix.row(i);
    for (int j = 0; j < m; ++j) {
      int val = neighbor_matrix(i, j);
      if (val != INT_MIN) {
        current_neighbors.push_back(val - 1); // 1-based to 0-based
      }
    }

    int k = current_neighbors.size();

    if (k == 0) {
      D(i) = gaussian_kernel(0.0, sigma2) + nugget_stabilizer; // No neighbors, variance is just the kernel at 0 plus nugget
      // D(i) += nugget_stabilizer; // Add nugget for numerical stability
      // In a parallel loop, 'continue' correctly proceeds to the thread's next iteration
      continue;
    }

    arma::uvec Ni = arma::conv_to<arma::uvec>::from(current_neighbors);

    // --- 2. Perform calculations ---
    // All variables declared here are private to each thread
    arma::mat neighbor_coords = coords.rows(Ni);
    arma::rowvec current_coord = coords.row(i);

    arma::vec C_iN(k);
    arma::mat C_NN(k, k);

    for (int j = 0; j < k; ++j) {
      double d = arma::norm(current_coord - neighbor_coords.row(j));
      C_iN(j) = gaussian_kernel(d, sigma2);
    }

    for (int r = 0; r < k; ++r) {
      for (int s = r; s < k; ++s) {
        double d = arma::norm(neighbor_coords.row(r) - neighbor_coords.row(s));
        double cov_val = gaussian_kernel(d, sigma2);
        C_NN(r, s) = cov_val;
        C_NN(s, r) = cov_val;
      }
    }
    C_NN.diag() += nugget_stabilizer;

    arma::vec a = arma::solve(C_NN, C_iN, arma::solve_opts::fast + arma::solve_opts::likely_sympd);

    // Writing to D(i) is thread-safe because each thread has a unique 'i'
    D(i) = gaussian_kernel(0.0, sigma2) + nugget_stabilizer - arma::dot(a, C_iN);

    // --- 3. Store results for the sparse matrix triplet ---
    // This part writes to shared std::vectors and is NOT thread-safe by default.
    // It must be protected by a "critical" section, ensuring only one thread
    // can add elements to the vectors at a time.
#ifdef _OPENMP
#pragma omp critical
#endif
{
  for (int j = 0; j < k; ++j) {
    A_i_vec.push_back(i + 1);
    A_j_vec.push_back(Ni(j) + 1);
    A_x_vec.push_back(a(j));
  }
}
  }

  // Return results to R
  return Rcpp::List::create(
    Rcpp::_["A_i"] = Rcpp::wrap(A_i_vec),
    Rcpp::_["A_j"] = Rcpp::wrap(A_j_vec),
    Rcpp::_["A_x"] = Rcpp::wrap(A_x_vec),
    Rcpp::_["D"] = D,
    Rcpp::_["n"] = n
  );
}

//' Compute the log-likelihood of the NNGP model with with tau as the shrinkage parameter
//' @param x An n-dimensional vector of observations
//' @param coords An n x p matrix of coordinates
//' @param w A p-dimensional row vector of weights for each dimension
//' @param tau The shrinkage parameter (between 0 and 1)
//' @param neighbor_matrix An n x m integer matrix of neighbor indices (1-based, with NA for missing neighbors)
//' @param nugget A small value added to the diagonal for numerical stability
//' @return The log-likelihood value
//' @examples
//' \dontrun{
//' x <- rnorm(20)  # 20 observations
//' coords <- matrix(runif(100), nrow=20)  # 20 points in 5D
//' w <- runif(5)  # weights for each dimension
//' tau <- 0.5
//' neighbor_matrix <- matrix(sample(c(1:20, NA), 60, replace=TRUE), nrow=20)  # 3 neighbors per point
//' loglik <- rnngp_loglik_tauonly(x, coords, w, tau, neighbor_matrix)
//' }
// [[Rcpp::export]]
double rnngp_loglik_tauonly(const arma::vec& x,
                            const arma::mat& coords,
                            const arma::rowvec& w,
                            double tau,
                            const arma::imat& neighbor_matrix,
                            double nugget = 1e-8) {
  int n = coords.n_rows;
  // calculate conv_coords = t(sqrt(w)*t(coords))
  arma::mat conv_coords = coords.each_row() % arma::sqrt(w);

  // Call the builder function. Note that tau2 is now an explicit parameter.
  // The builder function should be updated to accept tau2 and add it to the diagonal.
  // Assuming build_AD_rcpp is modified to handle tau2 correctly:
  Rcpp::List AD = build_AD_rcpp(conv_coords, neighbor_matrix, tau, 1.0-tau+nugget);

  // --- FIX: Convert 1-based indices from R to 0-based for Armadillo ---
  arma::uvec A_i = Rcpp::as<arma::uvec>(AD["A_i"]) - 1;
  arma::uvec A_j = Rcpp::as<arma::uvec>(AD["A_j"]) - 1;
  arma::vec A_x = Rcpp::as<arma::vec>(AD["A_x"]);
  arma::vec D = Rcpp::as<arma::vec>(AD["D"]);

  // 1. Create a 2-column matrix of (row, col) locations.
  //    The size will be (number_of_non_zero_elements x 2).
  arma::umat locations = arma::join_rows(A_i, A_j);

  // 2. Construct the sparse matrix A using the transpose of the locations matrix
  //    and the vector of values. The constructor expects a (2 x n_elem) locations matrix.
  arma::sp_mat A(locations.t(), A_x, n, n);

  // --- EFFICIENCY: Calculate (I - A) * x directly ---
  arma::vec Lx = x - (A * x);

  // The rest of your logic is correct
  arma::vec D_inv_sqrt = 1.0 / arma::sqrt(D);
  arma::vec DhalfLx = D_inv_sqrt % Lx;

  double quad_form = arma::dot(DhalfLx, DhalfLx);
  double log_det = arma::sum(arma::log(D));

  const double log2pi = std::log(2.0 * arma::datum::pi);
  double loglik = -0.5 * (n * log2pi + log_det + quad_form);

  return loglik;
}

//' Same as rnngp_loglik_tauonly but also return Ut as a list such that U.ImA = I-A
//' and U.D_inv_sqrt = D^{-1/2}
// [[Rcpp::export]]
Rcpp::List rnngp_loglik_tauonly_use_AD(const arma::vec& x,
                                       const arma::mat& coords,
                                       const arma::rowvec& w,
                                       double tau,
                                       const arma::imat& neighbor_matrix,
                                       double nugget = 1e-8) {
  int n = coords.n_rows;

  // Step 1: Transform coordinates
  arma::mat conv_coords = coords.each_row() % arma::sqrt(w);

  // Step 2: Build A and D using the builder function
  Rcpp::List AD = build_AD_rcpp(conv_coords, neighbor_matrix, tau, 1.0 - tau + nugget);

  // Step 3: Convert 1-based indices from R to 0-based for Armadillo
  arma::uvec A_i = Rcpp::as<arma::uvec>(AD["A_i"]) - 1;
  arma::uvec A_j = Rcpp::as<arma::uvec>(AD["A_j"]) - 1;
  arma::vec A_x = Rcpp::as<arma::vec>(AD["A_x"]);
  arma::vec D = Rcpp::as<arma::vec>(AD["D"]);

  // Step 4: Construct sparse matrix A
  arma::umat locations = arma::join_rows(A_i, A_j);
  arma::sp_mat A(locations.t(), A_x, n, n);

  // Step 5: Compute I - A
  arma::sp_mat I_minus_A = arma::speye<arma::sp_mat>(n, n) - A;

  // Step 6: Compute (I-A) * x and scale by D^{-1/2}
  arma::vec Lx = I_minus_A * x;
  arma::vec D_inv_sqrt = 1.0 / arma::sqrt(D);
  arma::vec Utx = D_inv_sqrt % Lx;

  // Step 8: Compute log-likelihood components
  double quad_form = arma::dot(Utx, Utx);
  double log_det = arma::sum(arma::log(D));
  const double log2pi = std::log(2.0 * arma::datum::pi);
  double loglik = -0.5 * (n * log2pi + log_det + quad_form);

  // Step 9: Return results
  return Rcpp::List::create(
    Rcpp::_["log_density"] = loglik,
    Rcpp::_["Ut"] = Rcpp::List::create( // <-- Create the nested list
      Rcpp::_["ImA"] = Rcpp::wrap(I_minus_A),
      Rcpp::_["D_inv_sqrt"] = Rcpp::wrap(D_inv_sqrt)
    )
  );
}
