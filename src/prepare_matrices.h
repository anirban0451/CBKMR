#ifndef PREPARE_MATRICES_H
#define PREPARE_MATRICES_H
#include <vector>
#include <RcppArmadillo.h>
// Function to build the A and D matrices for the NNGP model
Rcpp::List build_AD_rcpp(
    const arma::mat& coords,
    const arma::imat& neighbor_matrix,
    double sigma2,
    double nugget_stabilizer = 1e-8
);
// Function to compute the log-likelihood of the NNGP model with tau as the shrinkage parameter
double rnngp_loglik_tauonly(const arma::vec& x,
                            const arma::mat& coords,
                            const arma::rowvec& w,
                            double tau,
                            const arma::imat& neighbor_matrix,
                            double nugget = 1e-8);
// Function to compute the log-likelihood of the NNGP model and return Ut
Rcpp::List rnngp_loglik_tauonly_use_AD(const arma::vec& x,
                                       const arma::mat& coords,
                                       const arma::rowvec& w,
                                       double tau,
                                       const arma::imat& neighbor_matrix,
                                       double nugget = 1e-8);
#endif // PREPARE_MATRICES_H
