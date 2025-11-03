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
