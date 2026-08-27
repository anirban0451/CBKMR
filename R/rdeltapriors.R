### Some helper functions for delta priors

# Prior densities
rprior.logdens.unif <- function(x, r.params) {
  r.a <- r.params$r.a
  r.b <- r.params$r.b
  dunif(x, r.a, r.b, log=TRUE)
}

rprior.logdens.dgamma <- function(x, r.params) {
  mu.r <- r.params$mu.r
  sigma.r <- r.params$sigma.r
  dgamma(x, shape=mu.r^2/sigma.r^2, rate=mu.r/sigma.r^2, log=TRUE)
}

rprior.logdens.dhcauchy <- function(x, r.params) {
  sigma.r <- r.params$sigma.r
  dhcauchy(x, sigma=sigma.r, log=TRUE)
}

rprior.logdens.invunif <- function(x, r.params) {
  r.a <- r.params$r.a
  r.b <- r.params$r.b
  ifelse(1/r.b <= x & x <= 1/r.a, -2*log(x) - log(r.b - r.a), log(0))
}

# Proposal generators and densities - move 1
rprop.gen1.unif <- function(r.params) {
  r.a <- r.params$r.a
  r.b <- r.params$r.b
  runif(1, r.a, r.b)
}
rprop.logdens1.unif <- function(x, r.params) {
  r.a <- r.params$r.a
  r.b <- r.params$r.b
  dunif(x, r.a, r.b, log=TRUE)
}

# Proposal generators and densities - move 2
rprop.gen2.truncnorm <- function(current, r.params) {
  r.a <- r.params$r.a
  r.b <- r.params$r.b
  r.jump <- r.params$r.jump2
  truncnorm::rtruncnorm(1, a = r.a, b = r.b, mean = current, sd = r.jump)
}
rprop.logdens2.truncnorm <- function(prop, current, r.params) {
  r.a <- r.params$r.a
  r.b <- r.params$r.b
  r.jump <- r.params$r.jump2
  log(truncnorm::dtruncnorm(prop, a = r.a, b = r.b, mean = current, sd = r.jump))
}
