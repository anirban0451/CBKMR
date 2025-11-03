# Some helper functions for delta priors
rprior.logdens <- function(x, r.params) {
  r.a <- r.params$r.a
  r.b <- r.params$r.b
  dunif(x, r.a, r.b, log=TRUE)
}
rprop.gen1 <- function(r.params) {
  r.a <- r.params$r.a
  r.b <- r.params$r.b
  runif(1, r.a, r.b)
}
rprop.logdens1 <- function(x, r.params) {
  r.a <- r.params$r.a
  r.b <- r.params$r.b
  dunif(x, r.a, r.b, log=TRUE)
}
rprop.gen2 <- function(current, r.params) {
  r.a <- r.params$r.a
  r.b <- r.params$r.b
  r.jump <- r.params$r.jump2
  truncnorm::rtruncnorm(1, a = r.a, b = r.b, mean = current, sd = r.jump)
}
rprop.logdens2 <- function(prop, current, r.params) {
  r.a <- r.params$r.a
  r.b <- r.params$r.b
  r.jump <- r.params$r.jump2
  log(truncnorm::dtruncnorm(prop, a = r.a, b = r.b, mean = current, sd = r.jump))
}
rprop.gen <- function(current, r.params) {
  r.a <- r.params$r.a
  r.b <- r.params$r.b
  r.jump <- r.params$r.jump
  truncnorm::rtruncnorm(1, a = r.a, b = r.b, mean = current, sd = r.jump)
}
rprop.logdens <- function(prop, current, r.params) {
  r.a <- r.params$r.a
  r.b <- r.params$r.b
  r.jump <- r.params$r.jump
  log(truncnorm::dtruncnorm(prop, a = r.a, b = r.b, mean = current, sd = r.jump))
}
