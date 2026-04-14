# Phase II: Bayesian linear regression (Gibbs sampler) for DS 4420

suppressPackageStartupMessages({
  if (!requireNamespace("MASS", quietly = TRUE)) {
    stop("Install MASS: install.packages(\"MASS\")")
  }
})

#' Gibbs sampler for Bayesian normal linear regression
#'
#' Model: y = X * beta + eps, eps ~ N(0, sigma^2 I)
#' Prior: beta ~ N(0, tau^2 * I),  sigma^2 ~ InverseGamma(a0, b0)
#' (semi-conjugate; full conditionals are Normal and Inverse-Gamma.)
#'
#' @param X Design matrix (include intercept column).
#' @param y Response vector.
#' @param n_iter Total MCMC iterations.
#' @param burn Burn-in discarded from posterior summaries.
#' @param tau Prior scale for each coefficient (excluding intercept handling below).
#' @param a0,b0 Shape and rate for IG prior on sigma^2 (weakly informative).
gibbs_linear_regression <- function(X, y, n_iter = 8000L, burn = 2000L,
                                    tau = 100, a0 = 0.01, b0 = 0.01) {
  y <- as.numeric(y)
  X <- as.matrix(X)
  n <- nrow(X)
  p <- ncol(X)

  XtX <- crossprod(X)
  Xty <- crossprod(X, y)

  prior_prec <- rep(1 / tau^2, p)
  prior_prec[1] <- 1 / (tau^2 * 10)

  beta <- rep(0, p)
  sigma2 <- stats::var(y)

  samp_beta <- matrix(NA_real_, n_iter, p)
  samp_sigma2 <- numeric(n_iter)

  for (t in seq_len(n_iter)) {
    # beta | sigma^2, y ~ MVN
    prec <- XtX / sigma2 + diag(prior_prec, nrow = p, ncol = p)
    cov_beta <- solve(prec)
    mean_beta <- cov_beta %*% (Xty / sigma2)
    z <- stats::rnorm(p)
    beta <- mean_beta + as.vector(t(chol(cov_beta)) %*% z)

    # sigma^2 | beta, y ~ InverseGamma(a0 + n/2, b0 + 0.5 * RSS)
    rss <- sum((y - X %*% beta)^2)
    shape <- a0 + n / 2
    rate <- b0 + rss / 2
    sigma2 <- 1 / stats::rgamma(1, shape = shape, rate = rate)

    samp_beta[t, ] <- beta
    samp_sigma2[t] <- sigma2
  }

  keep <- seq.int(burn + 1L, n_iter)
  list(
    beta = samp_beta[keep, , drop = FALSE],
    sigma2 = samp_sigma2[keep]
  )
}

posterior_linpred_draws <- function(X_new, beta_draws) {
  X_new %*% t(beta_draws)
}

main <- function() {
  data(Boston, package = "MASS")
  X_full <- as.matrix(cbind(1, Boston[, seq_len(13)]))
  colnames(X_full) <- c("(Intercept)", names(Boston)[seq_len(13)])
  y_full <- Boston$medv

  set.seed(42)
  n <- nrow(X_full)
  n_train <- floor(0.8 * n)
  train_idx <- sample.int(n, n_train)
  test_idx <- setdiff(seq_len(n), train_idx)

  X_train <- X_full[train_idx, , drop = FALSE]
  y_train <- y_full[train_idx]
  X_test <- X_full[test_idx, , drop = FALSE]
  y_test <- y_full[test_idx]

  cat("Bayesian linear regression (Gibbs sampler)\n")
  cat("------------------------------------------\n")
  cat(sprintf("Train n = %d, Test n = %d, p = %d\n", nrow(X_train), nrow(X_test), ncol(X_train)))

  fit <- gibbs_linear_regression(X_train, y_train)

  beta_mean <- colMeans(fit$beta)
  names(beta_mean) <- colnames(X_full)

  pred_draws <- posterior_linpred_draws(X_test, fit$beta)
  y_hat <- rowMeans(pred_draws)

  rmse <- sqrt(mean((y_test - y_hat)^2))
  mae <- mean(abs(y_test - y_hat))

  cat(sprintf("Test RMSE (posterior predictive mean): %.3f\n", rmse))
  cat(sprintf("Test MAE (posterior predictive mean): %.3f\n", mae))
  cat(sprintf("Posterior mean of sigma (sqrt of variance): %.3f\n", sqrt(mean(fit$sigma2))))
  cat("\nPosterior mean of coefficients (first 6):\n")
  print(round(beta_mean[seq_len(min(6L, length(beta_mean)))], 4))

  fig_dir <- "figures"
  dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
  png(
    filename = file.path(fig_dir, "bayes_pred_vs_actual.png"),
    width = 5 * 300,
    height = 5 * 300,
    res = 300
  )
  lim <- range(c(y_test, y_hat))
  plot(
    y_test,
    y_hat,
    xlab = "Actual medv ($1000s)",
    ylab = "Predicted medv ($1000s)",
    main = sprintf("Bayesian linear regression (test RMSE = %.3f)", rmse),
    pch = 16,
    col = rgb(0.2, 0.45, 0.7, 0.75),
    asp = 1,
    xlim = lim,
    ylim = lim
  )
  abline(0, 1, lty = 2, lwd = 1)
  dev.off()
  cat(sprintf("\nSaved figure: %s\n", file.path(fig_dir, "bayes_pred_vs_actual.png")))

  invisible(list(fit = fit, rmse = rmse, beta_mean = beta_mean))
}

main()

