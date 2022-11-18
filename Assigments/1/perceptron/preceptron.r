get_y_of_boundry <- function(x, w) {
  return(x * w[1] + w[3]) / -w[2]
}

get_norm <- function(w) {
  return(sqrt(sum(w^2)))
}



run_perceptron <- function(X, y) {
  w <- rnorm(3) # initial guess of norm vector w

  continue <- T
  while (continue) {
    cont <- 0
    for (i in 1:N) {
      if (sign(X[i, ] %*% w * y[i]) == 1) {
        cont <- cont + 1
      } else {
        w <- w + y[i] * X[i, ]
      }
      if (cont == N) {
        continue <- F
        print("Finished")
      }
    }
  }


  w
}

set.seed(2022)

N <- 20
x1 <- runif(N, 1, 2)
x2 <- runif(N, 1, 2)
X <- cbind(x1, x2, 1)
y <- ifelse(x2 > -.6 * x1 + 2.35, -1, 1)

w <- run_perceptron(X, y)

# Plotting


# Graphing settings
t <- seq(min(X[, 1]), max(X[, 1]), 0.1)
center <- c(mean(X[, 1]), mean(X[, 2]))
lwd <- 2
plot(X[, 1:2], col = y + 3)
points(t, (t * w[1] + w[3]) / -w[2], type = "l", col = "green", lwd = lwd)
