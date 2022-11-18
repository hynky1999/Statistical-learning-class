rm(list = ls())
#--- generate the data ---#
DGP_ellipse <- function(N = 50, seed = 8312) {
    set.seed(seed)
    oval_fun <- function(x, a = 1, b = 0.5) {
        b * sqrt(1 - (x / a)^2)
    }
    x11 <- runif(N, -1, 1)
    x12 <- c(oval_fun(x11[1:(.5 * N)]), -oval_fun(x11[(.5 * N + 1):N])) + rnorm(N, 0, 0.05)
    X <- cbind(x11, x12)
    x21 <- runif(N, -1.5, 1.5)
    x22 <- c(oval_fun(x21[1:(.5 * N)], a = 1.5, b = 0.75), -oval_fun(x21[(.5 * N + 1):N], a = 1.5, b = 0.75)) + rnorm(N, 0, 0.05)
    X <- rbind(X, cbind(x21, x22))
    Q <- eigen(matrix(c(1, -4, -4, 1), 2, 2))$vectors
    X <- X %*% Q
    y <- c(rep(1, N), rep(0, N))
    d <- cbind(y, X)
    return(d)
}
N <- 10
d <- DGP_ellipse(N)
y <- d[, 1]
X <- d[, -1]

# visualize
plot(X, pch = 20, col = y + 2, xlab = "X1", ylab = "X2", asp = 1, cex = 3)
#--- generate the data OVER ---#

print(nrow(X))
#--- tr_te_split ---#
id <- sample(1:(2 * N), N * 0.2)
X_tr <- X[-id, ]
X_te <- X[id, ]
y_tr <- y[-id]
y_te <- y[id]
#--- tr_te_split OVER ---#


get_gram_matrix <- function(X) {
    # Way faster than cell by cell
    (X %*% t(X))^2
}

get_centralized_gram_matrix <- function(gram_matrix) {
    N <- nrow(gram_matrix)
    C <- 1 / N * rep(1, N) %*% t(rep(1, N))
    gram_matrix - C %*% gram_matrix - gram_matrix %*% C - C %*% gram_matrix %*% C
}

learn_kpac <- function(X) {
    gram_matrix <- get_gram_matrix(X)
    centralized_gram_matrix <- get_centralized_gram_matrix(gram_matrix)

    eigen(centralized_gram_matrix)
}


transform_with_component <- function(kcpa_mat, train_X, X, component) {
    lambda <- kcpa_mat$values[component]
    u <- matrix(kcpa_mat$vectors[, component])
    N <- nrow(train_X)
    C <- 1 / N * rep(1, N) %*% t(rep(1, N))
    gramm_matrix <- get_gram_matrix(train_X)



    # iterate over over the rows of X
    # Can be vectorized but I am lazy and it would be more mystical
    result <- rep(1, nrow(X))

    for (i in 1:nrow(X)) {
        # get the row
        row <- X[i, ]
        # already transposed
        dot_product <- train_X %*% row
        # (1,n) (n,n) (n,1) =
        centralisation <- 1 / N * gramm_matrix %*% rep(1, N)
        y <- 1 / lambda * t(u) %*% (diag(N) - C) %*% (dot_product^2 - centralisation)

        result[i] <- y[1, 1]
    }

    result
}



pca_eigs <- learn_kpac(X_tr)
# Plot third component

third_comp <- pca_eigs$vectors[, 3]

# Sanity check
stopifnot(all.equal(transform_with_component(pca_eigs, X_tr, X_tr, 3), third_comp))

# Get test results
test_results <- transform_with_component(pca_eigs, X_tr, X_te, 3)








# Plot without test set
plot(third_comp, col = y_tr + 2, cex = 2, pch = 20)

# Plot with test set
plot(c(third_comp, test_results), col = c(y_tr, y_te) + 2, cex = 2, pch = 20)
