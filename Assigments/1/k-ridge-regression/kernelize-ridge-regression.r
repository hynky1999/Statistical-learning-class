set.seed(2020)

library(kernlab)
library(MASS)
data <- Boston


normalize_data <- function(data, mean, sd) {
    centralized <- data - t(replicate(nrow(data), mean))
    stand <- centralized / t(replicate(nrow(data), sd))
    stand
}


mse <- function(predicted, actual) {
    sqrt(sum((predicted - actual)^2) / length(actual))
}

# 4.1

predict_ridge <- function(X_train, X_test, Y_train, lambda, sigma) {
    X_train <- as.matrix(X_train)
    X_test <- as.matrix(X_test)
    Y_train <- as.matrix(Y_train)

    mean <- colMeans(X_train)
    std <- apply(X_train, 2, sd)

    X_train <- normalize_data(X_train, mean, std)
    X_test <- normalize_data(X_test, mean, std)

    # get the gram matrix
    kernel_fc <- rbfdot(sigma = sigma)
    K <- kernelMatrix(kernel_fc, X_train)

    a <- solve(K + lambda * diag(nrow(K))) %*% Y_train

    predicted <- t(a) %*% kernelMatrix(kernel_fc, X_train, X_test)

    ######################################################
    # kernel(x_train_1, x_test_1) kernel(x_train_1, x_test_2)
    # kernel(x_trian_2, x_test_1) kenrel(x_train_2, x_test_2)
    # ker

    t(predicted)
}



sigma <- 0.1
lambda <- 0.1
train_ind <- sample(1:nrow(data), 400)
X_train <- data[train_ind, -14]
Y_train <- data[train_ind, 14]
X_test <- data[-train_ind, -14]
Y_test <- data[-train_ind, 14]



# 4.2

# Regression
model <- lm(Y_train ~ ., data = X_train)
summary(model)

predicted <- predict(model, X_test)
baseline_mse <- mse(predicted, Y_test)

# 4.3
log_seq <- function(start, end, step) {
    10^(seq(start, end, step))
}


possible_vals <- log_seq(-6, -1, 0.5)

# Generate lambdas and sigmas
lambda_sigmas <- expand.grid(lambda = possible_vals, sigma = possible_vals)




k <- 10
best_model <- NULL
best_error <- Inf
shuffled <- sample(length(X_train[, 1]), length(X_train[, 1]))
folds <- split(shuffled, cut(shuffled, breaks = k, labels = FALSE))
for (i in 1:nrow(lambda_sigmas)) {
    print(i)
    total_error <- 0
    lambda <- lambda_sigmas[i, 1]
    sigma <- lambda_sigmas[i, 2]
    for (fold in folds) {
        test_x <- X_train[fold, ]
        train_x <- X_train[-fold, ]
        test_y <- Y_train[fold]
        train_y <- Y_train[-fold]
        predicted <- predict_ridge(train_x, test_x, train_y, lambda, sigma)


        error <- mse(predicted, test_y)
        total_error <- total_error + error
    }

    if (total_error < best_error) {
        best_error <- total_error
        best_model <- lambda_sigmas[i, ]
    }
}




y <- predict_ridge(X_train, X_test, Y_train, best_model[["lambda"]], best_model[["sigma"]])

mse(y, Y_test)
