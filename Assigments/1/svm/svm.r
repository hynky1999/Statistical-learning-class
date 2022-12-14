set.seed(8312)
load("Assigments/1/ridge-regression/HWD.RData")


# split data into training and test sets
train_ind <- sample(1:nrow(data), 0.8 * nrow(data))

get_acc <- function(pre_res, Y_test) {
    acc <- sum(pre_res == Y_test) / length(Y_test)
    acc
}
data[, 1] <- as.factor(data[, 1])
# training set
X_train <- data[train_ind, -1]
Y_train <- as.factor(data[train_ind, 1])

X_test <- data[-train_ind, -1]
Y_test <- as.factor(data[-train_ind, 1])


train_svm <- function(X_train, Y_train, C, kernel = "vanilladot") {
    model <- ksvm(X_train, Y_train, kernel = kernel, C = C, cross = 10, type = "C-svc")
    model
}

library(kernlab)
# Train svm


train_svm_with_cross <- function(X_train, Y_train, Cs_vals, kernel = "vanilladot") {
    # Generate lambdas and sigmas
    # No cross validation here as it is done in train_svm
    Cs <- expand.grid(C = Cs_vals)
    best_model <- NULL
    best_error <- Inf
    kpars <- list()
    for (i in 1:nrow(Cs)) {
        print(i)
        C <- Cs[i, 1]
        model <- train_svm(X_train, Y_train, C, kernel)
        error <- cross(model)
        total_error <- error

        if (total_error < best_error) {
            best_error <- total_error
            kpars <- kpar(kernelf(model))
            best_model <- Cs[i, ]
        }
    }
    list(C = best_model, kpars = kpars)
}


possible_vals <- c(0.0001, 0.0005, 0.001, 0.005, 0.1, 1)

best_vanilla_dot_param <- train_svm_with_cross(X_train, Y_train, possible_vals, kernel = "vanilladot")

best_rbdot_param <- train_svm_with_cross(X_train, Y_train, possible_vals, kernel = "rbfdot")



vanilla_dot_model <- ksvm(X_train, Y_train, C = best_vanilla_dot_param$C, kernel = "vanilladot", type = "C-svc")


rbfdot_model <- ksvm(X_train, Y_train, kernel = kernel, C = C, type = "C-svc", kpar = best_rbdot_param$kpars)

predicted_vanilla_test <- predict(vanilla_dot_model, X_test)
predicted_rbf_test <- predict(rbfdot_model, X_test)

print(get_acc(predicted_vanilla_test, Y_test))
print(get_acc(predicted_rbf_test, Y_test))

predicted_vanilla_test <- predict(vanilla_dot_model, X_train)
predicted_rbf_test <- predict(rbfdot_model, X_train)

print(get_acc(predicted_vanilla_test, Y_train))
print(get_acc(predicted_rbf_test, Y_train))

print(best_vanilla_dot_param)
print(best_rbdot_param)
