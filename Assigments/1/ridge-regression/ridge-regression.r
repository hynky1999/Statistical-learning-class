set.seed(8312)


get_acc <- function(pre_res, Y_test) {
    acc <- sum(predicted_test == Y_test) / length(Y_test)
}

# kappa
get_kappa <- function(pre_res, Y_test) {
    expected_acc <- 0
    unique_label_len <- length(unique(Y_test))
    for (i in 1:unique_label_len) {
        expected_acc <- expected_acc + sum(pre_res == i) / length(Y_test) * sum(Y_test == i) / length(Y_test)
    }
    acc <- get_acc(pre_res, Y_test)
    (acc - expected_acc) / (1 - expected_acc)
}


plot_cfs <- function(cfs, colors, bw = FALSE) {
    par(mfrow = c(2, 5))
    print(length(cfs))
    for (i in 1:length(cfs)) {
        weights <- as.numeric(cfs[[i]])[-1]
        z <- matrix(weights, 16, 16, byrow = T)
        rotate_z <- t(z[, 16:1])
        if (bw == TRUE) {
            rotate_z <- rotate_z != 0
        }
        image(rotate_z, col = colors(256))
    }
}

# load HWD.RData
load("Assigments/1/ridge-regression/HWD.RData")


# split data into training and test sets
train_ind <- sample(1:nrow(data), 0.8 * nrow(data))


# training set
X_train <- data[train_ind, -1]
Y_train <- data[train_ind, 1]

X_test <- data[-train_ind, -1]
Y_test <- data[-train_ind, 1]


library(glmnet)
# cross validate wtih cv.glmnet
gamma <- cv.glmnet(X_train, Y_train, family = "multinomial", nfolds = 10, alpha = 1)

# plot cv.glmnet
plot(gamma)


predicted_test <- predict(gamma, as.matrix(X_test), s = "lambda.min", type = "class")




# Acc calculation
print(get_acc(predicted_test, Y_test))


# Kappa calculation
print(get_kappa(predicted_test, Y_test))
# the way you visualize the images of digit

colors <- c("white", "black")
cus_col <- colorRampPalette(colors = colors)

plot_cfs(coef(gamma, s = "lambda.min"), bw = TRUE, cus_col)
