# Load packages
library("ggplot2") # visualization
library("ggthemes") # visualization
library("scales") # visualization
library("dplyr") # data manipulation
library("mice") # imputation
library("randomForest") # classification algorithm



get_acc <- function(pre_res, Y_test) {
    acc <- sum(pre_res == Y_test) / length(Y_test)
}

# Load data
train <- read.csv("titan_train.csv", stringsAsFactors = F)
test <- read.csv("titan_test.csv", stringsAsFactors = F)
train_n_rows <- nrow(train)

full <- bind_rows(train, test) # bind training & test data
# check data
str(full)

# Grab title from passenger names
full$Title <- gsub("(.*, )|(\\..*)", "", full$Name)
# Show title counts by sex
table(full$Sex, full$Title)

# Titles with very low cell counts to be combined to "rare" level
rare_title <- c(
    "Dona", "Lady", "the Countess", "Capt", "Col", "Don",
    "Dr", "Major", "Rev", "Sir", "Jonkheer"
)
# Also reassign mlle, ms, and mme accordingly
full$Title[full$Title == "Mlle"] <- "Miss"
full$Title[full$Title == "Ms"] <- "Miss"
full$Title[full$Title == "Mme"] <- "Mrs"
full$Title[full$Title %in% rare_title] <- "Rare Title"
# Show title counts by sex again
table(full$Sex, full$Title)


full$Surname <- sapply(
    full$Name,
    function(x) strsplit(x, split = "[,.]")[[1]][1]
)


full$Fsize <- full$SibSp + full$Parch + 1

# Create a family variable
full$Family <- paste(full$Surname, full$Fsize, sep = "_")


# Use ggplot2 to visualize the relationship between family size & survival
ggplot(full[1:891, ], aes(x = Fsize, fill = factor(Survived))) +
    geom_bar(stat = "count", position = "dodge") +
    scale_x_continuous(breaks = c(1:11)) +
    labs(x = "Family Size") +
    theme_few()


# Discretize family size
full$FsizeD[full$Fsize == 1] <- "singleton"
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- "small"
full$FsizeD[full$Fsize > 4] <- "large"

mosaicplot(table(full$FsizeD, full$Survived), main = "Family Size by Survival", shade = TRUE)


# This variable appears to have a lot of missing values
full$Cabin[1:28]
strsplit(full$Cabin[2], NULL)[[1]]

full$Deck <- factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))

full[c(62, 830), "Embarked"]

embark_fare <- full %>%
    filter(PassengerId != 62 & PassengerId != 830)

ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
    geom_boxplot() +
    geom_hline(aes(yintercept = 80),
        colour = "red", linetype = "dashed", lwd = 2
    ) +
    scale_y_continuous(labels = dollar_format()) +
    theme_few()


full$Embarked[c(62, 830)] <- "C"

full[1044, ]

ggplot(
    full[full$Pclass == "3" & full$Embarked == "S", ],
    aes(x = Fare)
) +
    geom_density(fill = "#99d6ff", alpha = 0.4) +
    geom_vline(aes(xintercept = median(Fare, na.rm = T)),
        colour = "red", linetype = "dashed", lwd = 1
    ) +
    scale_x_continuous(labels = dollar_format()) +
    theme_few()

full$Fare[1044] <- median(full[full$Pclass == "3" & full$Embarked == "S", ]$Fare, na.rm = TRUE)


# Show number of missing Age values
sum(is.na(full$Age))

# Make variables factors into factors
factor_vars <- c(
    "PassengerId", "Pclass", "Sex", "Embarked",
    "Title", "Surname", "Family", "FsizeD", "Survived"
)

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

set.seed(129)



# Perform mice imputation, excluding certain less-than-useful variables:
mice_mod <- mice(
    full[, !names(full) %in%
        c("PassengerId", "Name", "Ticket", "Cabin", "Family", "Surname", "Survived")],
    method = "rf"
)

mice_output <- complete(mice_mod)


par(mfrow = c(1, 2))
hist(full$Age,
    freq = F, main = "Age: Original Data",
    col = "darkgreen", ylim = c(0, 0.04)
)
hist(mice_output$Age,
    freq = F, main = "Age: MICE Output",
    col = "lightgreen", ylim = c(0, 0.04)
)

full$Age <- mice_output$Age

sum(is.na(full$Age))


ggplot(full[1:891, ], aes(Age, fill = factor(Survived))) +
    geom_histogram() +
    # I include Sex since we know (a priori) it's a significant predictor
    facet_grid(. ~ Sex) +
    theme_few()



full$Child[full$Age < 18] <- "Child"
full$Child[full$Age >= 18] <- "Adult"

table(full$Child, full$Survived)


full$Mother <- "Not Mother"
full$Mother[full$Sex == "female" & full$Parch > 0 & full$Age > 18 & full$Title !=
    "Miss"] <- "Mother"


# Show counts
table(full$Mother, full$Survived)
# Finish by factorizing our two new factor variables
full$Child <- factor(full$Child)
full$Mother <- factor(full$Mother)

# Set X_train to everything but excluded column

train <- full[1:train_n_rows, ]
test <- full[(train_n_rows + 1):nrow(full), ]

# Should we split to test and train ?
# Train a random forest
rf_mod <- randomForest(
    Survived ~ Pclass + Sex + SibSp + Age + Fare + Embarked + Title + FsizeD + Mother + Child,
    data = train,
    importance = TRUE,
)

predictions <- predict(rf_mod, train, predict.all = T)
acc_predictions <- as.integer(as.vector(predictions$aggregate))
ind_predictions <- matrix(as.integer(predictions$individual), ncol = ncol(predictions$individual))


# Err.rate of whole model
print(1-get_acc(acc_predictions, train$Survived))
# avg Err.rate of individual trees
print(mean(apply(ind_predictions, 2, function(x) 1-get_acc(x, train$Survived))))



dev.off()
plot(rf_mod)
legend("topleft", colnames(rf_mod$err.rate), col = 1:3, fill = 1:3)


importance <- importance(rf_mod)

varImportance <- data.frame(
    Variables = row.names(importance),
    Importance = round(importance[, "MeanDecreaseGini"], 2)
)

# Create a rank variable based on importance
# Add rank based on orderd
rankImportance <- varImportance %>%
    mutate(Rank = paste0("#", dense_rank(desc(Importance))))



ggplot(rankImportance, aes(x = reorder(Variables, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "#99d6ff") +
    geom_text(aes(label = Rank), vjust = -0.25, size = 3) +
    coord_flip() +
    labs(x = "Variable", y = "Importance") +
    theme_few()
