# Loading some required libraries
library(tidyverse) # for it's awesomeness
library(caret) # for training our models
library(corrplot) # for visualising correlations
library(gam) # for building a Loess model using the caret package
library(rpart) # for building a classification tree model using the caret package
library(randomForest) # for building a random forest model using the caret package
library(xgboost) # for building an extreme gradient boosting model using the caret package
library(pROC) # for calculating AUC values

# Importing the diabetes dataset
setwd("C:/Users/cyc_c/Desktop/Data Science/projects/diabetes_risk_prediction")
diabetes <- read.csv("diabetes.csv")

# Let's conduct some preliminary exploration of the data
head(diabetes)
dim(diabetes)
str(diabetes)
summary(diabetes)

# Checking for missing values
any(is.na(diabetes))

# Checking for duplicates in the data
diabetes[duplicated(diabetes),]

# Let's look at the overall prevelance of diabetes in this population
mean(diabetes$Outcome)

# We are going to plot all the variables for a quick visual inspection
# First, let's convert Outcome variable to factor with labels for plotting
diabetes_plot <- diabetes %>%
  mutate(Outcome = factor(Outcome, labels = c("Non Diabetic", "Diabetic")))

# Plot of number of pregnancies by diabetes status
diabetes_plot %>%
  ggplot(aes(x = Pregnancies, fill = factor(Outcome))) +
  geom_histogram(position = "identity", bins = 10, colour = "black", fill = "mediumpurple") +
  facet_wrap(~Outcome) +
  labs(title = "Distribution of Pregnancies by Diabetes Status",
       x = "Pregnancies",
       y = "Count",
       fill = "Diabetes Status")

# Plot of blood glucose by diabetes status
diabetes_plot %>%
  ggplot(aes(x = Glucose, fill = factor(Outcome))) +
  geom_histogram(position = "identity", bins = 10, colour = "black", fill = "mediumpurple") +
  facet_wrap(~Outcome) +
  labs(title = "Distribution of Blood Glucose by Diabetes Status",
       x = "Glucose",
       y = "Count",
       fill = "Diabetes Status")
# Note to remove 0 readings

# Plot of blood pressure by diabetes status
diabetes_plot %>%
  ggplot(aes(x = BloodPressure, fill = factor(Outcome))) +
  geom_histogram(position = "identity", bins = 10, colour = "black", fill = "mediumpurple") +
  facet_wrap(~Outcome) +
  labs(title = "Distribution of Blood Pressure by Diabetes Status",
       x = "Blood Pressure",
       y = "Count",
       fill = "Diabetes Status")
# Note to remove 0 readings

# Plot of skin thickness by diabetes status
diabetes_plot %>%
  ggplot(aes(x = SkinThickness, fill = factor(Outcome))) +
  geom_histogram(position = "identity", bins = 10, colour = "black", fill = "mediumpurple") +
  facet_wrap(~Outcome) +
  labs(title = "Distribution of Skin Thickness by Diabetes Status",
       x = "Skin Thickness",
       y = "Count",
       fill = "Diabetes Status")
# Note to remove 0 readings

# Plot of insulin levels by diabetes status
diabetes_plot %>%
  ggplot(aes(x = Insulin, fill = factor(Outcome))) +
  geom_histogram(position = "identity", bins = 10, colour = "black", fill = "mediumpurple") +
  facet_wrap(~Outcome) +
  labs(title = "Distribution of Insulin Levels by Diabetes Status",
       x = "Insulin Levels",
       y = "Count",
       fill = "Diabetes Status")
# Note to remove 0 readings

# Plot of BMI by diabetes status
diabetes_plot %>%
  ggplot(aes(x = BMI, fill = factor(Outcome))) +
  geom_histogram(position = "identity", bins = 10, colour = "black", fill = "mediumpurple") +
  facet_wrap(~Outcome) +
  labs(title = "Distribution of BMI by Diabetes Status",
       x = "BMI",
       y = "Count",
       fill = "Diabetes Status")
# Note to remove 0 readings

# Plot of Diabetes Pedigree Function (DPF) by diabetes status
diabetes_plot %>%
  ggplot(aes(x = DiabetesPedigreeFunction, fill = factor(Outcome))) +
  geom_histogram(position = "identity", bins = 10, colour = "black", fill = "mediumpurple") +
  facet_wrap(~Outcome) +
  labs(title = "Distribution of DPF by Diabetes Status",
       x = "DPF",
       y = "Count",
       fill = "Diabetes Status")

# Plot of age by diabetes status
diabetes_plot %>%
  ggplot(aes(x = Age, fill = factor(Outcome))) +
  geom_histogram(position = "identity", bins = 10, colour = "black", fill = "mediumpurple") +
  facet_wrap(~Outcome) +
  labs(title = "Distribution of Age by Diabetes Status",
       x = "Age",
       y = "Count",
       fill = "Diabetes Status")

# We need to deal with the missing values.
# Let's try imputing missing values with median
diabetes$BloodPressure[diabetes$BloodPressure == 0] <- median(diabetes$BloodPressure[diabetes$BloodPressure != 0])
diabetes$Insulin[diabetes$Insulin == 0] <- median(diabetes$Insulin[diabetes$Insulin != 0])
diabetes$Glucose[diabetes$Glucose == 0] <- median(diabetes$Glucose[diabetes$Glucose != 0])
diabetes$BMI[diabetes$BMI == 0] <- median(diabetes$BMI[diabetes$BMI != 0])
diabetes$SkinThickness[diabetes$SkinThickness == 0] <- median(diabetes$SkinThickness[diabetes$SkinThickness != 0])

# Checking for correlations between predictors
diabetes_correlation <- diabetes[-9] # Remove Outcome
diabetes_correlation <- cor(diabetes_correlation)
corrplot(diabetes_correlation, method = "color", type = "lower",
         addCoef.col = "black", number.cex = 0.5, tl.cex = 0.7, tl.col = "black")

# It is time to build our model. Let's create training and test sets.
# But first, let's code the outcome as a factor before splitting the data to allow us to build our model
diabetes$Outcome <- factor(diabetes$Outcome)

# Final hold-out test set will be 20% of diabetes data
set.seed(123, sample.kind="Rounding") # if using R 3.6 or later
test_index <- createDataPartition(y = diabetes$Outcome, times = 1, p = 0.2, list = FALSE)
validation_index <- diabetes[-test_index,]
final_test_set <- diabetes[test_index,]

# Create training (80%) and test (20%) sets from the remaining data
validation_index <- createDataPartition(y = diabetes$Outcome, p = 0.2, list = FALSE)
train_set <- diabetes[-validation_index,]
test_set <- diabetes[validation_index,]

rm(test_index, validation_index) # Removing the unneeded dataset

# For all models, we will apply cross validation and tune parameters for optimisation where applicable
# Let's see how a GLM model fares
train_glm <- train(Outcome ~ .,
                   method = "glm",
                   trControl = trainControl(method = "cv", number = 10, p = .9),
                   data = train_set)

summary(train_glm) # This is for visualising the importance of each predictor in the model

predict_glm <- predict(train_glm, newdata = test_set, type = "raw")

results <- tibble(method = "GLM",
                           accuracy = confusionMatrix(predict_glm, test_set$Outcome)$overall[["Accuracy"]], # to evaluate accuracy of the model
                           AUC = as.numeric(roc(test_set$Outcome, as.numeric(predict_glm))$auc) # to evaluate AUC of the model
                  )
results

# Let's see how a KNN model fares
set.seed(123)
train_knn <- train(Outcome ~ ., method = "knn",
                   data = train_set,
                   tuneGrid = data.frame(k = seq(151, 251, 2)),
                   trControl = trainControl(method = "cv", number = 10, p = .9))

ggplot(train_knn, highlight = TRUE) # to visualise the relationship between k and accuracy

train_knn$bestTune # to find the best k parameter
train_knn$finalModel # to see the outcome distribution

predict_knn <- predict(train_knn, newdata = test_set, type = "raw")

results <- bind_rows(results,
                     tibble(method = "KNN",
                            accuracy = confusionMatrix(predict_knn, test_set$Outcome)$overall[["Accuracy"]],
                            AUC = as.numeric(roc(test_set$Outcome, as.numeric(predict_knn))$auc))
                     )
results

# Let's try a Loess model
train_loess <- train(Outcome ~ .,
                     method = "gamLoess",
                     tuneGrid = expand.grid(span = seq(0.5, 1.45, len = 20), degree = 1),
                     trControl = trainControl(method = "cv", number = 10, p = .9),
                     data = train_set)

ggplot(train_loess, highlight = TRUE)
varImp(train_loess)

predict_loess <- predict(train_loess, newdata = test_set, type = "raw")

results <- bind_rows(results,
                     tibble(method = "Loess",
                            accuracy = confusionMatrix(predict_loess, test_set$Outcome)$overall[["Accuracy"]],
                            AUC = as.numeric(roc(test_set$Outcome, as.numeric(predict_loess))$auc))
                     )
results

# Let's try a classification tree model
train_rpart <- train(Outcome ~ .,
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
                     trControl = trainControl(method = "cv", number = 10, p = .9),
                     data = train_set)

# display the decision tree
ggplot(train_rpart, highlight = TRUE)

# view the final decision tree
plot(train_rpart$finalModel, margin = 0.1) # plot tree structure
text(train_rpart$finalModel) # add text labels

predict_rpart <- predict(train_rpart, newdata = test_set, type = "raw")

results <- bind_rows(results,
                     tibble(method = "Decision Tree",
                            accuracy = confusionMatrix(predict_rpart, test_set$Outcome)$overall[["Accuracy"]],
                            AUC = as.numeric(roc(test_set$Outcome, as.numeric(predict_rpart))$auc))
                     )
results

# Let's try a Random Forest model
train_rf <- train(Outcome ~ .,
                  method = "Rborist",
                  tuneGrid = data.frame(predFixed = seq(2, sqrt(ncol(train_set) - 1), length = 5),
                                        minNode = seq(5, 100, length = 5)),
                  trControl = trainControl(method = "cv", number = 10, p = ,9),
                  data = train_set)

ggplot(train_rf, highlight = TRUE)

predict_rf <- predict(train_rf, newdata = test_set, type = "raw")

results <- bind_rows(results,
                     tibble(method = "Random Forest",
                            accuracy = confusionMatrix(predict_rf, test_set$Outcome)$overall[["Accuracy"]],
                            AUC = as.numeric(roc(test_set$Outcome, as.numeric(predict_rf))$auc))
                     )
results

# Let's try an Extreme Gradient Boosting model
train_xgb <- train(Outcome ~ .,
                  method = "xgbTree",
                  tuneGrid = expand.grid( # Creating a tuning parameter
                    nrounds = seq(25, 250, length = 10),
                    max_depth = c(2, 4, 6, 8),
                    eta = c(0.05, 0.1, 0.2),
                    gamma = 0,
                    colsample_bytree = 1,
                    min_child_weight = 1,
                    subsample = 1),
                  trControl = trainControl(method = "cv", number = 10, p = .9),
                  objective = "binary:logistic",
                  data = train_set) # for binary classification

ggplot(train_xgb, highlight = TRUE)

predict_xgb <- predict(train_xgb, newdata = test_set, type = "raw")

results <- bind_rows(results,
                     tibble(method = "Extreme Gradient Boosting",
                            accuracy = confusionMatrix(predict_xgb, test_set$Outcome)$overall[["Accuracy"]],
                            AUC = as.numeric(roc(test_set$Outcome, as.numeric(predict_xgb))$auc))
                     )
results

# Let's try a Naive Bayes model
train_nb <- train(Outcome ~ .,
                  method = "nb",
                  trControl = trainControl(method = "cv", number = 10, p = .9),
                  data = train_set)

predict_nb <- predict(train_nb, newdata = test_set, type = "raw")

results <- bind_rows(results,
                     tibble(method = "Naive Bayes",
                            accuracy = confusionMatrix(predict_nb, test_set$Outcome)$overall[["Accuracy"]],
                            AUC = as.numeric(roc(test_set$Outcome, as.numeric(predict_nb))$auc))
                     )
results

# Let's create a few ensemble models using a voting approach

# Let's try combining the 4 most accurate models: GLM, KNN, Loess, and XGB
# Combine predictions into a data frame
top4_votes <- data.frame(predict_glm, predict_knn, predict_loess, predict_xgb)
top4_votes <- sapply(top4_votes, as.numeric) # Converting to numeric values for analysis in next section

# Apply majority voting to get final ensemble prediction
predict_top4_ensemble <- ifelse(rowMeans(top4_votes) > 1, "1", "0")

results <- bind_rows(results,
                     tibble(method = "Top 4 Ensemble",
                            accuracy = mean(predict_top4_ensemble == test_set$Outcome),
                            AUC = as.numeric(roc(test_set$Outcome, as.numeric(predict_top4_ensemble))$auc))
                     )
results

# Let's try combining the 3 most accurate models: GLM, KNN, and XGB
# Combine predictions into a data frame
top3_votes <- data.frame(predict_glm, predict_knn, predict_xgb)
top3_votes <- sapply(top3_votes, as.numeric) # Converting to numeric values for analysis in next section

# Apply majority voting to get final ensemble prediction
predict_top3_ensemble <- ifelse(rowMeans(top3_votes) > 1, "1", "0")

results <- bind_rows(results,
                     tibble(method = "Top 3 Ensemble",
                            accuracy = mean(predict_top3_ensemble == test_set$Outcome),
                            AUC = as.numeric(roc(test_set$Outcome, as.numeric(predict_top3_ensemble))$auc))
                     )
results

# Let's try combining the 2 most accurate models: GLM and KNN
# Combine predictions into a data frame
top2_votes <- data.frame(predict_glm, predict_knn)
top2_votes <- sapply(top2_votes, as.numeric) # Converting to numeric values for analysis in next section

# Apply majority voting to get final ensemble prediction
predict_top2_ensemble <- ifelse(rowMeans(top2_votes) > 1, "1", "0")

results <- bind_rows(results,
                     tibble(method = "Top 2 Ensemble",
                            accuracy = mean(predict_top2_ensemble == test_set$Outcome),
                            AUC = as.numeric(roc(test_set$Outcome, as.numeric(predict_top2_ensemble))$auc))
                     )
results

# It is time to evaluate our chosen models' performance on our final set
predict_glm_final <- predict(train_glm, newdata = final_test_set, type = "raw")
predict_knn_final <- predict(train_knn, newdata = final_test_set, type = "raw")

final_votes <- data.frame(predict_glm_final, predict_knn_final)
final_votes <- sapply(final_votes, as.numeric)

predict_final_ensemble <- ifelse(rowMeans(final_votes) > 1, "1", "0")

final_results <- tibble(method = c("GLM", "KNN", "Top 2 Ensemble"),
                        accuracy = c(confusionMatrix(predict_glm_final, final_test_set$Outcome)$overall[["Accuracy"]],
                                     confusionMatrix(predict_knn_final, final_test_set$Outcome)$overall[["Accuracy"]],
                                     mean(predict_final_ensemble == final_test_set$Outcome)),
                        AUC = c(as.numeric(roc(final_test_set$Outcome, as.numeric(predict_glm_final))$auc),
                                as.numeric(roc(final_test_set$Outcome, as.numeric(predict_knn_final))$auc),
                                as.numeric(roc(final_test_set$Outcome, as.numeric(predict_final_ensemble))$auc))
                        )

final_results