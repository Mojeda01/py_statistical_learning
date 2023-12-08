# Necessary Libraries
library(glmnet)
library(keras)

# Setting up Data
nba_train <- read.table("nba_train.txt", sep=";", header=TRUE)
nba_test <- read.table("nba_test.txt", sep=";", header=TRUE)

# Clean the data
nba_train_clean <- na.omit(nba_train)
nba_test_clean <- na.omit(nba_test)

# Remove unclean data
rm(nba_test, nba_train)

#-------------------------------------------------------------------------------
# Multiple Linear Regression - MLRv1

model_v1 <- lm(X3P. ~ G + GS + MP + X2P + X2PA + X2P.
               + FT + FTA + FT. + ORB + DRB + TRB + AST + STL
               + BLK  + TOV + PF + PTS, data = nba_train_clean)
summary(model_v1)

# Plotting the regression
fitted_values <- predict(model_v1) # calculate the fitted values from the model
observed_values <- model_v1$model$X3P. # Get the observed values of the response variable.
plot(observed_values, fitted_values, xlab="Observed Values", ylab="Predicted Values for X3P.",
     main="Scatter plot of Observed vs. Fitted values V1")
abline(a=0, b=1, col="red") # Add a fitted line.

#-------------------------------------------------------------------------------
# Multiple Linear Regression - MLRv2
model_v2 <- lm(X3P. ~ PTS + TOV + FT. + X2P , data = nba_train_clean)
summary(model_v2)

# Plotting the regression for model_v2 
fitted_values_v2 <- predict(model_v2)
plot(observed_values, fitted_values_v2, xlab="Observed Values", ylab="Predicted Values for X3P. Training",
     main="Scatter plot of Observed vs. Fitted values V2")
abline(a=0, b=1, col="red") # Add a fitted line.

# Make predictions using the model on the test set.
test_predictions <- predict(model_v2, newdata=nba_test_clean) # The test predictions.
test_pred_df <- as.data.frame(test_predictions)
# The test_prediction object now contains the predicted X3P. values for the

# par(mfrow) makes it possible to display two graphs.
par(mfrow=c(2,2))
# Create a density plot of the predicted values
plot(density(test_predictions), main = "Density of Predicted X3P.", xlab = "Predicted X3P.", ylab = "Density")
# Histogram
plot(hist(test_predictions), main = "Histogram for test predictions", xlab="Test Predictions")

# Valid data for submission.
yhat = c(31.00430, 28.52672, 24.49877, 30.96667, 25.19085, 39.29634, 37.19372, 29.54059, 32.97256, 25.04399, 27.64957, 30.07482,
         30.31519, 34.26984, 31.94492, 28.01404, 35.76279, 35.36992, 35.45747, 28.69800, 29.82676, 35.73305, 41.81628, 
         31.65745, 45.03199, 35.38210, 25.93590, 39.99549, 30.09085, 36.04705, 32.36596, 30.31437, 43.52798, 27.40965,
         30.89658, 40.63883, 30.01654, 25.90254, 34.74504, 43.14734, 38.32288, 40.27859, 39.57262, 28.58134, 38.06137,
         26.00364, 44.13155, 27.14372, 28.50152, 33.78245, 30.84091, 26.67777, 31.09945, 25.68920, 32.22902, 36.29952,
         35.09252, 38.75306, 25.41976, 27.78649, 30.77884, 38.91083, 26.86374, 34.76578, 47.02081, 31.16002, 42.97429,
         22.45212, 29.17299, 28.79225, 28.01001, 24.66684, 26.76315, 34.30999, 33.24768, 36.82644, 34.88421, 27.03602,
         39.40955, 24.17156, 31.74596, 31.83858, 32.59989, 34.02872, 30.28441, 31.23077, 33.08012, 27.99908, 35.40451,
         25.26869, 38.44384, 33.76697, 32.01500, 26.03491, 30.62186, 22.78677, 27.59709)

meanyhat <- mean(yhat)
meanyhat
#-------------------------------------------------------------------------------
# Ridge and Lasso Regression that follows MLRv2

# Prepare the data
x <- model.matrix(X3P. ~ PTS + TOV + FT. + X2P, data = nba_train_clean[,-1]) # Predictor matrix, removing intercept
y <- nba_train_clean$X3P. # Response vector

# Fit ridge regression model
# alpha = 0 indicates ridge regression
ridge_model <- glmnet(x, y, alpha = 0, standardize = TRUE)
# Perform cross-validation to find the optimal lambda
cv_ridge <- cv.glmnet(x, y, alpha = 0, standardize = TRUE)
# Extract the optimal lambda value
optimal_lambda <- cv_ridge$lambda
# View the model coefficients at the optimal lambda
coef_ridge <- coef(ridge_model, s = optimal_lambda)
print(coef_ridge)
# Predict using the ridge regression model at the optimal lambda
ridge_pred <- predict(ridge_model, newx = x, s = optimal_lambda)

# Plotting the data
plot(hist(ridge_pred), main="Histogram for ridge prediction X3P.", xlab="Ridge Prediction for X3P.")
plot(density(ridge_pred), main="Density function plot for ridge prediction of X3P.",
     xlab="Ridge prediction for X3P.")

# Lasso Regression - glmnet already loaded
lasso_model <- glmnet(x, y, alpha = 1, standardize = TRUE) # Fit the lasso regression model
cv_lasso <- cv.glmnet(x, y, alpha = 1, standardize = TRUE) # Perform cross-validation to find optimal lambda
optimal_lambda_lasso <- cv_lasso$lambda.min # Extract the optimal lambda value
coef_lasso <- coef(lasso_model, s = optimal_lambda_lasso) # View the model coefficients at the optimal lambda
lasso_pred <- predict(lasso_model, newx = x, s = optimal_lambda)

# Plotting lasso Regression
plot(hist(lasso_pred), main="Histogram for lasso prediction X3P.", xlab="lasso Prediction for X3P.")
plot(density(lasso_pred), main="Density function plot for lasso prediction of X3P.",
     xlab="lasso prediction for X3P.")

# Lasso and Ridge regression predictions
mean(ridge_pred)
mean(lasso_pred)

lr_mlrv2_diff <- meanyhat - mean(ridge_pred) # Difference between MLR v2 and ridge/lasso prediction.
lr_mlrv2_diff

#-------------------------------------------------------------------------------
# Neural Network - nn_model_v1

# Prepare data for neural network
x_train <- as.matrix(nba_train_clean[c("PTS", "TOV", "FT.", "X2P")])
y_train <- nba_train_clean$X3P.

x_test <- as.matrix(nba_test_clean[c("PTS", "TOV", "FT.", "X2P")])
# Note: we do not have y_test yet because we are predicting it.

# Define the model architecture
nn_model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = 'relu', input_shape = dim(x_train)[2]) %>%
  layer_dense(units = 1) # No activation function for regression.

# Compile the model for regression
nn_model %>% compile(
  loss = "mean_squared_error", # Suitable for regression
  optimizer = optimizer_rmsprop(),
  metrics = c("mean_absolute_error")
)

# Fit the model to the training data
history <- nn_model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2 # Use a portion of the training data as a validation set.
)

predictions <- nn_model %>% predict(x_test)
mean(predictions)

#-------------------------------------------------------------------------------
# DATASETS GENERATED FROM THE MODELS.

# MLRv2
meanyhat <- mean(yhat)
meanyhat
# Ridge and Lasso Regression
mean(ridge_pred)
mean(lasso_pred)
#nn_model_v1
mean(predictions)

# The mean of all models
mean_all_models <- mean(meanyhat, mean(ridge_pred), mean(lasso_pred), mean(predictions))
mean_all_models

# THE DATASET SUBMISSION FOR EXERCISE 1:
nn_model_v1_dataset <- c(33.03048, 33.89549, 23.36052, 30.88625, 31.33442, 
                         34.18607, 30.71064, 31.63374, 32.89982, 24.38804,
                         26.40159, 27.85608, 32.37954, 32.23981, 29.02593,
                         29.01753, 35.22495, 34.92168, 34.92273, 36.74648,
                         38.72455, 29.82740, 38.11998, 29.65278, 35.52913,
                         35.33921, 29.20581, 39.74421, 30.11693, 35.45050,
                         32.87713, 38.80767, 31.57839, 26.62194, 38.88888,
                         32.67337, 25.31357, 25.75896, 34.92522, 32.58925,
                         31.99974, 35.89716, 34.66804, 28.41680, 34.05063,
                         23.36460, 32.51867, 32.33500, 32.29827, 32.89693,
                         29.73924, 28.57775, 30.99202, 30.59409, 34.92828,
                         36.79679, 30.61975, 34.41978, 20.09712, 27.73823,
                         29.01493, 34.34999, 16.94636, 26.96427, 39.35625,
                         29.97249, 33.25623, 32.04742, 28.46832, 31.60778,
                         27.33512, 27.85569, 30.22571, 32.42680, 34.39135,
                         38.55454, 36.55794, 33.59258, 31.07469, 23.73543,
                         32.79361, 33.34867, 30.15983, 32.80956, 30.67480,
                         34.81236, 37.15068, 25.33694, 33.73828, 31.98571,
                         38.48336, 33.31398, 36.96995, 23.48883, 33.20068,
                         27.21384, 28.97112)