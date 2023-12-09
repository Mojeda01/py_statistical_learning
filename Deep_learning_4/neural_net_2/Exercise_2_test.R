#Libraries
library(boot)
library(glmnet)
library(keras)

# Load the datasets
train <- read.table("framingham_train.txt", sep=";", header=TRUE)
test <- read.table("framingham_test.txt", sep=";", header=TRUE)

# *****************************************************************************
# TASK (a) ********************************************************************

# Initialize a vector to hold the predictions
predictions <- numeric(nrow(train))

# Loop through each row in the dataset
for (i in 1:nrow(train)) {
  # Fit the model without the ith observation
  fit <- glm(TenYearCHD ~ ., data = train[-i, ], family = binomial())
  
  #Predict the omitted observation
  probability <- predict(fit, newdata = train[i, , drop = FALSE], type = "response")
  predictions[i] <- ifelse(probability > 0.5, 1, 0)
}

# Calculate the LOOCV error rate
loocv_error_rate <- mean(predictions != train$TenYearCHD)

# Print the LOOCV error rate
print(loocv_error_rate)

print(predictions) # For the training set

#--------------------------------------------------------------
# Test set logistic regression

# Fit the logistic regression model using the entire training set
final_model <- glm(TenYearCHD ~ . , data = train, family = binomial())

# Predict the probabbilities on the test set
test_probabilities <- predict(final_model, newdata = test, type = "response")

# Convert probabilities to binary predictions using a threshold of 0.5
test_predictions <- ifelse(test_probabilities > 0.5, 1, 0)

# Evaluate the model's performance on the test set
test_error_rate <- mean(test_predictions != test$TenYearCHD)

# Print the test error rate
print(test_error_rate) # test error rate for the test set.
print(loocv_error_rate) # test error rate for the training set

# *****************************************************************************
# TASK (b) ********************************************************************

# Prepare the matrix of predictors and the response vector
x_train <- as.matrix(train[, -which(names(train) == "TenYearCHD")])
y_train <- train$TenYearCHD

# Perform cross-validation to find the optimal lambda
cv_fit <- cv.glmnet(x_train, y_train, family = "binomial", type.measure = "class")

# Extract the optimal lambda value
optimal_lambda <- cv_fit$lambda.min

# Fit the Lasso model on the training set using the optimal lambda
lasso_model <- glmnet(x_train, y_train, family = "binomial", lambda = optimal_lambda)

# Predict on the training set
train_probabilities <- predict(lasso_model, newx = x_train, type = "response")
train_predictions <- ifelse(train_probabilities > 0.5, 1, 0)
train_error_rate <- mean(train_predictions != y_train)

# Predict on the test set
x_test <- as.matrix(test[, -which(names(test) == "TenYearCHD")])
y_test <- test$TenYearCHD

test_probabilities <- predict(lasso_model, newx = x_test, type = "response")
test_predictions <- ifelse(test_predictions > 0.5, 1, 0)
test_error_rate <- mean(test_predictions != y_test)

#Output the optimal lambda and error rates
list(optimal_lambda = optimal_lambda, train_error_rate = train_error_rate,
     test_error_rate = test_error_rate)

# *****************************************************************************
# TASK (c) ********************************************************************

x_train <- as.matrix(train[, -which(names(train) == "TenYearCHD")])
y_train <- as.numeric(train$TenYearCHD) # Binary outcomes need to be numeric

# Define the network architecture
model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu", input_shape = dim(x_train)[2]) %>%
  layer_dense(units = 1, activation = "sigmoid") # output layer for binary classification

# Compile the model
model %>% compile(
  loss = "binary_crossentropy", # Appropriate for binary outcomes
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

# Train the model
history <- model %>% fit(
  x_train, y_train,
  epochs = 100, # The number of epochs should be chosen based on validation performance
  batch_size = 32, # Batch size can be adjusted based on dataset size.
  validation_split = 0.2 # Hold out 20% of the data for validation.
)

# Predict the probabilities on the test set
test_probs <- model %>% predict(x_test)

#Convert probabilities to binary predictions using a threshold of 0.5
test_preds <- ifelse(test_probs > 0.5, 1, 0)

# Calculate the test error rate
test_error_rate <- mean(test_preds != y_test)

# Print out the test error rate
cat('Test error rate:', test_error_rate, '\n')

# *****************************************************************************
# TASK (d) ********************************************************************

# Rates to be beat
a <- 0.1467047
b <- 0.17
c <- 0.17

x_train <- as.matrix(train[, -which(names(train) == "TenYearCHD")])
y_train <- as.numeric(train$TenYearCHD) # Binary outcomes need to be numeric


# Add early stopping
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 25, restore_best_weights = TRUE)
# Implement learning rate to layers
reduce_lr <- callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.2, patience = 5, min_lr = 0.001)

# Define the network architecture
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = 'relu', kernel_initializer = 'he_normal', input_shape = dim(x_train)[2]) %>%
  layer_dense(units = 32, activation = 'relu', kernel_initializer = 'he_normal') %>%
  layer_dense(units = 1, activation = 'sigmoid')

# Compile the model
model %>% compile(
  loss = "binary_crossentropy", # Appropriate for binary outcomes
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = c("accuracy")
)

# Train the model
history <- model %>% fit(
  x_train, y_train,
  epochs = 40, 
  batch_size = 32, 
  validation_split = 0.2, 
  callbacks = list(early_stop, reduce_lr)
)

# Predict the probabilities on the test set
test_probs <- model %>% predict(x_test)

#Convert probabilities to binary predictions using a threshold of 0.5
test_preds <- ifelse(test_probs > 0.5, 1, 0)

# Calculate the test error rate
test_error_rate <- mean(test_preds != y_test)

# Print out the test error rate
cat('Test error rate:', test_error_rate, '\n')








