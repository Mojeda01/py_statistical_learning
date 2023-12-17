library(glmnet)

# Fixing the data
DKKNOK <- EXR.USDEURGBP$OBS_VALUE[1:1260]
USDNOK <- EXR.USDEURGBP$OBS_VALUE[1261:2520]
GBPNOK <- EXR.USDEURGBP$OBS_VALUE[2521:3780]
EURNOK <- EXR.USDEURGBP$OBS_VALUE[3781:5040]
SEKNOK <- EXR.USDEURGBP$OBS_VALUE[5041:6300]
DATES <- EXR.USDEURGBP$TIME_PERIOD[1:1260]

# Using the fixed data to create an dataframe
df <- data.frame(DKKNOK, USDNOK, GBPNOK, EURNOK, SEKNOK)

ln <- lm(df$USDNOK ~ df$DKKNOK + df$GBPNOK + df$EURNOK + df$SEKNOK, data = df)
summary(ln)
plot(ln)

predictions <- predict(ln)

# Plot your data
plot(predictions, type = "l", main = "Plot of Predicted values",
     xlab = "Date", ylab = "predicted vs. real USDNOK", col = "magenta", frame.plot = FALSE, xaxt = 'n')

lines(df$USDNOK, col = "lightblue")

legend(1, 11.0, legend=c("Predicted Values", "Real Dataset"),
       col=c("magenta", "lightblue"), lty=1:2, cex=0.5)

###########################################################################
# RIDGE REGRESSION

# Prepare the data
x <- as.matrix(df[, c("DKKNOK", "GBPNOK", "EURNOK", "SEKNOK")])
y <- df$USDNOK

# Fit the ridge regression model
ridge_model <- glmnet(x, y, alpha = 0)

# View the model
print(ridge_model)

# Optional: Use cross-validation to find the optimal lambda
cv_ridge <- cv.glmnet(x, y, alpha = 0)
best_lambda <- cv_ridge$lambda.min
print(best_lambda)

# Fit the model with the best lambda
ridge_model_opt <- glmnet(x, y, alpha = 0, lambda = best_lambda)
print(ridge_model_opt)

ridge_predictions <- predict(ridge_model_opt, newx = x)

# Plot your data
plot(predictions, type = "l", main = "Plot of Predicted values",
     xlab = "Date", ylab = "predicted vs. real USDNOK", col = "magenta", frame.plot = FALSE, xaxt = 'n')

lines(df$USDNOK, col = "lightblue")
lines(ridge_predictions, col="green")

legend(1, 11.0, legend=c("Linear Regression", "Real Dataset", "Ridge Regression"),
       col=c("magenta", "lightblue", "green"), lty=1:2, cex=0.5)

###########################################################################
# LASSO REGRESSION

# Fit the Lasso regression model
lasso_model <- glmnet(x, y, alpha = 1)

# Optionally use cross-validation to find the optimal lambda
cv_lasso <- cv.glmnet(x, y, alpha = 1)
best_lambda <- cv_lasso$lambda.min

# Fit the model using the best lambda
lasso_model_opt <- glmnet(x, y, alpha = 1, lambda = best_lambda)

lasso_predictions <- predict(lasso_model_opt, newx = x)

# Plot your data
plot(predictions, type = "l", main = "Plot of Predicted values",
     xlab = "Date", ylab = "predicted vs. real USDNOK", col = "magenta", frame.plot = FALSE, xaxt = 'n')

lines(df$USDNOK, col = "lightblue")
lines(ridge_predictions, col="green")
lines(lasso_predictions, col="red")

legend(1, 11.0, legend=c("Linear Regression", "Real Dataset", "Ridge Regression", "Lasso Regression"),
       col=c("magenta", "lightblue", "green", "red"), lty=1:2, cex=0.5)


mean(predictions)
mean(ridge_predictions)
mean(lasso_predictions)

