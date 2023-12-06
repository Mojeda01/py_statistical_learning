library(ISLR2)
library(glmnet)
library(keras)

Gitters <- na.omit(Hitters)
n <- nrow(Gitters)
set.seed(13)
ntest <- trunc(n/3)
testid <- sample(1:n, ntest)

lfit <- lm(Salary ~ ., data = Gitters[-testid, ])
lpred <- predict(lfit, Gitters[testid, ])
with(Gitters[testid, ], mean(lpred - Salary))

x <- scale(model.matrix(Salary ~ . - 1, data = Gitters))
y <- Gitters$Salary

# The first line makes a call to model.matrix(), which produces the same matrix that was used by lm()
# (the -1 omits the intercept). This function automatically converts factors to dummy variables.
# The scale() function standardizes the matrix so each column has mean zero and variance one.

cvfit <- cv.glmnet(x[-testid, ], y[-testid],
                   type.measure = "mae")
cpred <- predict(cvfit, x[testid, ], s = "lambda.min")
mean(abs(y[testid] - cpred))

# To fit the neural network, we first set up a model structure that describes the network.

modnn <- keras_model_sequential() %>%
  layer_dense(units = 50, activation = "relu",
              input_shape = ncol(x))