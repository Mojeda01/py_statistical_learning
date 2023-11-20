import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Artificial Data
nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x ** 2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)

# Our model needs an intercept so we add a column of 1s:
X1 = sm.add_constant(X)
y = np.dot(X1, beta) + e

# Fit and Summary
model = sm.OLS(y, X1)
results = model.fit()
print(results.summary())
print()

print("Parameters: ", results.params)
print("R2: ", results.rsquared)

# OLS non-linear curve but linear in parameters
nsample = 50
sig = 0.5
x = np.linspace (0, 20, nsample)
X = np.column_stack((x, np.sin(x), (x-5) ** 2, np.ones(nsample)))
beta = [0.5, 0.5, -0.02, 5.0]

y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)

# Fit and summary
res = sm.OLS(y, X).fit()
print(res.summary())

# Extract other quantities of interest
print("Parameters: ", res.params)
print("Standard Errors: ", res.bse)
print("Predicted Values: ", res.predict())

# Draw a plot to compare the true relationship to OLS predictor. Confidence intervals around the predictors
# are built using the wls_prediction_std command.

pred_ols = res.get_prediction()
iv_l = pred_ols.summary_frame()["obs_ci_lower"]
iv_u = pred_ols.summary_frame()["obs_ci_upper"]

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x, y, "o", label="data")
ax.plot(x, y_true, "b-", label="True")
ax.plot(x, res.fittedvalues, "r--.", label="OLS")
ax.plot(x, iv_u, "r--")
ax.plot(x, iv_l, "r--")
ax.legend(loc="best")
plt.show()























