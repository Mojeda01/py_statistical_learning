
"""
This script generates artificial data for three distinct groups and performs an Ordinary Least Squares (OLS)
regression analysis using dummy variables. Group 0 serves as the omitted/benchmark category.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

def generate_data(nsample=50):
    if nsample <= 0:
        raise ValueError("Sample size must be positive.")
    
    np.random.seed(9876789)
    groups = np.zeros(nsample, int)
    groups[20:40] = 1
    groups[40:] = 2
    return groups

def prepare_ols_data(groups):
    dummy = pd.get_dummies(groups).values
    time_values = np.linspace(0, 20, len(groups))
    # Drop reference category
    X = np.column_stack((time_values, dummy[:, 1:]))
    X = sm.add_constant(X, prepend=False)
    return X

def generate_observed_response(X, beta):
    true_response = np.dot(X, beta)
    random_noise = np.random.normal(size=len(X))
    observed_response = true_response + random_noise
    return observed_response

groups = generate_data()
X = prepare_ols_data(groups)

beta = [1.0, 3, -3, 10]
y = generate_observed_response(X, beta)

res2 = sm.OLS(y, X).fit()
print(res2.summary())

pred_ols2 = res2.get_prediction()
iv_l = pred_ols2.summary_frame()["obs_ci_lower"]
iv_u = pred_ols2.summary_frame()["obs_ci_upper"]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(X[:, 1], y, "o", label="Data")
ax.plot(X[:, 1], res2.fittedvalues, "r--", label="Predicted")
ax.plot(X[:, 1], iv_u, "r--")
ax.plot(X[:, 1], iv_l, "r--")
ax.set_xlabel("Time")
ax.set_ylabel("Response")
ax.set_title("OLS Regression Analysis")
legend = ax.legend(loc="best")

plt.show()
