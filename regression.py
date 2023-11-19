import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)

print()
print(" - LINEAR REGRESSION - ")
print()

boston = load_data("Boston")
print(boston.columns)

X = pd.DataFrame({"intercept" : np.ones(boston.shape[0]),
                  "lstat" : boston["lstat"]})
print()
print(X[:4])
print()

y = boston["medv"]
model = sm.OLS(y, X)
results = model.fit()
print(summarize(results))
print()

design = MS(['lstat'])
design = design.fit(boston)
X = design.transform(boston)
print(X[:4])
print()

design = MS(["lstat"])
X = design.fit_transform(boston)
print(X[:4])
print()

print(results.summary())
print(results.params)
print()

new_df = pd.DataFrame({"lstat":[5, 10, 15]})
newX = design.transform(new_df)
print(newX)
print()

new_predictions = results.get_prediction(newX)
print(new_predictions.predicted_mean)
print()
print(new_predictions.conf_int(alpha=0.05))
print()
print(new_predictions.conf_int(obs=True, alpha=0.05))
print()
#-------------------------------------------------------

def abline(ax, b, m):
    # Add a line with slope m and intercept b to ax
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b,  m * xlim[1] + b]
    ax.plot(xlim, ylim)

def abline(ax, b, m, *args, **kwargs):
    # Add a line with slope m and intercept b to ax
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim, *args, **kwargs)

ax = boston.plot.scatter("lstat", "medv")
abline(ax, results.params[0], results.params[1],
       "r--", linewidth=3)
ax = subplots(figsize=(8, 8))
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel("Fitted value")
ax.set_ylabel("Residual")
ax.axhline(0, c="k", ls="--")








