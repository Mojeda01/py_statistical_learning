import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Generate the data
np.random.seed(0) # Setting a seed for reproducibility
num_rows = 5
income = np.random.uniform(400, 1000, num_rows)
foodexp = np.random.uniform(250, 500, num_rows)
df = pd.DataFrame({"income":income, "foodexp":foodexp})
print(df.head())
print()

# Second data set?
data = sm.datasets.engel.load_pandas().data
print(data.head())
print()

# Least Absolute Deviation
# The LAD model is a special case of quantile regression where q=0.5
mod = smf.quantreg("foodexp ~ income", data=data)
res = mod.fit(q=0.5)
print(res.summary())

# Prepare data for plotting
# For convenience, we place the quantile regression results in a Pandas DataFrame,
# and the OLS results in dictionary

quantiles = np.arange(0.05, 0.96, 0.1)


def fit_model(q):
    res = mod.fit(q=q)
    return [q, res.params["Intercept"], res.params["income"]] + res.conf_int().loc[
        "income"
    ].tolist()


models = [fit_model(x) for x in quantiles]
models = pd.DataFrame(models, columns=["q", "a", "b", "lb", "ub"])

ols = smf.ols("foodexp ~ income", data).fit()
ols_ci = ols.conf_int().loc["income"].tolist()
ols = dict(
    a=ols.params["Intercept"], b=ols.params["income"], lb=ols_ci[0], ub=ols_ci[1]
)

print()
print(models)
print(ols)

# First plot
x = np.arange(data.income.min(), data.income.max(), 50)
get_y = lambda a, b: a + b * x

fig, ax = plt.subplots(figsize=(8,6))

for i in range(models.shape[0]):
    y = get_y(models.a[i], models.b[i])
    ax.plot(x, y, linestyle="dotted", color="grey")

y = get_y(ols["a"], ols["b"])

ax.plot(x, y, color="red", label="OLS")
ax.scatter(data.income, data.foodexp, alpha=0.2)
ax.set_xlim((240, 3000))
ax.set_ylim((240, 2000))
legend = ax.legend()
ax.set_xlabel("Income", fontsize=16)
ax.set_ylabel("Food Expenditure", fontsize=16)

plt.show()