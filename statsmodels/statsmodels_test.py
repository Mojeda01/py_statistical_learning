import statsmodels.api as sm
import pandas
from patsy import dmatrices

print()
print(" --- STATSMODELS API TEST ---")
print()

print("--- DATA ---")
df = sm.datasets.get_rdataset("Guerry", "HistData").data
vars = ["Department", "Lottery", "Literacy", "Wealth", "Region"]
print(vars)
print()

# We select the variables of interest and look at the bottom 5 rows.
df = df[vars]
print(df[-5:])
print()

# Notice that there is one missing observation in the region column. We eliminate it using a DataFrame method
# provided by pandas.

df = df.dropna()
print(df[-5:])
print()

# To fit most of the models covered by statsmodels, you will need to create two design matrices.
# The first is a matriax of endogenous variable(s) (i.e. dependent, response, regressand, etc.). The 
# second is a matrix of exogenous variable(s) (i.e. indepedent, predictor, regressor, etc.). The OLS
# coefficient estiamtes are calculated.


# Wheree y is an N*1 column of data on lottery wagers per capita (Lottery). X is N*7 with an intercept,
# the Literacy and Wealth variables, and 4 region binary variables.

# The patsy module provides a convenient function to prepare design matrices using R-like formulas.
# We use patsy's dmatrices function to create design matrices:
print("-- DESIGN MATRICES --")
y, X = dmatrices("Lottery ~ Literacy + Wealth + Region", data=df, return_type="dataframe")
print(y[:3])
print(X[:3])
print()

# Notice that dmatrices has
# 1. Split the categorical region variable into a set of indicator variables.
# 2. Added a constant to the exogenous regressors matrix.
# 3. returned pandas DataFrames instead of simple numpy arrays. This is useful because DataFrame allow
#    statsmodels to carry-over meta-data (e.g variable names) when reporting results.

print("--- MODEL FIT AND SUMMARY ---")
mod = sm.OLS(y, X)
res = mod.fit()
print(res.summary())
print()

print("--- PARAMETERS ---")
print(res.params)
print()