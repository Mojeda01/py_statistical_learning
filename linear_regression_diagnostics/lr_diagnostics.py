import statsmodels
import statsmodels.formula.api as smf
import pandas as pd

# Load data
data_url = "https://raw.githubusercontent.com/nguyen-toan/ISLR/07fd968ea484b5f6febc7b392a28eb64329a4945/dataset/Advertising.csv"
df = pd.read_csv(data_url).drop('Unnamed: 0', axis=1)
print(df.head())

# Fitting Linear Model
res = smf.ols(formula="Sales ~ TV + Radio + Newspaper", data=df).fit()
print(res.summary())

## Diagnostic Figures/Table
# a. residual
# b. qq
# c. scale location
# d. leverage
## And table
# a. vif