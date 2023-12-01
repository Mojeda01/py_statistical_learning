import statsmodels.api as sm
import pandas
from patsy import dmatrices

print()
print(" --- STATSMODELS API TEST ---")
print()

df = sm.datasets.get_rdataset("Guerry", "HistData").data
vars = ["Department", "Lottery", "Literacy", "Wealth", "Region"]
print(vars)
print()

