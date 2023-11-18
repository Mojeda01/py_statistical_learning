import numpy as np
import pandas as pd

print("Random Series with Index")
s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
print(s)
print(s.index)
print("------")
print()

print("Random Series as numbered list")
s1 = pd.Series(np.random.randn(5))
print(s1)
print("------")
print()

d = {"b": 1, "a":0, "c":2}
dSeries = pd.Series(d)
print(dSeries)
print("------")
print()

d1 = {"a":0.0, "b":1.0, "c":2.0}
print(pd.Series(d))
print(pd.Series(d, index=["b", "c", "d", "a"]))
































