import numpy as np
from scipy.stats import norm

path = "/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test7_1.csv"

alpha = 0.05
PV = 1.0

r = np.loadtxt(path, delimiter=",", skiprows=1)

mu = np.mean(r)
sigma = np.std(r, ddof=1)

# quantile
z = norm.ppf(alpha)

# absolute var
VaR_abs = -PV * (mu + z * sigma)

# var difference
VaR_diff = -PV * (z * sigma)

print(VaR_abs)
print(VaR_diff)

