import numpy as np
from scipy.stats import t

path = "/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test7_2.csv"

alpha = 0.05
PV = 1.0

r = np.loadtxt(path, delimiter=",", skiprows=1)

nu, mu, scale = t.fit(r)

mu = np.mean(r)
sigma = np.std(r, ddof=1)

# scale for t distribution
scale = sigma * np.sqrt((nu - 2) / nu)

# quantile
z_t = t.ppf(alpha, df=nu)

# absolute var
VaR_abs = -PV * (mu + z_t * scale)

# var diff
VaR_diff = -PV * (z_t * scale)

print(VaR_abs)
print(VaR_diff)
