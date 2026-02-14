import numpy as np
import math

path = "/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test7_2.csv"
alpha = 0.05
PV = 1.0

x = np.loadtxt(path, delimiter=",", skiprows=1)

x = np.sort(x)
n = x.shape[0]

n_up = int(math.ceil(alpha * n)) - 1
n_dn = int(math.floor(alpha * n)) - 1

v = 0.5 * (x[n_up] + x[n_dn])

VaR_abs = -PV * v
mu = float(np.mean(x))
VaR_diff = VaR_abs + PV * mu

print(VaR_abs)
print(VaR_diff)