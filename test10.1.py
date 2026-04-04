import numpy as np
import pandas as pd
from scipy.optimize import minimize


cov_df = pd.read_csv("/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test5_2.csv")
covar = cov_df.to_numpy(dtype=float)
n = covar.shape[0]

# Portfolio volatility
def pvol(w):
    return np.sqrt(w @ covar @ w)

# Component standard deviation
def pCSD(w):
    p_vol = pvol(w)
    return w * (covar @ w) / p_vol

# Sum squared error of component SDs
def sseCSD(w):
    csd = pCSD(w)
    mCSD = np.mean(csd)
    dCsd = csd - mCSD
    return 1.0e5 * np.sum(dCsd ** 2)


w0 = np.ones(n) / n

# Constraints and bounds
constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
bounds = [(0.0, None)] * n

# Optimize
res = minimize(
    sseCSD,
    w0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-12, "maxiter": 2000}
)

w = res.x
w = w / np.sum(w)


result = pd.DataFrame({"W": w})
print(result.to_string(index=False))