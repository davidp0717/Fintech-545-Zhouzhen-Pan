import numpy as np
import pandas as pd
from scipy.optimize import minimize


cov_df = pd.read_csv("/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test5_2.csv")
covar = cov_df.to_numpy(dtype=float)
n = covar.shape[0]


# target
target_budget = np.array([2, 2, 2, 2, 1], dtype=float)
target_budget = target_budget / target_budget.sum()


# Portfolio volatility
def pvol(w):
    return np.sqrt(w @ covar @ w)


# standard deviation
def pCSD(w):
    sigma_p = pvol(w)
    return w * (covar @ w) / sigma_p


def sse_risk_budget(w):
    csd = pCSD(w)
    risk_share = csd / np.sum(csd)
    return 1.0e5 * np.sum((risk_share - target_budget) ** 2)


w0 = np.ones(n) / n


constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
bounds = [(0.0, None)] * n


# optimize

res = minimize(
    sse_risk_budget,
    w0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-12, "maxiter": 2000}
)

w = res.x
w = w / np.sum(w)


output = pd.DataFrame({"W": w})
print(output.to_string(index=False))