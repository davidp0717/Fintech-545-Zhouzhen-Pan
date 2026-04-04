import numpy as np
import pandas as pd
from scipy.optimize import minimize


covar = pd.read_csv("testfiles/data/test5_3.csv").to_numpy(dtype=float)
means = pd.read_csv("testfiles/data/test10_3_means.csv")["Mean"].to_numpy(dtype=float)

rf = 0.04
n = len(means)


# objective

def neg_sharpe(w):
    port_return = w @ means - rf
    port_vol = np.sqrt(w @ covar @ w)
    return -port_return / port_vol

# optimization

constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
bounds = [(0.1, 0.5)] * n
w0 = np.full(n, 1.0 / n)

result = minimize(
    neg_sharpe,
    w0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-12, "maxiter": 1000}
)

w = result.x
w = w / np.sum(w)


print("W")
for val in w:
    print(val)