import math
import numpy as np

path = "/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test5_1.csv"

Sigma = np.loadtxt(path, delimiter=",", skiprows=1)

def chol_pd(a):
    n = a.shape[0]
    L = np.zeros((n, n), dtype=float)

    for j in range(n):
        s = 0.0
        if j > 0:
            s = float(np.dot(L[j, :j], L[j, :j]))

        temp = a[j, j] - s
        if temp <= 0.0:
            raise ValueError("Matrix is not PD (or numerical issue).")

        L[j, j] = math.sqrt(temp)
        ir = 1.0 / L[j, j]

        for i in range(j + 1, n):
            s = 0.0
            if j > 0:
                s = float(np.dot(L[i, :j], L[j, :j]))
            L[i, j] = (a[i, j] - s) * ir

    return L

L = chol_pd(Sigma)

#simulate
np.random.seed(1234)
Z = np.random.randn(100000, Sigma.shape[0])
X = Z @ L.T

#simulated covariance
Sigma_sim = np.cov(X, rowvar=False, ddof=1)
print(Sigma_sim)

