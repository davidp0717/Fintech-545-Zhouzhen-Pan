import math
import numpy as np

path = "/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test5_2.csv"   # <-- first file is input

Sigma = np.loadtxt(path, delimiter=",", skiprows=1)


def chol_psd(a, tol=1e-8):
    n = a.shape[0]
    L = np.zeros((n, n), dtype=float)

    for j in range(n):
        s = 0.0
        if j > 0:
            s = float(np.dot(L[j, :j], L[j, :j]))

        temp = a[j, j] - s

        #adjust tiny negative values to 0 for psd
        if temp < 0.0 and temp >= -tol:
            temp = 0.0

        #conditions for Non-PSD
        if temp < 0.0:
            raise ValueError("Matrix is not PSD ")

        L[j, j] = math.sqrt(temp)

        #move to next column for 0 eigenvalue
        if L[j, j] == 0.0:
            continue

        ir = 1.0 / L[j, j]

        for i in range(j + 1, n):
            s = 0.0
            if j > 0:
                s = float(np.dot(L[i, :j], L[j, :j]))
            L[i, j] = (a[i, j] - s) * ir
    return L


L = chol_psd(Sigma)

#simulattion
np.random.seed(1234)
Z = np.random.randn(100000, Sigma.shape[0])
X = Z @ L.T

Sigma_sim = np.cov(X, rowvar=False, ddof=1)
print(Sigma_sim)
