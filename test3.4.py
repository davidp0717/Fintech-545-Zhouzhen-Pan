import numpy as np
import pandas as pd

df = pd.read_csv("/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/testout_1.4.csv")
pc = df.to_numpy(dtype=np.float64)

pc = 0.5 * (pc + pc.T)

n = pc.shape[0]

epsilon = 1e-9
maxIter = 100
tol = 1e-9

#Higham
W = np.eye(n, dtype=np.float64)

deltaS = np.zeros((n, n), dtype=np.float64)

Yk = pc.copy()
norml = np.finfo(np.float64).max

i = 1
while i <= maxIter:
    Rk = Yk - deltaS
    Rk = 0.5 * (Rk + Rk.T)

    vals, vecs = np.linalg.eigh(Rk)
    vals = np.maximum(vals, 0.0)
    Xk = vecs @ np.diag(vals) @ vecs.T
    Xk = 0.5 * (Xk + Xk.T)

    deltaS = Xk - Rk

    Yk = Xk.copy()
    for d in range(n):
        Yk[d, d] = 1.0

    diff = Yk - pc
    norm = np.sqrt(np.sum(diff * diff))

    minEigVal = np.min(np.linalg.eigvalsh(Yk))

    if (norm - norml) < tol and minEigVal > (-epsilon):
        break

    norml = norm
    i += 1

out = Yk

np.set_printoptions(precision=15, suppress=True)
print(out)