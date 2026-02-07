import numpy as np
import pandas as pd

df = pd.read_csv("/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/testout_1.3.csv")
pc_cov = df.to_numpy(dtype=np.float64)

pc_cov = 0.5 * (pc_cov + pc_cov.T)

n = pc_cov.shape[0]

ep = 1e-9
maxIterator = 100
tol = 1e-9

invSD_vec = np.zeros(n, dtype=np.float64)
for i in range(n):
    invSD_vec[i] = 1.0 / np.sqrt(pc_cov[i, i])

pc = np.zeros((n, n), dtype=np.float64)
for i in range(n):
    for j in range(n):
        pc[i, j] = invSD_vec[i] * pc_cov[i, j] * invSD_vec[j]

pc = 0.5 * (pc + pc.T)

#Higham
W = np.eye(n, dtype=np.float64)

deltaS = np.zeros((n, n), dtype=np.float64)

Yk = pc.copy()
norml = np.finfo(np.float64).max

i = 1
while i <= maxIterator:
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

    if (norm - norml) < tol and minEigVal > (-ep):
        break

    norml = norm
    i += 1

R_higham = Yk


#corr to cov
SD_vec = np.zeros(n, dtype=np.float64)
for i in range(n):
    SD_vec[i] = 1.0 / invSD_vec[i]

cov_higham = np.zeros((n, n), dtype=np.float64)
for i in range(n):
    for j in range(n):
        cov_higham[i, j] = SD_vec[i] * R_higham[i, j] * SD_vec[j]

cov_higham = 0.5 * (cov_higham + cov_higham.T)

np.set_printoptions(suppress=True)
print(cov_higham)
