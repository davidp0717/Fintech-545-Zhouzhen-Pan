import numpy as np
import pandas as pd

df = pd.read_csv("/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/testout_3.1.csv")
A = df.to_numpy(dtype=np.float64)

A = 0.5 * (A + A.T)

n = A.shape[0]

root = np.zeros((n, n), dtype=np.float64)

for j in range(n):
    s = 0.0
    if j > 0:
        s = root[j, :j].T @ root[j, :j]
    temp = A[j, j] - s

    if (temp <= 0.0) and (temp >= -1e-8):
        temp = 0.0

    root[j, j] = np.sqrt(temp) if temp >= 0.0 else np.nan

    if root[j, j] != 0.0:
        ir = 1.0 / root[j, j]

        for i in range(j + 1, n):
            s = 0.0
            if j > 0:
                s = root[i, :j].T @ root[j, :j]

            root[i, j] = (A[i, j] - s) * ir

np.set_printoptions(suppress=True)
print(root)

