import numpy as np
import pandas as pd

ep = 0.0

df = pd.read_csv("/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/testout_1.3.csv")
A = df.to_numpy(dtype=np.float64)

out = A.copy()

n = out.shape[0]

#cov check
invSD = None
is_correlation = True

for i in range(n):
    if abs(out[i, i] - 1.0) > 1e-12:
        is_correlation = False
        break

#cov to corr convert
if not is_correlation:
    invSD = np.zeros(n, dtype=np.float64)
    for i in range(n):
        invSD[i] = 1.0 / np.sqrt(out[i, i])

    corr = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            corr[i, j] = invSD[i] * out[i, j] * invSD[j]

    out = corr

#Eigenvalue decomposition
vals, vecs = np.linalg.eigh(out)

for i in range(n):
    if vals[i] < ep:
        vals[i] = ep

#normalize to let diag = 1
denom = np.zeros(n, dtype=np.float64)

for i in range(n):
    s = 0.0
    for k in range(n):
        s += (vecs[i, k] * vecs[i, k]) * vals[k]
    denom[i] = s

T = np.diag(np.sqrt(1.0 / denom))

#PSD
B = T @ vecs @ np.diag(np.sqrt(vals))
out = B @ B.T

#corr to cov conversion
if invSD is not None:
    SD = np.diag(1.0 / invSD)

    out = SD @ out @ SD

np.set_printoptions(suppress=True)
print(out)
