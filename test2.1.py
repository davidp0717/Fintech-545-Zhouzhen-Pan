import pandas as pd
import numpy as np

lambda_ = 0.97

df = pd.read_csv("/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test2.csv")

X = df.to_numpy(dtype=np.float64)

n, p = X.shape

#set up weights
w = np.zeros(n, dtype=np.float64)

tw = 0.0
for i in range(n):
    w[i] = (1.0 - lambda_) * (lambda_ ** (i + 1))
    tw += w[i]

for i in range(n):
    w[i] = w[i] / tw

w = w[::-1]

#EW mean
mu = np.zeros(p, dtype=np.float64)

for t in range(n):
    for j in range(p):
        mu[j] += w[t] * X[t, j]

#EW cov
cov = np.zeros((p, p), dtype=np.float64)

for t in range(n):
    for i in range(p):
        for j in range(p):
            cov[i, j] += w[t] * (X[t, i] - mu[i]) * (X[t, j] - mu[j])

print(cov)