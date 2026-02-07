import pandas as pd
import numpy as np

lambda_var = 0.97
lambda_corr = 0.94

df = pd.read_csv("/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test2.csv")
X = df.to_numpy(dtype=np.float64)

n, p = X.shape

#weights
w_var = np.zeros(n, dtype=np.float64)
tw = 0.0
for i in range(n):
    w_var[i] = (1.0 - lambda_var) * (lambda_var ** (i + 1))
    tw += w_var[i]
for i in range(n):
    w_var[i] = w_var[i] / tw
w_var = w_var[::-1]

#EW mean for var
mu_var = np.zeros(p, dtype=np.float64)
for t in range(n):
    for j in range(p):
        mu_var[j] += w_var[t] * X[t, j]

#EW cov for var
cov_var = np.zeros((p, p), dtype=np.float64)
for t in range(n):
    for i in range(p):
        for j in range(p):
            cov_var[i, j] += w_var[t] * (X[t, i] - mu_var[i]) * (X[t, j] - mu_var[j])

#D_97
stddev_97 = np.zeros(p, dtype=np.float64)
for i in range(p):
    stddev_97[i] = np.sqrt(cov_var[i, i])

D_97 = np.diag(stddev_97)

#EW cov
#weights for corr
w_corr = np.zeros(n, dtype=np.float64)
tw = 0.0
for i in range(n):
    w_corr[i] = (1.0 - lambda_corr) * (lambda_corr ** (i + 1))
    tw += w_corr[i]
for i in range(n):
    w_corr[i] = w_corr[i] / tw
w_corr = w_corr[::-1]

#EW mean for corr
mu_corr = np.zeros(p, dtype=np.float64)
for t in range(n):
    for j in range(p):
        mu_corr[j] += w_corr[t] * X[t, j]

#EW cov for corr
cov_corr = np.zeros((p, p), dtype=np.float64)
for t in range(n):
    for i in range(p):
        for j in range(p):
            cov_corr[i, j] += w_corr[t] * (X[t, i] - mu_corr[i]) * (X[t, j] - mu_corr[j])

#R_94
R_94 = np.zeros((p, p), dtype=np.float64)
for i in range(p):
    for j in range(p):
        denom = np.sqrt(cov_corr[i, i] * cov_corr[j, j])
        R_94[i, j] = cov_corr[i, j] / denom

final_cov = D_97 @ R_94 @ D_97
print(final_cov)
