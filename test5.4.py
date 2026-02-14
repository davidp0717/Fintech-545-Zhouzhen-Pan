import math
import numpy as np

path = "/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test5_3.csv"

ep = 1e-9
maxIter = 100
tol = 1e-9

pc_cov = np.loadtxt(path, delimiter=",", skiprows=1).astype(np.float64)

pc_cov = 0.5 * (pc_cov + pc_cov.T)

n = pc_cov.shape[0]

#cov to corr
invSD_vec = np.zeros(n, dtype=np.float64)
for i in range(n):
    invSD_vec[i] = 1.0 / math.sqrt(pc_cov[i, i])

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

    if (norm - norml) < tol and minEigVal > (-ep):
        break

    norml = norm
    i += 1

R_higham = Yk


# corr to cov: Cov = D * R * D
SD_vec = np.zeros(n, dtype=np.float64)
for i in range(n):
    SD_vec[i] = 1.0 / invSD_vec[i]

cov_higham = np.zeros((n, n), dtype=np.float64)
for i in range(n):
    for j in range(n):
        cov_higham[i, j] = SD_vec[i] * R_higham[i, j] * SD_vec[j]

cov_higham = 0.5 * (cov_higham + cov_higham.T)


#PSD Cholesky
def chol_psd(a, tol=1e-8):
    n = a.shape[0]
    L = np.zeros((n, n), dtype=float)

    for j in range(n):
        s = 0.0
        if j > 0:
            s = float(np.dot(L[j, :j], L[j, :j]))

        temp = a[j, j] - s

        #adjust value close to 0
        if temp < 0.0 and temp >= -tol:
            temp = 0.0

        if temp < 0.0:
            raise ValueError(f"Matrix not PSD at diagonal {j}: temp={temp}")

        L[j, j] = math.sqrt(temp)

        # skip column
        if L[j, j] == 0.0:
            continue

        ir = 1.0 / L[j, j]

        for i in range(j + 1, n):
            s = 0.0
            if j > 0:
                s = float(np.dot(L[i, :j], L[j, :j]))
            L[i, j] = (a[i, j] - s) * ir

    return L


#monte carlo simulation
L = chol_psd(cov_higham)

np.random.seed(1234)
Z = np.random.randn(100000, n)
X = Z @ L.T 

Sigma_sim = np.cov(X, rowvar=False, ddof=1)

np.set_printoptions(suppress=True)
print(Sigma_sim)
