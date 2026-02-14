import math
import numpy as np

path = "/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test5_3.csv"
A = np.loadtxt(path, delimiter=",", skiprows=1)

def chol_psd(a, tol=1e-8):
    n = a.shape[0]
    L = np.zeros((n, n), dtype=float)

    for j in range(n):
        s = 0.0
        if j > 0:
            s = float(np.dot(L[j, :j], L[j, :j]))

        temp = a[j, j] - s

        if temp < 0.0 and temp >= -tol:
            temp = 0.0

        if temp < 0.0:
            raise ValueError("Matrix is not PSD")

        L[j, j] = math.sqrt(temp)

        if L[j, j] == 0.0:
            continue

        ir = 1.0 / L[j, j]

        for i in range(j + 1, n):
            s = 0.0
            if j > 0:
                s = float(np.dot(L[i, :j], L[j, :j]))
            L[i, j] = (a[i, j] - s) * ir

    return L


# near_psd
def near_psd(a, epsilon=0.0):
    n = a.shape[0]
    out = a.copy()
    
    is_corr = np.allclose(np.diag(out), np.ones(n), rtol=1e-7, atol=1e-7)

    invSD = None
    if not is_corr:
        sd = np.sqrt(np.diag(out))
        invSD = np.diag(1.0 / sd)
        out = invSD @ out @ invSD

    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    vecs_sq = vecs * vecs 
    
    #SVD, update the eigen value and scale
    t = 1.0 / (vecs_sq @ vals)         
    T = np.diag(np.sqrt(t))
    Lvals = np.diag(np.sqrt(vals))
    B = T @ vecs @ Lvals
    out = B @ B.T

    #Add back the variance
    if invSD is not None:
        SD = np.diag(1.0 / np.diag(invSD))  # this is D
        out = SD @ out @ SD

    return out


#nonPSD matrix -> PSD matrix
Sigma = near_psd(A, epsilon=0.0)

#matrix root
L = chol_psd(Sigma)

#simulate
np.random.seed(1234)
Z = np.random.randn(100000, Sigma.shape[0])   # Z ~ N(0, I)

X = Z @ L.T

Sigma_sim = np.cov(X, rowvar=False, ddof=1)
print(Sigma_sim)
