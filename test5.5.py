import numpy as np

path = "/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test5_2.csv"

Sigma = np.loadtxt(path, delimiter=",", skiprows=1).astype(float)
Sigma = 0.5 * (Sigma + Sigma.T)


def simulate_pca_explicit(a, nsim, pct_explained=0.99, tol_eig=1e-8):

    #Eigenvalue decomp
    vals, vecs = np.linalg.eigh(a)

    #reverse order
    vals = vals[::-1]
    vecs = vecs[:, ::-1]

    #total variance
    total_variance = 0.0
    for v in vals:
        total_variance += v

    #only keep positive values
    positive_vals = []
    positive_vecs = []

    for i in range(len(vals)):
        if vals[i] >= tol_eig:
            positive_vals.append(vals[i])
            positive_vecs.append(vecs[:, i])

    positive_vals = np.array(positive_vals)
    positive_vecs = np.column_stack(positive_vecs)

    cumulative_variance = 0.0
    K = 0

    for i in range(len(positive_vals)):
        cumulative_variance += positive_vals[i]
        K += 1
        if cumulative_variance / total_variance >= pct_explained:
            break

    #keep first k
    vals_kept = positive_vals[:K]
    vecs_kept = positive_vecs[:, :K]

    # B = Q * sqrt(Lambda)
    sqrt_vals = np.zeros((K, K))
    for i in range(K):
        sqrt_vals[i, i] = np.sqrt(vals_kept[i])

    B = vecs_kept @ sqrt_vals

    np.random.seed(1234)
    Z = np.random.randn(K, 100000)
    X = (B @ Z).T

    return X


sim = simulate_pca_explicit(Sigma, 100000, pct_explained=0.99)


Sigma_sim = np.cov(sim, rowvar=False, ddof=1)

np.set_printoptions(suppress=True)
print(Sigma_sim)
