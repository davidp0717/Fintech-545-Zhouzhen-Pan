import numpy as np
import pandas as pd

def normal_dist():
    df = pd.read_csv(
        "/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test7_1.csv"
    )

    x = df["x1"].to_numpy(dtype=float)

    mu_hat = np.sum(x) / len(x)
    sigma_2_hat = np.sum((x - mu_hat) ** 2) / (len(x) - 1)
    sigma_hat = np.sqrt(sigma_2_hat)

    return mu_hat, sigma_hat

mu, sigma = normal_dist()
print("mu =", mu)
print("sigma =", sigma)

