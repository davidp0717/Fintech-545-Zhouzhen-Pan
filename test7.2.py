import pandas as pd
from scipy.stats import t

def fit_t_mle():
    df = pd.read_csv("/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test7_2.csv")
    x = df["x1"].to_numpy(dtype=float)

    nu, mu, s = t.fit(x)

    print("mu =", float(mu))
    print("s  =", float(s))
    print("nu =", float(nu))

fit_t_mle()
