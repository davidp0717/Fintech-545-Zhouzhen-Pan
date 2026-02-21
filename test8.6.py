import pandas as pd
import numpy as np
import math
from scipy.stats import t

alpha = 0.05
nsim = 100000
seed = 12345


df = pd.read_csv("/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test7_2.csv")
x = df.iloc[:, 0].to_numpy()

# Fit t-dist
nu, mu, s = t.fit(x)

# Simulate from fitted dist
np.random.seed(seed)
sim = t.rvs(df=nu, loc=mu, scale=s, size=nsim)

# ES
sorted = np.sort(sim)
n = len(sorted)

nup = int(math.ceil(n * alpha))
ndn = int(math.floor(n * alpha))

v = 0.5 * (sorted[nup - 1] + sorted[ndn - 1])
es = sorted[sorted <= v].mean()

ES_absolute = -es
ES_diff_from_mean = ES_absolute + sim.mean()


print("ES Absolute,ES Diff from Mean")
print(ES_absolute,ES_diff_from_mean)