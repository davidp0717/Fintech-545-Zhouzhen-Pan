import pandas as pd
from scipy.stats import norm

alpha = 0.05

df = pd.read_csv("/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test7_1.csv")
x = df.iloc[:, 0].to_numpy()

# statistics from the sample
mu = x.mean()
sigma = x.std(ddof=1)

# ES
z = norm.ppf(alpha)
phi = norm.pdf(z)

ES_absolute = -mu + sigma * (phi / alpha)
ES_diff_from_mean = sigma * (phi / alpha)


print("ES Absolute, ES Diff from Mean")
print(ES_absolute, ES_diff_from_mean)
