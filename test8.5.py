import pandas as pd
from scipy.stats import t

alpha = 0.05

df = pd.read_csv("/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test7_2.csv")
x = df.iloc[:, 0].to_numpy()

# Fit Students-t dist
nu, mu, s = t.fit(x)

# Left-tail quantile
q = t.ppf(alpha, df=nu, loc=mu, scale=s)

# standardization
a = (q - mu) / s

# ES for Student-t
# E[T | T <= a] = -((nu + a^2)/(nu - 1)) * (f(a)/F(a)), derived from the distribution 
# formula and the general ES formula
pdf_a = t.pdf(a, df=nu)
cdf_a = t.cdf(a, df=nu)

mean_std = -((nu + a * a) / (nu - 1.0)) * (pdf_a / cdf_a)
mean = mu + s * mean_std

ES_absolute = -mean
ES_diff_from_mean = ES_absolute + mu


print("ES Absolute,ES Diff from Mean")
print(f"{ES_absolute:.10f},{ES_diff_from_mean:.10f}")
