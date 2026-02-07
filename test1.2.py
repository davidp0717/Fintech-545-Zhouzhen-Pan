import pandas as pd

df = pd.read_csv("/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test1.csv")

df_clean = df.dropna(axis=0, how="any")

corr_matrix = df_clean.corr()

print(corr_matrix)
