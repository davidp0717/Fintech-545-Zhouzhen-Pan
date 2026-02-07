import pandas as pd

df = pd.read_csv("/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test1.csv")

corr_matrix = df.corr()
print(corr_matrix)