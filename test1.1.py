import pandas as pd
import numpy as np

input_file = "test1.csv"
output_file = "my_output.csv"

df = pd.read_csv("/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test1.csv")

df_clean = df.dropna(axis=0, how="any")

cov_matrix = df_clean.cov()
print(cov_matrix)
