import pandas as pd

df = pd.read_csv("/Users/panzhouzhen/Documents/FinTech-545-Fall2025/testfiles/data/test6.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

returns = df / df.shift(1) - 1
returns = returns.dropna()
returns = returns.reset_index()

print(returns)
