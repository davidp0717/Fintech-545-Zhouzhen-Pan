import numpy as np
import pandas as pd

stock_df = pd.read_csv("testfiles/data/test11_2_stock_returns.csv")
factor_df = pd.read_csv("testfiles/data/test11_2_factor_returns.csv")
beta_df = pd.read_csv("testfiles/data/test11_2_beta.csv")
weights_df = pd.read_csv("testfiles/data/test11_2_weights.csv")

stock_names = stock_df.columns.tolist()     
factor_names = factor_df.columns.tolist() 

stockReturns = stock_df.values.astype(float)
ffReturns = factor_df.values.astype(float)
w = weights_df["W"].values.astype(float)


beta_df = beta_df.set_index("Stock").loc[stock_names, factor_names]
betas = beta_df.values.astype(float).T 

n_days, n_stocks = stockReturns.shape
n_factors = ffReturns.shape[1]


pReturn = np.zeros(n_days)
residReturn = np.zeros(n_days)
weights_arr = np.zeros((n_days, n_stocks))
factorWeights_arr = np.zeros((n_days, n_factors))
lastW = w.copy()

for i in range(n_days):
    weights_arr[i, :] = lastW

    # β * w
    factorWeights_arr[i, :] = betas @ lastW

    # evolve weights
    lastW = lastW * (1.0 + stockReturns[i, :])
    pR = lastW.sum()
    lastW = lastW / pR

    pReturn[i] = pR - 1.0

    residReturn[i] = pReturn[i] - factorWeights_arr[i, :] @ ffReturns[i, :]

# carino K
totalRet = np.exp(np.sum(np.log(1.0 + pReturn))) - 1.0
K = np.log(1.0 + totalRet) / totalRet
carinoK = np.log(1.0 + pReturn) / pReturn / K


#return attribution
attrib_matrix = ffReturns * factorWeights_arr * carinoK[:, None]
alpha_attrib = residReturn * carinoK

attrib_df = pd.DataFrame(attrib_matrix)
attrib_df["Alpha"] = alpha_attrib


#table 
factor_names = [f"F{i+1}" for i in range(n_factors)] + ["Alpha"]

rows = {"Value": ["TotalReturn", "Return Attribution"]}

# factor returns
for j, name in enumerate(factor_names[:-1]):
    series = ffReturns[:, j]
    tr = np.exp(np.sum(np.log(1.0 + series))) - 1.0
    atr = attrib_df.iloc[:, j].sum()
    rows[name] = [tr, atr]

# Alpha
tr_alpha = np.exp(np.sum(np.log(1.0 + residReturn))) - 1.0
atr_alpha = attrib_df["Alpha"].sum()
rows["Alpha"] = [tr_alpha, atr_alpha]


rows["Portfolio"] = [totalRet, totalRet]

Attribution = pd.DataFrame(rows)

# volatility attribution
Y_vol = np.column_stack([ffReturns * factorWeights_arr, residReturn])
X_vol = np.column_stack([np.ones(n_days), pReturn])

B = np.linalg.inv(X_vol.T @ X_vol) @ X_vol.T @ Y_vol
betas_vol = B[1, :]

port_std = np.std(pReturn, ddof=1)
cSD = betas_vol * port_std

vol_row = {"Value": "Vol Attribution"}
for i, name in enumerate(factor_names):
    vol_row[name] = cSD[i]
vol_row["Portfolio"] = port_std

Attribution = pd.concat([Attribution, pd.DataFrame([vol_row])], ignore_index=True)

print("\nFactor Attribution Table:")
print(Attribution.to_string(index=False))