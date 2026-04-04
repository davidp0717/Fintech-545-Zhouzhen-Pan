import numpy as np
import pandas as pd


returns_df = pd.read_csv("testfiles/data/test11_1_returns.csv")
weights_df = pd.read_csv("testfiles/data/test11_1_weights.csv")

stocks = list(returns_df.columns)
w = weights_df["W"].to_numpy(dtype=float)
matReturns = returns_df[stocks].to_numpy(dtype=float)


# calculate return
n_days = matReturns.shape[0]
n_stocks = matReturns.shape[1]

pReturn = np.zeros(n_days)
# weights evolution
weights_arr = np.zeros((n_days, n_stocks))

lastW = w.copy()


for i in range(n_days):
    weights_arr[i, :] = lastW

    # Update weights
    lastW = lastW * (1.0 + matReturns[i, :])

    # gross return
    pR = lastW.sum()

    # normalization
    lastW = lastW / pR

    pReturn[i] = pR - 1.0


# carino K
totalRet = np.exp(np.sum(np.log(1.0 + pReturn))) - 1.0
K = np.log(1.0 + totalRet) / totalRet
carinoK = np.log(1.0 + pReturn) / pReturn / K

attrib_matrix = matReturns * weights_arr * carinoK[:, None]
attrib_df = pd.DataFrame(attrib_matrix, columns=stocks)


# return attribution
rows = {"Value": ["TotalReturn", "Return Attribution"]}

for s in stocks:
    series = returns_df[s].to_numpy(dtype=float)
    tr = np.exp(np.sum(np.log(1.0 + series))) - 1.0
    atr = attrib_df[s].sum()
    rows[s] = [tr, atr]

rows["Portfolio"] = [totalRet, totalRet]

Attribution = pd.DataFrame(rows)

# RV attribution

Y_vol = matReturns * weights_arr
X_vol = np.column_stack([np.ones(n_days), pReturn])

B_vol = np.linalg.inv(X_vol.T @ X_vol) @ X_vol.T @ Y_vol
betas = B_vol[1, :]

port_std = np.std(pReturn, ddof=1)
cSD = betas * port_std

vol_row = {"Value": "Vol Attribution"}
for i, s in enumerate(stocks):
    vol_row[s] = cSD[i]
vol_row["Portfolio"] = port_std

Attribution = pd.concat([Attribution, pd.DataFrame([vol_row])], ignore_index=True)


print("\nAttribution Table:")
print(Attribution.to_string(index=False))