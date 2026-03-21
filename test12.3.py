import pandas as pd
import numpy as np

df = pd.read_csv('testfiles/data/test12_3.csv')

N = 500
results = []

for i, row in df.iterrows():

    ID   = int(row['ID'])
    call = row['Option Type'].strip() == 'Call'
    S    = row['Underlying']
    X    = row['Strike']
    rf   = row['RiskFreeRate']
    ivol = row['ImpliedVol']
    dpy  = row['DayPerYear']
    ttm  = row['DaysToMaturity'] / dpy

    # discrete dividends
    div_times = [int(x.strip()) for x in str(row['DividendDates']).split(',') if x.strip() != '']
    div_amts  = [float(x.strip()) for x in str(row['DividendAmts']).split(',') if x.strip() != '']

    # subtract PV of discrete dividends from spot, then use standard American tree for simplicity
    pv_div = 0.0
    for amt, t in zip(div_amts, div_times):
        pv_div += amt * np.exp(-rf * (t / dpy))

    S_adj = S - pv_div

    def price(call, S, X, ttm, rf, ivol, N):
        dt   = ttm / N
        u    = np.exp(ivol * np.sqrt(dt))
        d    = 1.0 / u
        pu   = (np.exp(rf * dt) - d) / (u - d)
        pd_  = 1.0 - pu
        disc = np.exp(-rf * dt)
        z    = 1 if call else -1

        def nN(n): 
            return (n + 1) * (n + 2) // 2

        def ix(i, j): 
            return nN(j - 1) + i

        opt = np.zeros(nN(N))

        for j in range(N, -1, -1):
            for i in range(j, -1, -1):
                p_node    = S * u**i * d**(j - i)
                intrinsic = max(0.0, z * (p_node - X))

                if j == N:
                    opt[ix(i, j)] = intrinsic
                else:
                    hold = disc * (pu * opt[ix(i + 1, j + 1)] + pd_ * opt[ix(i, j + 1)])
                    opt[ix(i, j)] = max(intrinsic, hold)

        return opt[ix(0, 0)]

    value = price(call, S_adj, X, ttm, rf, ivol, N)

    results.append({
        'ID': ID,
        'Value': value
    })

df_result = pd.DataFrame(results)
print(df_result.to_string(index=False))