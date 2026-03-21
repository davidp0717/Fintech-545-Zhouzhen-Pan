import pandas as pd
import numpy as np

df = pd.read_csv('testfiles/data/test12_1.csv').dropna()

N = 500

results = []

for i, row in df.iterrows():

    ID   = int(row['ID'])
    call = row['Option Type'].strip() == 'Call'
    S    = row['Underlying']
    X    = row['Strike']
    rf   = row['RiskFreeRate']
    q    = row['DividendRate']
    ivol = row['ImpliedVol']
    dpy  = row['DayPerYear']
    ttm  = row['DaysToMaturity'] / dpy
    b    = rf - q           

    # binomial tree
    def price(call, S, X, ttm, rf, b, ivol, N):
        dt   = ttm / N
        u    = np.exp(ivol * np.sqrt(dt))
        d    = 1.0 / u
        pu   = (np.exp(b * dt) - d) / (u - d)
        pd_  = 1.0 - pu
        disc = np.exp(-rf * dt)
        z    = 1 if call else -1

        # triangular array
        def nN(n): return (n + 1) * (n + 2) // 2
        def ix(i, j): return nN(j - 1) + i

        opt = np.zeros(nN(N))
        for j in range(N, -1, -1):
            for i in range(j, -1, -1):
                p_node    = S * u**i * d**(j - i)
                intrinsic = max(0.0, z * (p_node - X))
                if j == N:
                    opt[ix(i, j)] = intrinsic
                else:
                    hold = disc * (pu * opt[ix(i+1, j+1)] + pd_ * opt[ix(i, j+1)])
                    opt[ix(i, j)] = max(intrinsic, hold)
        return opt[ix(0, 0)]

    # value
    value = price(call, S, X, ttm, rf, b, ivol, N)

    # delta & gamma
    dS  = 1.5
    v_p = price(call, S + dS, X, ttm, rf, b, ivol, N)
    v_m = price(call, S - dS, X, ttm, rf, b, ivol, N)
    delta = (v_p - v_m) / (2 * dS)
    gamma = (v_p + v_m - 2 * value) / (dS ** 2)

    # vega
    dv  = 0.0001
    v_p = price(call, S, X, ttm, rf, b, ivol + dv, N)
    v_m = price(call, S, X, ttm, rf, b, ivol - dv, N)
    vega = (v_p - v_m) / (2 * dv)

    # rho
    dr  = 0.0001
    v_p = price(call, S, X, ttm, rf + dr, b, ivol, N)
    v_m = price(call, S, X, ttm, rf - dr, b, ivol, N)
    rho = (v_p - v_m) / (2 * dr)

    # theta
    dt  = 1.0 / dpy
    v_p = price(call, S, X, ttm + dt, rf, b, ivol, N)
    v_m = price(call, S, X, ttm - dt, rf, b, ivol, N)
    theta = (v_p - v_m) / (2 * dt)

    results.append({
        'ID': ID, 'Value': value, 'Delta': delta,
        'Gamma': gamma, 'Vega': vega, 'Rho': rho, 'Theta': theta
    })

df_result = pd.DataFrame(results)
print(df_result.to_string(index=False))