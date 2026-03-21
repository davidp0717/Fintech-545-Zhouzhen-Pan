import pandas as pd
import numpy as np
from scipy.stats import norm

df = pd.read_csv('testfiles/data/test12_1.csv').dropna()

results = []

for i, row in df.iterrows():

    ID   = int(row['ID'])
    call = row['Option Type'].strip() == 'Call'
    S    = row['Underlying']
    X    = row['Strike']
    rf   = row['RiskFreeRate']
    q    = row['DividendRate']
    ivol = row['ImpliedVol']

    ttm = row['DaysToMaturity'] / row['DayPerYear']   # time to maturity in years
    b   = rf - q                                       # cost of carry

    #d1 and d2
    d1 = (np.log(S / X) + (b + ivol**2 / 2) * ttm) / (ivol * np.sqrt(ttm))
    d2 = d1 - ivol * np.sqrt(ttm)

    #value
    if call:
        value = S * np.exp((b - rf) * ttm) * norm.cdf(d1) - X * np.exp(-rf * ttm) * norm.cdf(d2)
    else:
        value = X * np.exp(-rf * ttm) * norm.cdf(-d2) - S * np.exp((b - rf) * ttm) * norm.cdf(-d1)

    #delta
    if call:
        delta = np.exp((b - rf) * ttm) * norm.cdf(d1)
    else:
        delta = np.exp((b - rf) * ttm) * (norm.cdf(d1) - 1)

    #gamma
    gamma = norm.pdf(d1) * np.exp((b - rf) * ttm) / (S * ivol * np.sqrt(ttm))

    #vega
    vega = S * np.exp((b - rf) * ttm) * norm.pdf(d1) * np.sqrt(ttm)

    #theta
    if call:
        theta = (- S * np.exp((b - rf) * ttm) * norm.pdf(d1) * ivol / (2 * np.sqrt(ttm))
                 - (b - rf) * S * np.exp((b - rf) * ttm) * norm.cdf(d1)
                 - rf * X * np.exp(-rf * ttm) * norm.cdf(d2))
    else:
        theta = (- S * np.exp((b - rf) * ttm) * norm.pdf(d1) * ivol / (2 * np.sqrt(ttm))
                 + (b - rf) * S * np.exp((b - rf) * ttm) * norm.cdf(-d1)
                 + rf * X * np.exp(-rf * ttm) * norm.cdf(-d2))

    #rho
    if call:
        rho = ttm * X * np.exp(-rf * ttm) * norm.cdf(d2)
    else:
        rho = -ttm * X * np.exp(-rf * ttm) * norm.cdf(-d2)

    results.append({'ID': ID, 'Value': value, 'Delta': delta,
                    'Gamma': gamma, 'Vega': vega, 'Rho': rho, 'Theta': theta})

df_result = pd.DataFrame(results)
print(df_result.to_string(index=False))