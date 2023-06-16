import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def calc_etf_logreturns(filename):
    df = pd.read_csv(filename)
    df = df.set_index("Date")
    df["log_returns"] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df.dropna(inplace=True)
    return df[['log_returns']]


def calculate_ewma_variance(df_etf_returns, decay_factor,  window):
    weights = decay_factor ** np.arange(window)
    weights /= weights.sum()
    df = df_etf_returns.copy()
    df['squared_log_returns'] = df['log_returns'] ** 2
    for i in range(1, window+1):
        df[f'squared_log_return_lag_{i}'] = df['squared_log_returns'].shift(i)
    relevant_cols = \
        [f'squared_log_return_lag_{i}' for i in range(1, window + 1)]
    df_subset = df[relevant_cols]
    df[f'volatility_forecast_{decay_factor}'] = \
        np.sqrt(np.dot(df_subset, weights))
    df.dropna(inplace=True)
    df_volatility_forecast = \
        df[[f'volatility_forecast_{decay_factor}']]
    return df_volatility_forecast


# Ellenőrzés
df1 = calc_etf_logreturns('VOO.csv')
volatility_forecast_94 = calculate_ewma_variance(df1, 0.94, 100)

df2 = calc_etf_logreturns('VOO.csv')
volatility_forecast_97 = calculate_ewma_variance(df1, 0.97, 100)

both = volatility_forecast_94.merge(volatility_forecast_97,
                                    left_on='Date', right_on='Date')
both.plot()
plt.show()
