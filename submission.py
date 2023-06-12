import numpy as np
import pandas as pd


def calculate_historical_var(df_portfolio_returns, alpha):
    quantile = 1-alpha
    df_result = df_portfolio_returns.quantile(quantile)
    df_result.index = [0]
    return df_result[0]


def calculate_ewma_variance(df_etf_returns, decay_factor, window):
    weights = decay_factor**np.arange(window)
    weights /= weights.sum()
    df = df_etf_returns.copy()
    df["Log Returns"] = np.log(df_etf_returns)
    df["Log Returns Sqrd"] = df["Log Returns"] ** 2
    for i in range(1, window+1):
        df[f"Log Returns Sqrd_lag_{i}"] = df["Log Returns Sqrd"].shift(i)
    relevant_cols = [f'Log Returns Sqrd_lag_{i}' for i in range(1, window + 1)]
    df_subset = df[relevant_cols]
    df[f'volatility_forecast_{decay_factor}'] = np.sqrt(np.dot(df_subset, weights))
    return df[f'volatility_forecast_{decay_factor}']

