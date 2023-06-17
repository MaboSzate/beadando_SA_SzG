import numpy as np
import pandas as pd
import simulated_var as sv


def calculate_historical_var(df_portfolio_returns, alpha):
    quantile = 1 - alpha
    df_result_ret = df_portfolio_returns.quantile(quantile)
    return float(df_result_ret.iloc[0])


def simulated_returns(expected_return, volatility, correlation, numOfSim):
    pf_expected_return = sv.calc_portfolio_expected_return(expected_return)
    pf_std_dev = sv.calc_portfolio_std_dev(volatility, correlation)
    sim_returns = np.random.normal(pf_expected_return, pf_std_dev, numOfSim)
    return sim_returns


def calculate_ewma_variance(df_etf_returns, decay_factor, window):
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
