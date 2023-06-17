import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ETF loghozamainak meghatározása
def calc_etf_logreturns(etf):
    df = pd.read_csv(f'{etf}.csv')
    df = df.set_index("Date")
    df["log_returns"] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df.dropna(inplace=True)
    return df[['log_returns']]


# EWMA kiszámítása
def calculate_ewma_variance(df_etf_returns, decay_factor, window):
    # Hozamok meghatározása
    weights = decay_factor ** np.arange(window)
    weights /= weights.sum()
    df = df_etf_returns.copy()
    # Négyzetes hozamok
    df['squared_log_returns'] = df['log_returns'] ** 2
    # Késleltetett négyzetes hozamokat tartalmazó mátrix
    for i in range(1, window+1):
        df[f'squared_log_return_lag_{i}'] = df['squared_log_returns'].shift(i)
    relevant_cols = \
        [f'squared_log_return_lag_{i}' for i in range(1, window + 1)]
    df_subset = df[relevant_cols]
    # Volatiliás előrejelzése
    df[f'volatility_forecast_{decay_factor}'] = \
        np.sqrt(np.dot(df_subset, weights))
    # Hiányzó adatokat tartalmazó részek törlése
    df.dropna(inplace=True)
    # Kívánt oszlop kivétele
    df_volatility_forecast = \
        df[[f'volatility_forecast_{decay_factor}']]
    return df_volatility_forecast


# Eredmény ábrázolása
def plot_ewma(etf):
    df = calc_etf_logreturns(etf)
    ewma_94 = calculate_ewma_variance(df, 0.94, 100)
    ewma_97 = calculate_ewma_variance(df, 0.97, 100)
    ewma = ewma_94.merge(ewma_97, left_on='Date', right_on='Date')
    ewma.plot()
    plt.title(etf)
    plt.legend(['decay_factor = 0.94', 'decay_factor = 0.97'])
    plt.xticks(rotation=-20)
    plt.show()


if __name__ == '__main__':
    # Plotolás
    plot_ewma('VOO')
    plot_ewma('MOO')

