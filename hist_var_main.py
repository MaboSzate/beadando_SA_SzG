import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ETF beolvasása
def read_etf_file(etf):
    filename = f'{etf}.csv'
    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


# Hozamok kiszámítása, type értéke alapján effektív vagy loghozam
def calc_etf_returns(etf, type):
    df = read_etf_file(etf)
    df = df[['Adj Close']]
    if type == 'simple':
        df["returns"] = df["Adj Close"]/df["Adj Close"].shift(1) - 1
    if type == 'log':
        df["returns"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    df = df[['returns']]
    df.columns = [etf]
    return df


# Az ETF-ek hozamainak összeillesztése egy dataframe-be
def calc_joined_returns(d_weights):
    l_df = []
    for etf, value in d_weights.items():
        df_temp = calc_etf_returns(etf, 'simple')
        l_df.append(df_temp)
    df_joined = pd.concat(l_df, axis=1)
    df_joined.sort_index(inplace=True)
    df_joined.dropna(inplace=True)
    return df_joined


# ETF-ekből képzett portfólió hozamainak kiszámítása
def calc_portfolio_returns(d_weights):
    df_joined = calc_joined_returns(d_weights)
    df_weighted_returns = df_joined * pd.Series(d_weights)
    s_portfolio_return = df_weighted_returns.sum(axis=1)
    df_portfolio = pd.DataFrame(s_portfolio_return, columns=['pf'])
    return df_portfolio


# Historikus VaR kiszámítása
# A hozamok megfelelő kvantilisához tartozó értéket határozzuk meg,
# ez lesz a VaR
def calculate_historical_var(df_portfolio_returns, alpha):
    quantile = 1 - alpha
    df_result_ret = df_portfolio_returns.quantile(quantile)
    return float(df_result_ret.iloc[0])


# Azon súly megtalálása, amely mellett a historikus VaR értéke a legjobb, és
# VaR értékek plotolása a súlyozás függvényében
def find_best_var(etf1, etf2, conf_level):
    d_weights = {etf1: np.arange(0, 1.01, 0.01),
                 etf2: np.arange(1, -0.01, -0.01)}
    df = pd.DataFrame(d_weights)
    df.index = range(len(df))
    l_vars = []
    best_var = -np.inf
    best_row = -1
    for i in range(len(df)):
        d_weights = df.loc[i].to_dict()
        df_returns = calc_portfolio_returns(d_weights)
        var = calculate_historical_var(df_returns, conf_level)
        if var > best_var:
            best_var = var
            best_row = i
        l_vars.append(var)
    df["var"] = l_vars
    df.plot(x=etf1, y="var")
    values = df.loc[best_row].to_dict()
    best_weight_1, best_weight_2, best_var_value = \
        values[etf1], values[etf2], values['var']
    print(f'Best weights: {etf1}: {round(best_weight_1, 5)}, '
          f'{etf2}: {round(best_weight_2, 5)}.')
    print(f'Best VaR value: {round(best_var_value, 5)}.')
    return values


if __name__ == '__main__':
    # Ellenőrzés
    df_returns = pd.DataFrame({'returns': np.arange(-0.05, 0.06, 0.01)})
    check = calculate_historical_var(df_returns, 0.95)
    print(check)

    # Legjobb VaR érték meghatározása és plotolás a súlyok függvényében
    find_best_var('VOO', 'MOO', 0.95)
    plt.show()
