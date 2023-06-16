import numpy as np
import pandas as pd


# def read_etf_file(etf):
#     filename = etf + '.csv'
#     df = pd.read_csv(filename, index_col=0)
#     df.index = pd.to_datetime(df.index)
#     return df
#
#
# def get_etf_returns(etf, return_type="log"):
#     df = read_etf_file(etf)
#     df = df[['Adj Close']]
#     if return_type == "simple":
#         df["returns"] = df["Adj Close"]/df["Adj Close"].shift(1)
#     if return_type == "log":
#         df['returns'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
#     df = df[['returns']]
#     df.columns = [etf]
#     return df
#
#
# def get_joined_returns(d_weights, from_date=None, to_date=None):
#     l_df = []
#     for etf, value in d_weights.items():
#         df_temp = get_etf_returns(etf, return_type='simple')
#         l_df.append(df_temp)
#     df_joined = pd.concat(l_df, axis=1)
#     df_joined.sort_index(inplace=True)
#     df_joined.dropna(inplace=True)
#     fromdate = pd.to_datetime(from_date)
#     todate = pd.to_datetime(to_date)
#     filtered_df = df_joined.loc[fromdate:todate]
#     return filtered_df
#
#
# def get_portfolio_returns(d_weights, from_date, to_date):
#     df_joined = get_joined_returns(d_weights)
#     df_weighted_returns = df_joined * pd.Series(d_weights)
#     s_portfolio_return = df_weighted_returns.sum(axis=1)
#     df_portfolio = pd.DataFrame(s_portfolio_return, columns=['pf'])
#     fromdate = pd.to_datetime(from_date)
#     todate = pd.to_datetime(to_date)
#     filtered_df = df_portfolio.loc[fromdate:todate]
#     return filtered_df
#
#
# def subtract_trading_date(actual_date, x):
#     date = pd.to_datetime(actual_date)
#     date_range = pd.bdate_range(end=date, periods=x+1)
#     result = date_range[0]
#     result_str = result.strftime('%Y-%m-%d')
#     return result_str
#
#
# def calc_historical_var(d_weights, conf_level, last_day_of_interval, window_in_days):
#     from_date = subtract_trading_date(last_day_of_interval, window_in_days)
#     df_ret = get_portfolio_returns(d_weights, from_date, last_day_of_interval)
#     quantile = 1-conf_level
#     df_result_ret = df_ret.quantile(quantile)
#     return df_result_ret.loc["pf"]
#
#
# def find_best_var(etf1, etf2, conf_level, last_day_of_interval, window_in_days):
#     d_weights = {etf1: np.arange(0, 1, 0.01), etf2: np.arange(1, 0, -0.01)}
#     df = pd.DataFrame(d_weights)
#     df.index = range(len(df))
#     l_vars = []
#     best_var = np.inf
#     best_row = -1
#     for i in range(len(df)):
#         d_weights = df.loc[i].to_dict()
#         var = calc_historical_var(d_weights, conf_level, last_day_of_interval, window_in_days)
#         if var < best_var:
#             best_var = var
#             best_row = i
#         l_vars.append(var)
#     df["var"] = l_vars
#     df.plot(x=etf1, y="var")
#     return df.loc[best_row].to_dict()

# ETF beolvasása
def read_etf_file(etf):
    filename = f'{etf}.csv'
    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


# Hozamok kiszámítása
def calc_etf_returns(etf):
    df = read_etf_file(etf)
    df = df[['Adj Close']]
    df["returns"] = df["Adj Close"]/df["Adj Close"].shift(1) - 1
    df = df[['returns']]
    df.columns = [etf]
    return df


# Az ETF-ek hozamainak összeillesztése egy dataframe-be
def calc_joined_returns(d_weights):
    l_df = []
    for etf, value in d_weights.items():
        df_temp = calc_etf_returns(etf)
        l_df.append(df_temp)
    df_joined = pd.concat(l_df, axis=1)
    df_joined.sort_index(inplace=True)
    df_joined.dropna(inplace=True)
    # fromdate = pd.to_datetime(from_date)
    # todate = pd.to_datetime(to_date)
    # filtered_df = df_joined.loc[fromdate:todate]
    return df_joined


# ETF-ekből képzett portfólió hozamainak kiszámítása
def calc_portfolio_returns(d_weights):
    df_joined = calc_joined_returns(d_weights)
    df_weighted_returns = df_joined * pd.Series(d_weights)
    s_portfolio_return = df_weighted_returns.sum(axis=1)
    df_portfolio = pd.DataFrame(s_portfolio_return, columns=['pf'])
    # fromdate = pd.to_datetime(from_date)
    # todate = pd.to_datetime(to_date)
    # filtered_df = df_portfolio.loc[fromdate:todate]
    return df_portfolio


# Historikus VaR kiszámítása
def calc_historical_var(df_returns, conf_level):
    # from_date = subtract_trading_date(last_day_of_interval, window_in_days)
    # df_ret = get_portfolio_returns(d_weights)
    quantile = 1-conf_level
    df_result_ret = df_returns.quantile(quantile)
    return float(df_result_ret.iloc[0])


# Azon súly megtalálása, amely mellett a historikus VaR értéke a legjobb
def find_best_var(etf1, etf2, conf_level):
    d_weights = {etf1: np.arange(0, 1, 0.01), etf2: np.arange(1, 0, -0.01)}
    df = pd.DataFrame(d_weights)
    df.index = range(len(df))
    l_vars = []
    best_var = -np.inf
    best_row = -1
    for i in range(len(df)):
        d_weights = df.loc[i].to_dict()
        df_returns = calc_portfolio_returns(d_weights)
        var = calc_historical_var(df_returns, conf_level)
        if var > best_var:
            best_var = var
            best_row = i
        l_vars.append(var)
    df["var"] = l_vars
    df.plot(x=etf1, y="var")
    return df.loc[best_row].to_dict()


if __name__ == '__main__':
    # Ellenőrzés
    df_returns = pd.DataFrame({'returns': np.arange(-0.05, 0.06, 0.01)})
    print(calc_historical_var(df_returns, 0.95))

    df1 = calc_portfolio_returns({'VOO': 0.5, 'MOO': 0.5})
    print(calc_historical_var(df1, 0.95))

    df2 = calc_portfolio_returns({'VOO': 0.3, 'MOO': 0.7})
    print(calc_historical_var(df2, 0.95))

    df2 = calc_portfolio_returns({'VOO': 0.7, 'MOO': 0.3})
    print(calc_historical_var(df2, 0.95))

    print(find_best_var('VOO', 'MOO', 0.95))
