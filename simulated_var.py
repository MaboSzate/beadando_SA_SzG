import numpy as np
import pandas as pd
import hist_var_main as hv
from scipy.stats import norm
import matplotlib.pyplot as plt


# Hozam és volatilitás meghatározása
# A loghozamok normális eloszlásúak, ezért ezekből számolunk várható értéket
# és szórást
def calc_mean_and_std_dev(etf, type):
    df_return = hv.calc_etf_returns(etf, type)
    return float(df_return.mean()), float(df_return.std())


# A súlyok meghatározása, amik fordítottan arányosak a volatilitással
def calc_weights(etf_list):
    reciprocal_std_dev_dict = {}
    sum_of_reciprocal_std_dev = 0
    for etf in etf_list:
        mean, std_dev = calc_mean_and_std_dev(etf, 'log')
        reciprocal_std_dev_dict[etf] = 1 / std_dev
        sum_of_reciprocal_std_dev += 1/std_dev
    weights = [reciprocal_std_dev_dict[etf]/sum_of_reciprocal_std_dev
               for etf in reciprocal_std_dev_dict.keys()]
    return weights


# Portfólió várható hozamának meghatározása
def calc_portfolio_expected_return(expected_return):
    weights = np.array(calc_weights(['VOO', 'MOO']))
    pf_expected_return = np.dot(weights, expected_return)
    return pf_expected_return


# Portfólió szórásának meghatározása feltételezett korreláció mellett
# A volatilitásokból meghatározzuk a C kovarianciamátrixot, és innen w
# súlyvektor mellett a szórás w'Cw gyöke
def calc_portfolio_std_dev(volatility, correlation):
    weights = np.array(calc_weights(['VOO', 'MOO']))
    vol_1 = volatility[0]
    vol_2 = volatility[1]
    cov = correlation * vol_1 * vol_2
    cov_matrix = np.array([[vol_1 ** 2, cov], [cov, vol_2 ** 2]])
    pf_variance = np.dot(weights, np.dot(cov_matrix, weights))
    return np.sqrt(pf_variance)


# x1 és x2 független normálisak a megfelelő várható értékkel és szórással,
# y1 és y2 a belőlük képzett korrelált változók
# A szimulált hozamok loghozamok
def simulated_returns(expected_return, volatility, correlation, numOfSim):
    pf_expected_return = calc_portfolio_expected_return(expected_return)
    pf_std_dev = calc_portfolio_std_dev(volatility, correlation)
    sim_returns = np.random.normal(pf_expected_return, pf_std_dev, numOfSim)
    return sim_returns


# Kovarianciamátrixon alapuló VaR meghatározása
# Kiszámoljuk a portfólió várható hozamát és szórását, ezek alapján az értékek
# alapján pedig kiszámoljuk a megfelelő normális eloszlás adott kvantilisát
def calc_covar_var_for_simulated_returns(expected_return, volatility,
                                         correlation, numOfSim, conf_level):
    sim_returns_log = simulated_returns(expected_return, volatility,
                                        correlation, numOfSim)
    # Átkonvertáljuk effektív hozammá
    sim_returns = np.exp(sim_returns_log) - 1
    sim_returns_mean = sim_returns.mean()
    sim_returns_std = sim_returns.std()
    quantile = 1 - conf_level
    var = norm.ppf(quantile, loc=sim_returns_mean, scale=sim_returns_std)
    return var


# VaR érték kirajzolása különböző korrelációk mellett
def plot_var_to_correlation(expected_return, volatility, numOfSim, conf_level):
    l_correlations = np.arange(-1, 1.01, 0.01)
    l_var = [calc_covar_var_for_simulated_returns(
        expected_return, volatility, correlation, numOfSim, conf_level)
             for correlation in l_correlations]
    df_var = pd.DataFrame(l_var, index=l_correlations, columns=['var'])
    df_var.plot(xlabel='Correlation',
                title='VaR value to the correlation of VOO and MOO')
    plt.show()


# Értékek meghatározása és plotolás
if __name__ == '__main__':
    r_VOO, std_dev_VOO = calc_mean_and_std_dev('VOO', 'log')
    r_MOO, std_dev_MOO = calc_mean_and_std_dev('MOO', 'log')

    expected_return = [r_VOO, r_MOO]
    volatility = [std_dev_VOO, std_dev_MOO]

    plot_var_to_correlation(expected_return, volatility, 5_000_000, 0.95)
