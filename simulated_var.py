import numpy as np
import pandas as pd
import hist_var_main as hv


# A loghozamok normális eloszlásúak, ezért ezekből számolunk várható értéket
# és szórást
def calc_mean_and_std_dev(etf):
    df_return = hv.calc_etf_returns(etf)
    return float(df_return.mean()), float(df_return.std())


# A súlyok fordítottan arányosak a volatilitással
def calc_weights(etf_list):
    reciprocal_std_dev_dict = {}
    sum_of_reciprocal_std_dev = 0
    for etf in etf_list:
        mean, std_dev = calc_mean_and_std_dev(etf)
        reciprocal_std_dev_dict[etf] = 1 / std_dev
        sum_of_reciprocal_std_dev += 1/std_dev
    weights = [reciprocal_std_dev_dict[etf]/sum_of_reciprocal_std_dev
               for etf in reciprocal_std_dev_dict.keys()]
    return weights


# x1 és x2 független normálisak a megfelelő várható értékkel és szórással,
# y1 és y2 a belőlük képzett korrelált változók
def simulated_returns(expected_return, volatility, correlation, numOfSim):
    pass


print(calc_mean_and_std_dev('VOO'))
print(calc_mean_and_std_dev('MOO'))
print(calc_weights(['VOO', 'MOO']))
