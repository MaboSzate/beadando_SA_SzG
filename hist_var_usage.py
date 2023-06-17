import hist_var_main as m
import matplotlib.pyplot as plt


def test_calc_historical_var():
    d_weights = {'VOO': 0.6, 'MOO': 0.4}
    conf_level = 0.95
    last_day_of_interval = '2020-03-01'
    window_in_days = 250
    var = m.calc_historical_var(d_weights, conf_level)
    return var


# print(test_calc_historical_var())


def test_find_best_var():
    etf1, etf2 = "VOO", "MOO"
    conf_level = 0.95
    last_day_of_interval = '2020-03-01'
    window_in_days = 250
    best = m.find_best_var(etf1, etf2, conf_level)
    plt.show()
    return best


print(test_find_best_var())

