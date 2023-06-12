import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import submission as s


class Test(unittest.TestCase):
    def test_hist_var(self):
        df_returns = pd.DataFrame({'returns': np.arange(-0.05, 0.06, 0.01)})
        var = s.calculate_historical_var(df_returns, 0.95)
        self.assertEqual(var, -0.045)


def test_ewma():
    filename = "MOO.csv"
    decay_factor = 0.94
    window = 100
    df = pd.read_csv(filename)
    df = df.set_index("Date")
    df["Returns"] = df['Adj Close']/df['Adj Close'].shift(1)
    df = df["Returns"]
    return s.calculate_ewma_variance(df, decay_factor, window)


test_ewma().plot()
plt.show()