import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import submission as s


def test_ewma(filename, decay_factor, window):
    df = pd.read_csv(filename)
    df = df.set_index("Date")
    df["Returns"] = df['Adj Close']/df['Adj Close'].shift(1)
    df = df["Returns"]
    return s.calculate_ewma_variance(df, decay_factor, window)


ewma_output = test_ewma("MOO.csv", 0.94, 100)
ewma_output.plot()
plt.show()
