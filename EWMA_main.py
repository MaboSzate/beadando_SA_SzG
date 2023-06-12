import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def calc_ewma_weights(decay, window):
    weights=decay**np.arange(window)
    return weights/weights.sum()


def implement_ewma(filename, decay1, decay2, window):
    df=pd.read_csv(filename)
    df=df.set_index("Date")
    df["Log Returns"]=np.log(df['Adj Close']/df['Adj Close'].shift(1))
    df["Log Returns Sqrd"]=df["Log Returns"]**2
    for i in range(1, window+1):
        df[f"Log Returns Sqrd_lag_{i}"] = df["Log Returns Sqrd"].shift(i)
    wts1=calc_ewma_weights(decay1,window)
    relevant_cols = [f'Log Returns Sqrd_lag_{i}' for i in range(1, window + 1)]
    df_subset = df[relevant_cols]
    df[f'volatility_forecast_{decay1}'] = np.sqrt(np.dot(df_subset, wts1))
    df[f'volatility_forecast_{decay1}'].plot(label="decay="+str(decay1))
    wts2 = calc_ewma_weights(decay2, window)
    df[f'volatility_forecast_{decay2}'] = np.sqrt(np.dot(df_subset, wts2))
    df[f'volatility_forecast_{decay2}'].plot(label="decay="+str(decay2))
    plt.legend()
    plt.show()


#implement_ewma("MOO.csv", 0.94,0.97,100)

