import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def create_data(filename, window, split=False):
    df = pd.read_csv(filename)
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    df = df[df.index.year >= 2021]  # to shorten runtime
    df["Log Returns"] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
    df["Log Returns Sqrd"] = df["Log Returns"]**2
    cols = []
    for i in range(1, window + 1):
        col = f'lag_{i}'
        df[col] = df['Log Returns Sqrd'].shift(i)
        cols.append(col)
    df.dropna(inplace=True)
    X = np.array(df[cols])
    # Miért ez az y?
    y = np.array(df['Log Returns Sqrd'])
    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, X_test, y_train, y_test
    else:
        return X, y


def create_polynomial_model(degree=1):
    name = "Polinomial_" + str(degree)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    return name, model


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse


def create_train_and_evaluate_polynomial_model(X_train, X_test,
                                               y_train, y_test, degree=15):
    name, model = create_polynomial_model(degree)
    model.fit(X_train, y_train)
    coefficients_on_train_set = model.named_steps['linearregression'].coef_
    y_pred = model.predict(X_test)
    mse_on_test_set = mean_squared_error(y_test, y_pred)
    return name, model, mse_on_test_set, coefficients_on_train_set, y_pred


def hyperparameter_search(X_train, X_test, y_train, y_test,
                          from_degree=1, to_degree=10):
    degrees = range(from_degree, to_degree+1)
    best_degree, best_mse, best_model = None, float('inf'), None
    d_mse = {}
    for degree in degrees:
        name, model, mse_on_test, coefficients_on_train_set, y_pred = \
            create_train_and_evaluate_polynomial_model(X_train, X_test,
                                                       y_train, y_test, degree)
        d_mse[degree] = mse_on_test
        print(f'For degree: {degree}, MSE: {mse_on_test}')
        if mse_on_test < best_mse:
            best_degree, best_mse, best_model = degree, mse_on_test, model
    print(f'Best degree: {best_degree}, Best MSE {best_mse}')
    print_coeffs('Coefficients: ', best_model)
    return best_model


def print_coeffs(text, model):
    if 'linear_regression' in model.named_steps.keys():
        linreg = 'linear_regression'
    else:
        linreg = 'linearregression'
    coeffs = np.concatenate(([model.named_steps[linreg].intercept_],
                             model.named_steps[linreg].coef_[1:]))
    coeffs_str = ' '.join(np.format_float_positional(coeff, precision=4)
                          for coeff in coeffs)
    print(text + coeffs_str)


def cross_validate(X, y, n_splits=5, from_degree=1, to_degree=10):
    degrees = range(from_degree, to_degree+1)
    kf = KFold(n_splits=n_splits)
    results = {}
    best_model = None
    best_degree = None
    best_mse = np.inf
    np.set_printoptions(precision=4)
    for degree in degrees:
        name, model = create_polynomial_model(degree)
        mse_sum = 0
        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            model, mse = train_and_evaluate_model(model, X_train, y_train,
                                                  X_val, y_val)
            print_coeffs("Coefficients: ", model)
            mse_sum += mse
        avg_mse = mse_sum / n_splits
        results[degree] = avg_mse
        print(f"For degree: {degree}, MSE: {avg_mse}")
        # fit for the whole dataset
        # model, mse = train_and_evaluate_model(model, X, y, X_val, y_val)1
        model.fit(X, y)
        print_coeffs("Final Coefficients: ", model)
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_degree = degree
            best_model = model
    print(f"Best model: degree={best_degree}, MSE={best_mse}")
    print_coeffs("Coefficients for best model: ", best_model)
    return best_model


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = create_data("MOO.csv", 10, split=True)
    hyperparameter_search(X_train, X_test, y_train, y_test, to_degree=5)

    #X, y = create_data('MOO.csv', 10, split=False)
    #cross_validate(X, y, to_degree=5)
