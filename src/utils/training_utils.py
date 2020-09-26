import gpflow
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot(X_test, y_test, y_pred, title, ker_str):
    ind = np.argsort(X_test[:, 0], )
    plt.plot(X_test[ind, 0], y_test[ind])
    line, = plt.plot(X_test[ind, 0], y_pred[0][ind])
    # plt.plot(m.feature.Z.value, np.zeros(
    #     m.feature.Z.value.shape), 'k|', mew=2)
    # col = line.get_color()
    # plt.plot(X_test[ind, 0], y_pred[0][ind] +
    #          2 * y_pred[1][ind]**0.5, col, lw=1.5)
    # plt.plot(X_test[ind, 0], y_pred[0][ind] -
    #          2 * y_pred[1][ind]**0.5, col, lw=1.5)
    plt.title(ker_str + "  " + title)


def load_data(datafile):
    data = pd.read_csv(datafile)
    data = np.array(data)
    data = data.astype(float)
    dim = data.shape[1]
    X = data[:, 0:dim - 1]
    y = data[:, -1].reshape(-1, 1)
    return X, y


def data_processing(X, y, scale=1):
    max_abs_scaler = preprocessing.MaxAbsScaler()
    std_scale = preprocessing.StandardScaler()
    # TODO: make this better for multiple dim data
    if X.shape[1] == 1:
        X_scaled = max_abs_scaler.fit_transform(X) * scale
    else:
        X_scaled = std_scale.fit_transform(X)
    y_scaled = std_scale.fit_transform(y)
    return X_scaled, y_scaled


def get_inducing(X, global_size, local_size):
    # TODO: check if out of range
    X = np.sort(X)
    datasize = X.shape[0]
    itv = np.int(datasize / global_size)
    inducing_global = X[::itv, :].copy()
    inducing_local = X[round(datasize * 0.4):round(datasize * 0.4) + local_size].copy()
    inducing = np.concatenate((inducing_global, inducing_local))
    return inducing

