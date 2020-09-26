import numpy as np
from sklearn import preprocessing
import pandas as pd
import pickle
from DistanceBO import BO
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath('')))
from analyse import gather_training_res as gtr
from utils import parse_cfg as cfg


def data_processing(X, y):
    std_scale = preprocessing.StandardScaler()
    X_scaled = std_scale.fit_transform(X)
    y_scaled = std_scale.fit_transform(y)
    return X_scaled, y_scaled


def load_data(datafile):
    data = pd.read_csv(datafile)
    data = np.array(data)
    data = data.astype(float)
    dim = data.shape[1]
    X = data[:, 0:dim - 1]
    y = data[:, -1].reshape(-1, 1)
    return X, y


def get_model_evidence(ker_str):
    # string formed kernel
    ind = domain.index(ker_str)
    evidence = evidence_all[ind]
    # Don't forget negative
    return - evidence


def get_time(ker_str):
    ind = domain.index(ker_str)
    train_time = train_time_all[ind]
    return train_time


def pre_trained_res(elbo_file_name):
    results = pd.read_csv(elbo_file_name)
    domain = results['kernel'].values
    evidence_all = results['elbo'].values
    train_time_all = results['time'].values
    rmse_all = results['rmse'].values

    base_kernels = ['s', 'r', 'p', 'l']
    b_inds = []
    for b in base_kernels:
        try:
            b_inds.append(np.argwhere(domain == b)[0][0])
        except:
            pass
    b_inds.extend(list(set(np.arange(len(domain))) - set(b_inds)))
    domain = domain[b_inds].tolist()
    evidence_all = evidence_all[b_inds].tolist()
    train_time_all = train_time_all[b_inds].tolist()
    rmse_all = rmse_all[b_inds].tolist()
    return domain, evidence_all, train_time_all, rmse_all


def run_bo(datafile, subset_candi_size, subset_size, result_filename):
    np.random.seed(100)
    X_all, y_all = load_data(datafile)
    X = X_all[1:subset_candi_size, :]
    y = y_all[1:subset_candi_size, :]
    X, y = data_processing(X, y)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    x_sub = X[indices[1:subset_size + 1], :]
    y_sub = y[indices[1:subset_size + 1], :]

    bo = BO(get_model_evidence, domain, x_sub, y_sub, 50, initnum=4)

    bo.bo_run()
    obs_x = np.array(bo.observations)
    obs_y = - np.array(bo.mins)
    bo_time = np.array(bo.time_elapsed)
    obs_time = []
    for ker in obs_x:
        obs_time.append(get_time(ker))
    obs_time = np.array(obs_time)

    best_ker_ind = evidence_all.index(max(evidence_all))

    bo_results = {'obs': obs_x, 'obs_elbo': bo.Y, 'best_ker_seen': bo.record_best_ker, 'best_ker_seen_elbo': obs_y,
                  'obs_time': obs_time,
                  'bo_time': bo_time,
                  'best_ker': domain[best_ker_ind],
                  'best_ker_elbo': max(evidence_all)}

    with open(result_filename, 'wb') as fout:
        pickle.dump(bo_results, fout)


if __name__ == "__main__":
    cfg_file = '../../config/swiss_cfg_1.json'
    subset_candi_size = 200000

    # cfg_file = '../../config/air_cfg_1.json'
    # subset_candi_size = 10000

    subset_size = 200
    _, paths, wf = cfg.get_settings(cfg_file, 'full')
    wf = '../../results/' + wf + "/"
    elbo_file_name = wf + 'fulldata_summary.csv'

    if not os.path.exists(elbo_file_name):
        k = gtr.get_kernel_names(wf)
        df = gtr.get_fulldata_res(wf, k)
        df.to_csv(elbo_file_name)

    domain, evidence_all, train_time_all, _ = pre_trained_res(elbo_file_name)

    evidence_all = evidence_all
    datafile = '../' + paths['datafile']
    bo_res_file = wf + 'bo_subsize' + str(subset_size) + '.pkl'
    run_bo(datafile, subset_candi_size, subset_size, bo_res_file)

