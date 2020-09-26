"""
@Tong
04-07-2019
Bayesian model averaging
get p(k) and y_prediction, real_y_test
compute rmse and mnlp
plot the results
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import os
from analyse import ana_utils
import paths

"""
train_results_one_ker = {'elbo_loggers': elbo_loggers_diff_init,
                        'time_loggers':time_loggers_diff_init,
                        'elbos': elbos_estimate_diff_init,
                        'y_preds': y_preds_diff_init,
                        'rmses': rmses_diff_init,
                        'time_elp': time_elp_diff_runs}
"""


# compute BMA rmse and mnlp for a certain SUBSET, file_str can indicate which kernels are used
def subset_BMA(working_folder, file_str, subset_batchnum, size_sample, prior_pg,
               normalized=False):
    valid_ker, elbos, rmse_multi_ker, y_preds_multi_ker = ana_utils.get_valid_ker_and_elbos_and_rmses(working_folder,
                                                                                                      subset_batchnum)
    test_data_file = working_folder + "testdata" + ".pkl"
    with open(test_data_file, 'rb') as fin:
        test_data = pickle.load(fin)
    y_test = test_data['y_test']

    prob_result_file = ana_utils.naming(file_str, subset_batchnum, size_sample, prior_pg, normalized, 'prob')
    prob_result_file = working_folder + 'ana_res/' + prob_result_file
    prob_k = pd.read_csv(prob_result_file, header=None)
    BMA_result_file = ana_utils.naming(file_str, subset_batchnum, size_sample, prior_pg, normalized, 'bma')
    BMA_result_file = working_folder + 'ana_res/' + BMA_result_file

    multi_ker_str = prob_k[0]
    data = {'kernels': valid_ker,
            'L_i': elbos,
            'rmse': rmse_multi_ker,
            'y_pred': y_preds_multi_ker}
    df0 = pd.DataFrame(data)
    df0 = df0.sort_values(['L_i'], ascending=False)
    if multi_ker_str is not None:
        df0 = df0.loc[df0['kernels'].isin(list(set(valid_ker) & set(multi_ker_str)))]

    # df0 has been changed above
    valid_ker = df0['kernels'].values.tolist()
    y_preds_multi_ker = df0['y_pred'].values.tolist()
    rmse_multi_ker = df0['rmse'].values.tolist()
    elbos = df0['L_i'].values.tolist()

    rmse_bayes_avg, _, _ = get_bayes_avg_rmse(valid_ker, prob_k, y_preds_multi_ker, y_test)
    mnlp_multi_ker, mnlp_bayes_avg = get_mnlp(valid_ker, prob_k, y_preds_multi_ker, y_test)
    mnlp_multi_ker.append(mnlp_bayes_avg)
    valid_ker.append('BMA')
    rmse_multi_ker.append(rmse_bayes_avg)
    elbos.append(0)
    result = {"kernels": valid_ker, "rmse": rmse_multi_ker, "elbo": elbos, "mnlp": mnlp_multi_ker}
    df = pd.DataFrame(result)
    print(df)
    df.to_csv(BMA_result_file, index=None)


def get_bayes_avg_rmse(kernels, prob_k, y_preds_multi_ker, y_test):
    data_shape = (len(y_test), 1)
    y_bayes_pred_mean = np.zeros(data_shape)
    y_bayes_pred_var = np.zeros(data_shape)
    for i in range(len(kernels)):
        curr_pk = prob_k[prob_k[0] == kernels[i]].iloc[0, 1]
        curr_y_preds_mean = y_preds_multi_ker[i][0].reshape(data_shape)
        curr_y_preds_var = y_preds_multi_ker[i][1].reshape(data_shape)
        # Bayesian prediction
        y_bayes_pred_mean = y_bayes_pred_mean + curr_pk * curr_y_preds_mean
        y_bayes_pred_var = y_bayes_pred_var + (curr_y_preds_var + curr_y_preds_mean ** 2) * curr_pk
    y_bayes_pred_var = y_bayes_pred_var - y_bayes_pred_mean ** 2

    rmse_bayes_avg = sqrt(mean_squared_error(y_test, y_bayes_pred_mean))
    return rmse_bayes_avg, y_bayes_pred_mean, y_bayes_pred_var


def get_mnlp(kernels, prob_k, y_preds_multi_ker, y_test):
    data_shape = (len(y_test), 1)
    y_bayes_pred_mean = np.zeros(data_shape)
    y_bayes_pred_var = np.zeros(data_shape)
    mnlp_multi_ker = []
    for i in range(len(kernels)):
        curr_pk = prob_k[prob_k[0] == kernels[i]].iloc[0, 1]
        curr_y_preds_mean = y_preds_multi_ker[i][0].reshape(data_shape)
        curr_y_preds_var = y_preds_multi_ker[i][1].reshape(data_shape)
        mnlp_curr = np.mean(0.5 * np.log((2 * np.pi) * curr_y_preds_var) +
                            0.5 * (((curr_y_preds_mean - y_test) ** 2) / curr_y_preds_var))
        mnlp_multi_ker.append(mnlp_curr)
        # Bayesian prediction
        y_bayes_pred_mean = y_bayes_pred_mean + curr_pk * curr_y_preds_mean
        y_bayes_pred_var = y_bayes_pred_var + (curr_y_preds_var + curr_y_preds_mean ** 2) * curr_pk
    y_bayes_pred_var = y_bayes_pred_var - y_bayes_pred_mean ** 2
    mnlp_bayes_avg = np.mean(0.5 * np.log((2 * np.pi) * y_bayes_pred_var) +
                             0.5 * (((y_bayes_pred_mean - y_test) ** 2) / y_bayes_pred_var))
    # mnlp = mean(0.5 * log((2 * pi) * (variance)) + 0.5 * (((pred - ytest) ^ 2) / variance))
    return mnlp_multi_ker, mnlp_bayes_avg


# os.chdir('/home/tengtong/BKS/src/')
os.chdir(paths.cwd)
