import pickle
import os
import numpy as np
import pandas as pd
from analyse import compute_prob_use_r as cpr


# todo: better
def get_scaler_subset(working_folder, smallest_batchnum, maxvalue, multi_ker_str=None):
    if not multi_ker_str:
        _, elbos = get_valid_ker_and_elbos(working_folder, smallest_batchnum)
    else:
        elbos = get_elbos_subset(working_folder, smallest_batchnum, multi_ker_str)
    elbos = np.array(elbos)
    elbos = elbos / smallest_batchnum
    scaler = maxvalue / min(elbos)  # assume negative elbo
    return scaler


def get_scaler_fulldata(working_folder, start_loc, logger_range, maxvalue, multi_ker_str=None, run_id=None):
    if not multi_ker_str:
        _, elbos_logger_multi_ker, _ = get_valid_ker_and_best_elbos_logger(working_folder, run_id)
    else:
        elbos_logger_multi_ker, _ = get_logger_fulldata(working_folder, multi_ker_str, run_id)
    elbos = elbo_estimate_fulldata(start_loc, logger_range, elbos_logger_multi_ker)
    scaler = maxvalue / min(elbos)
    return scaler


#  subsets
def get_time(save_folder_parent, subset_batchnum, kernels):
    save_folder_child = save_folder_parent + 'train_res_' + str(subset_batchnum) + '/'
    time_multi_ker = []
    for ker_str in kernels:
        fname = save_folder_child + ker_str + ".pkl"
        with open(fname, "rb") as fin:
            ELBOs_curr_ker = pickle.load(fin)
            train_res = pickle.load(fin)
            time_curr_ker = train_res["batchnum" + str(subset_batchnum)]['time_elp'][0]
            time_multi_ker.append(time_curr_ker)
    return time_multi_ker


def get_time_many_subsets(save_folder_parent, batch_sizes, minibatchsize, datasize):
    time_record = []
    time_accumulated = []
    for s in batch_sizes:
        kernels = get_valid_ker(save_folder_parent, s)
        time_multi_ker = get_time(save_folder_parent, s, kernels)
        time_record.append(sum(time_multi_ker))
        time_accumulated.append(sum(time_record))
    percentage = np.array(batch_sizes) * minibatchsize / datasize * 100
    time_results = {'percentage': percentage, 'time': np.array(time_accumulated)}
    df = pd.DataFrame(time_results)
    df.to_csv(save_folder_parent + 'totaltime.csv')
    return time_results


def get_valid_ker(save_folder_parent, subset_batchnum):
    # gathering the elbos for each kernel
    save_folder_child = save_folder_parent + 'train_res_' + str(subset_batchnum) + '/'
    kernels = []
    for file in os.listdir(save_folder_child):
        if file.endswith(".pkl"):
            kernels.append(file[0:-4])
    return kernels


def get_valid_ker_and_elbos(save_folder_parent, subset_batchnum):
    # gathering the elbos for each kernel
    save_folder_child = save_folder_parent + 'train_res_' + str(subset_batchnum) + '/'
    kernels = []
    for file in os.listdir(save_folder_child):
        if file.endswith(".pkl"):
            kernels.append(file[0:-4])
    # elbos
    kernels.sort()
    ELBOs = []
    for ker_str in kernels:
        fname = save_folder_child + ker_str + ".pkl"
        with open(fname, "rb") as fin:
            ELBOs_curr_ker = pickle.load(fin)
            ELBOs.append(ELBOs_curr_ker)
    return kernels, ELBOs


def get_elbos_subset(save_folder_parent, subset_batchnum, kernels):
    save_folder_child = save_folder_parent + 'train_res_' + str(subset_batchnum) + '/'
    kernels.sort()
    ELBOs = []
    for ker_str in kernels:
        fname = save_folder_child + ker_str + ".pkl"
        with open(fname, "rb") as fin:
            ELBOs_curr_ker = pickle.load(fin)
            ELBOs.append(ELBOs_curr_ker)
    return ELBOs


def get_valid_ker_and_elbos_and_rmses(save_folder_parent, subset_batchnum):
    # gathering the elbos for each kernel
    save_folder_child = save_folder_parent + 'train_res_' + str(subset_batchnum) + '/'
    all_ker_name_file = save_folder_child + 'trained_ker.txt'
    with open(all_ker_name_file) as f:
        kernels = f.readlines()
    kernels = [x.strip() for x in kernels]
    kernels = np.array(kernels[1:])
    kernels = np.unique(kernels)
    # elbos
    ELBOs = []
    # rmses
    RMSEs = []

    y_preds_multi_ker = []
    for ker_str in kernels:
        fname = save_folder_child + ker_str + ".pkl"
        with open(fname, "rb") as fin:
            ELBOs_curr_ker = pickle.load(fin)
            ELBOs.append(ELBOs_curr_ker)
            train_res = pickle.load(fin)
            RMSEs_curr_ker = train_res["batchnum" + str(subset_batchnum)]['rmses'][0][0]
            RMSEs.append(RMSEs_curr_ker)
            y_preds = train_res["batchnum" + str(subset_batchnum)]['y_preds'][0][0]
            y_preds_multi_ker.append(y_preds)
    return kernels, ELBOs, RMSEs, y_preds_multi_ker


def naming(file_str, subset_batchnum, size_sample, prior_pg, normalized, choice):
    if normalized:
        normalized_str = '_normalized'
    else:
        normalized_str = ''
    if choice == "prob":

        result_file = file_str + "_PK_" + "size" + str(subset_batchnum) + 'ss' + str(size_sample) + '_p' + str(
            prior_pg) + normalized_str + ".csv"

    else:
        result_file = file_str + "_BMA_result_" + "size" + str(subset_batchnum) + 'ss' + str(
            size_sample) + '_p' + str(prior_pg) + normalized_str + ".csv"

    return result_file


def get_top_kern(subset_batchnum, top_n, working_folder="test_large_kern_set_reuse/"):
    valid_ker, elbos = get_valid_ker_and_elbos(working_folder, subset_batchnum)
    elbos = np.array(elbos)
    max_elbos = np.amax(elbos, axis=1)
    max_elbos = np.array(max_elbos).reshape(-1)

    data = {'kernels': valid_ker,
            'L_i': max_elbos}
    df = pd.DataFrame(data)
    df = df.sort_values(['L_i'], ascending=False)
    # print(df)
    multi_ker_str = df['kernels'][0:top_n]
    return multi_ker_str


def get_top_kern_cmb(batch_sizes, topn, working_folder):
    all_top_ker = []
    for s in batch_sizes:
        top_ker = get_top_kern(s, top_n=topn, working_folder=working_folder)
        all_top_ker.extend(top_ker)
    all_top_ker = np.array(all_top_ker)
    return np.unique(all_top_ker)


# def elbo_estimate_avg(ker, train_results, subset_batchnum=0):
#     if subset_batchnum == 0:
#
#         elbos_per_ker = []
#         n_init = len(train_results[ker]["elbo_loggers"])
#         for j in range(n_init):
#             elbos_logger_per_init = train_results[ker]["elbo_loggers"][j]
#             logger_num = len(elbos_logger_per_init)
#             elbos_per_ker.append(-np.mean(elbos_logger_per_init[logger_num - 20: logger_num]))
#     else:
#         elbos_per_ker = []
#         n_init = len(train_results[ker]["batchnum" + str(subset_batchnum)]["elbo_loggers"][0])
#         for j in range(n_init):
#             elbos_logger_per_init = train_results[ker]["batchnum" + str(subset_batchnum)]["elbo_loggers"][0][
#                 j]
#             logger_num = len(elbos_logger_per_init)
#             elbos_per_ker.append(-np.mean(elbos_logger_per_init[logger_num - 20: logger_num]))
#     return elbos_per_ker


# ------for fulldata------
def get_kernel_names(working_folder):
    kernels = []
    for file in os.listdir(working_folder):
        if file.endswith(".pkl"):
            if file != 'testdata.pkl':
                kernels.append(file[0:-4])

    # sorting
    kernels.sort()
    return kernels


def get_logger_fulldata(working_folder, kernels, run_id=None):
    elbos_logger_multi_ker = []
    elbos_multi_ker = []
    for ker_str in kernels:
        ker_train_results = working_folder + ker_str + ".pkl"
        with open(ker_train_results, "rb") as fin:
            train_res = pickle.load(fin)
            elbos_per_ker = train_res["elbos"]
            i = elbos_per_ker.index(max(elbos_per_ker))
            if run_id is not None:
                i = run_id
            elbos_logger = train_res["elbo_loggers"][i]
            elbos_logger_multi_ker.append(elbos_logger)
            elbos_multi_ker.append(train_res["elbos"][i])
    return elbos_logger_multi_ker, elbos_multi_ker


def elbo_estimate_fulldata(start_loc, logger_range, elbos_logger_multi_ker):
    elbos_avg = []
    for i in range(len(elbos_logger_multi_ker)):
        elbos_logger = elbos_logger_multi_ker[i]
        elbos_avg.append(np.mean(elbos_logger[start_loc:start_loc + logger_range]))
    elbos_avg = - np.array(elbos_avg).reshape(-1)
    return elbos_avg


def get_valid_ker_and_best_elbos_logger(working_folder, run_id=None):
    """
    :param run_id: if None, get the maximum elbo of many runs
    :return:
    """
    kernels = get_kernel_names(working_folder)
    elbos_logger_multi_ker, elbos_multi_ker = get_logger_fulldata(working_folder, kernels, run_id)
    return kernels, elbos_logger_multi_ker, elbos_multi_ker


def naming_iter(file_str, i, size_sample, prior_pg, normalized, choice, position):
    if normalized:
        normalized_str = '_normalized'
    else:
        normalized_str = ''

    if choice == 'prob':
        if position == 'mid':
            result_file = file_str + "_PK_midpoint" + str(i) + 'ss' + str(size_sample) + '_p' + str(
                prior_pg) + normalized_str + ".csv"
        else:  # converged
            result_file = file_str + "_PK_converged_" + 'ss' + str(size_sample) + '_p' + str(
                prior_pg) + normalized_str + ".csv"
    else:
        result_file = file_str + "_BMA_midpoint" + str(i) + 'ss' + str(size_sample) + '_p' + str(
            prior_pg) + normalized_str + ".csv"
    return result_file


# --------
def compute_prob_wrap(valid_ker, elbos, top_n, multi_ker_str, size_sample, prior_pg, r_read_file_name, prob_result_file,
                      working_folder):
    data = {'kernels': valid_ker,
            'L_i': elbos}
    df = pd.DataFrame(data)
    # print(df)
    if top_n is not None:
        multi_ker_str = df['kernels'][0:top_n]
    if multi_ker_str is not None:
        df = df.loc[df['kernels'].isin(list(set(valid_ker) & set(multi_ker_str)))]
    df.to_csv(r_read_file_name, index=None)

    cwd = os.getcwd()
    cpr.compute_prob(size_sample, prior_pg, r_read_file_name, prob_result_file, working_folder)
    # the path will be changed when calling R, change back to the original one
    os.chdir(cwd)
