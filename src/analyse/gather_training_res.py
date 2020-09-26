import numpy as np
import pandas as pd
import pickle
import os


def get_kernel_names(working_folder):
    kernels = []
    for file in os.listdir(working_folder):
        if file.endswith(".pkl"):
            if file != 'testdata.pkl':
                kernels.append(file[0:-4])

    # sorting
    kernels.sort()
    return kernels


def get_fulldata_res(working_folder, kernels):
    elbos_multi_ker = []
    rmses_multi_ker = []
    time_multi_ker = []
    # get information
    for ker in kernels:
        ker_train_results = working_folder + ker + ".pkl"
        with open(ker_train_results, "rb") as fin:
            train_res = pickle.load(fin)
            elbos_per_ker = train_res["elbos"]
            if elbos_per_ker is not None:
                elbos_multi_ker.append(max(elbos_per_ker))
                i = elbos_per_ker.index(max(elbos_per_ker))
                rmses_per_ker = train_res["rmses"]
                rmses_multi_ker.append(rmses_per_ker[i])
                time_loggers_per_ker = train_res["total_t_per_run"]
                time_multi_ker.append(time_loggers_per_ker)
            else:
                elbos_multi_ker.append(None)
                rmses_multi_ker.append(None)
                time_multi_ker.append(None)
    dict = {'kernel': kernels, 'elbo': elbos_multi_ker, 'rmse': rmses_multi_ker, 'time': time_multi_ker}
    df = pd.DataFrame(dict)
    # df = df.sort_values(['elbo'], ascending=False)
    # print(df)
    return df


def get_bks_res_multi_seeds(result_folders, file_str, size_sample, prior_pg, saving_folder):
    num_runs = len(result_folders)
    percentage = []
    single_rmse = []
    bma_rmse = []
    time = []
    for i in range(num_runs):
        rmse_filename = result_folders[i] + 'plots_res/' + file_str + "_RMSE_comparison_" + 'ss' + str(
            size_sample) + '_p' + str(prior_pg) + "_normalized.csv"
        curr_rmse = pd.read_csv(rmse_filename)
        percentage = curr_rmse['percentage'].values
        single_rmse.append(curr_rmse['single'].values)
        bma_rmse.append(curr_rmse['Bayesian'].values)
        if i == 0:
            time = curr_rmse['time'].values
    # get the RMSE values for all runs in the last iteration
    single_rmse_last = np.array(single_rmse)[:, -1]
    bma_rmse_last = np.array(bma_rmse)[:, -1]
    last_rmse = {'single': single_rmse_last, 'bma': bma_rmse_last}
    df = pd.DataFrame(last_rmse)
    filename = saving_folder + file_str + "_RMSE_multi_last.csv"
    df.to_csv(filename, index=None)

    single_rmse_mean = np.mean(np.array(single_rmse), axis=0)
    single_rmse_std = np.std(np.array(single_rmse), axis=0)
    bma_rmse_mean = np.mean(np.array(bma_rmse), axis=0)
    bma_rmse_std = np.std(np.array(bma_rmse), axis=0)

    results = {'percentage': percentage, 'single_mean': single_rmse_mean, 'Bayesian_mean': bma_rmse_mean,
               'single_std': single_rmse_std, 'Bayesian_std': bma_rmse_std,
               'time': time}
    df = pd.DataFrame(results)
    # print(df)
    filename = saving_folder + file_str + "_RMSE_multi_comparison_" + 'ss' + str(
        size_sample) + '_p' + str(prior_pg) + "_normalized.csv"

    df.to_csv(filename, index=None)

    return filename


def full_res_one_ker(working_folder, ker):
    working_folder = '../../results/' + working_folder
    ker_train_results = working_folder + ker + ".pkl"
    with open(ker_train_results, "rb") as fin:
        train_res = pickle.load(fin)
        elbos_per_ker = train_res["elbos"]
        # time = train_res['total_t_per_run']
        # print(time)
        # pred_time = train_res['pred_time_elp']
        # print(pred_time)
        # print(train_res['time_loggers'][0][-1])

    return train_res


def sub_res_one_ker(save_folder_parent, subset_batchnum, ker):
    save_folder_child = '../../results/' + save_folder_parent + 'train_res_' + str(subset_batchnum) + '/'
    time_multi_ker = []
    fname = save_folder_child + ker + ".pkl"
    with open(fname, "rb") as fin:
        ELBOs_curr_ker = pickle.load(fin)
        train_res = pickle.load(fin)
        time_curr_ker = train_res["batchnum" + str(subset_batchnum)]['time_elp'][0]
        time_multi_ker.append(time_curr_ker)
        print(time_multi_ker)
        pred_time = train_res["batchnum" + str(subset_batchnum)]['pred_time_elp']
        print(pred_time)
    return train_res

# wf = '../../results/temper_full_mb512_bracket_inithyper_rerun/'
# wf = '../../results/swiss_full_mb128_inithyperrerun/'
# k = get_kernel_names(wf)
# df = get_fulldata_res(wf, k)
# df.to_csv(wf + 'fulldata_summary.csv')
