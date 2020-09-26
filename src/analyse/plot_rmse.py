"""
@Tong
2019-08-24
Plot RMSE of the kernel with the largest elbos and BMA results
Save the results as well (percentage, time)
"""
import numpy as np
import pandas as pd
import os
import paths
import matplotlib.pyplot as plt
from analyse import ana_utils


def gather_and_plot_rmse(working_folder, batch_sizes, file_str, size_sample, prior_pg, minibatchsize, datasize, datastd,
                         normalized=False):
    best_ker_rmse_all = []
    bma_rmse_all = []
    best_ker_mnlp_all = []
    bma_mnlp_all = []
    for subset_batchnum in batch_sizes:
        # prob_result_file = ana_utils.naming(file_str, subset_batchnum, size_sample, prior_pg, normalized, 'prob')
        # prob_result_file = working_folder + prob_result_file
        # prob_k = pd.read_csv(prob_result_file, header=None)
        # prob_best_ker = prob_k[prob_k[0] == rmses["kernels"][0]].iloc[0, 1]
        # prob_best_ker_all.append(prob_best_ker)

        BMA_result_file = ana_utils.naming(file_str, subset_batchnum, size_sample, prior_pg, normalized, 'bma')
        BMA_result_file = working_folder + 'ana_res/' + BMA_result_file

        rmses = pd.read_csv(BMA_result_file)

        best_ker_rmse = rmses["rmse"][0]
        bma_rmse = rmses["rmse"][len(rmses['rmse']) - 1]
        best_ker_rmse_all.append(best_ker_rmse)
        bma_rmse_all.append(bma_rmse)
        best_ker_mnlp = rmses["mnlp"][0]
        bma_mnlp = rmses["mnlp"][len(rmses['mnlp']) - 1]
        best_ker_mnlp_all.append(best_ker_mnlp)
        bma_mnlp_all.append(bma_mnlp)

    best_ker_rmse_all = np.array(best_ker_rmse_all) * datastd
    bma_rmse_all = np.array(bma_rmse_all) * datastd
    percentage = np.round(batch_sizes * minibatchsize / datasize * 100, 2)
    time_results = ana_utils.get_time_many_subsets(working_folder, batch_sizes, minibatchsize, datasize)
    results = {'percentage': percentage, 'single': best_ker_rmse_all, 'Bayesian': bma_rmse_all,
               'time': time_results['time']}
    df = pd.DataFrame(results)
    print(df)
    results_folder = working_folder + 'plots_res/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    filename = results_folder + file_str + "_RMSE_comparison_" + 'ss' + str(
        size_sample) + '_p' + str(prior_pg) + ".csv"
    if normalized:
        filename = results_folder + file_str + "_RMSE_comparison_" + 'ss' + str(
            size_sample) + '_p' + str(prior_pg) + "_normalized.csv"
    df.to_csv(filename, index=None)
    plt.figure()
    plt.plot(percentage, np.array(best_ker_rmse_all), label='single')
    plt.plot(percentage, np.array(bma_rmse_all), label='Bayesian')
    plt.xlabel('Percentage of data used(%)', fontsize=18)
    plt.ylabel('RMSE', fontsize=18)
    plt.legend(fontsize=18)

    plotfilename = results_folder + file_str + "_RMSE_comparison_" + 'ss' + str(
        size_sample) + '_p' + str(prior_pg) + ".pdf"
    if normalized:
        plotfilename = results_folder + file_str + "_RMSE_comparison_" + 'ss' + str(
            size_sample) + '_p' + str(prior_pg) + "_normalized.pdf"

    plt.savefig(plotfilename)


# os.chdir('/home/tengtong/BKS/src/')  # set working folder
os.chdir(paths.cwd)
