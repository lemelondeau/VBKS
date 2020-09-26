# need a good way to show pk
# gather pk
# gather elbos
import pandas as pd
import numpy as np
from analyse import ana_utils
import matplotlib.pyplot as plt
import os


def gather_pk_subset(working_folder, res_str, size_sample, prior_pg, batch_sizes, normalized):
    pk_gathering = []
    kernels = []
    for subset_batchnum in batch_sizes:
        prob_result_file = ana_utils.naming(res_str, subset_batchnum, size_sample, prior_pg, normalized, 'prob')
        prob_result_file = working_folder + 'ana_res/' + prob_result_file
        dat = pd.read_csv(prob_result_file, delimiter=',', header=None)
        # ordering according to kernel names to make it consistent
        dat = dat.sort_values(0)
        if not kernels:
            kernels = dat[0].values.tolist()
        if not kernels == dat[0].values.tolist():
            print("kernel names don't match!")
        pk = dat[1].values
        pk_gathering.append(pk)

    pk_gathering = np.array(pk_gathering)
    return pk_gathering, kernels


def gather_pk_miditer(working_folder, res_str, size_sample, prior_pg, locations, normalized):
    pk_gathering = []
    kernels = []
    for loc in locations:
        prob_result_file = ana_utils.naming_iter(res_str, loc, size_sample, prior_pg, normalized, 'prob', 'mid')
        prob_result_file = working_folder + 'ana_res/' + prob_result_file
        dat = pd.read_csv(prob_result_file, delimiter=',', header=None)
        # ordering according to kernel names to make it consistent
        dat = dat.sort_values(0)
        if not kernels:
            kernels = dat[0].values.tolist()
        if not kernels == dat[0].values.tolist():
            print("kernel names don't match!")
        pk = dat[1].values
        pk_gathering.append(pk)
    return pk_gathering, kernels


"""
pk_gathering:
k1  k2 k3
0.2 0.4 0.4
0.3 0.4 0.3
"""


def line_plot_pk(working_folder, res_str, size_sample, prior_pg, batch_sizes, normalized, minibatchsize, datasize):
    pk_gathering, kernels = gather_pk_subset(working_folder, res_str, size_sample, prior_pg, batch_sizes, normalized)
    save_plot_name = res_str + 's' + str(size_sample) + "_pg" + str(prior_pg) + '_line.pdf'
    percentage = batch_sizes * minibatchsize / datasize * 100
    f = plt.figure()
    for i in range(len(kernels)):
        plt.plot(percentage, pk_gathering[:, i], label=kernels[i])  # , marker=markers[i], markerfacecolor='None')
    plt.ylim([0, 1])
    plt.xlabel('Proportion of data used (%)', fontsize=22)
    plt.ylabel('Kernel posterior belief', fontsize=22)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    f.savefig(working_folder + 'plots_res/' + res_str + '_' + save_plot_name, bbox_inches='tight', pad_inches=0)


def bar_plot_pk(working_folder, res_str, size_sample, prior_pg, batch_sizes, normalized, mode):
    if mode == "subset":
        pk_gathering, kernels = gather_pk_subset(working_folder, res_str, size_sample, prior_pg, batch_sizes,
                                                 normalized)
        save_plot_name = res_str + 's' + str(size_sample) + "_pg" + str(prior_pg) + '_bar.pdf'
    else:
        # fulldata
        locations = batch_sizes
        pk_gathering, kernels = gather_pk_miditer(working_folder, res_str, size_sample, prior_pg, locations, normalized)
        save_plot_name = res_str + 's' + str(size_sample) + "_pg" + str(prior_pg) + '_bar_fulldata.pdf'
    #     make dataframe
    df = pd.DataFrame(pk_gathering, columns=kernels)
    # print(df)
    df.plot.bar(stacked=True)
    plt.xlabel('Proportion of data used (%)', fontsize=22)
    plt.ylabel('Kernel posterior belief', fontsize=22)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    # plt.show()
    if not os.path.exists(working_folder+'plots_res/'):
        os.mkdir(working_folder+'plots_res/')
    plt.savefig(working_folder + 'plots_res/' + res_str + '_' + save_plot_name, bbox_inches='tight', pad_inches=0)
