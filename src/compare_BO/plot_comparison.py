import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath('')))
from analyse import gather_training_res as gtr
from utils import parse_cfg as cfg


def get_model_evidence(ker_str):
    # string formed kernel
    ind = domain.index(ker_str)
    evidence = evidence_all[ind]
    # Don't forget negative
    return - evidence


def get_model_rmse(ker_str):
    ind = domain.index(ker_str)
    rmse = rmse_all[ind]
    return rmse


def get_time(ker_str):
    ind = domain.index(ker_str)
    train_time = train_time_all[ind]
    return train_time


def pre_trained_res(elbo_file_name):
    results = pd.read_csv(elbo_file_name)
    domain = results['kernel'].values.tolist()
    evidence_all = results['elbo'].values.tolist()
    train_time_all = results['time'].values.tolist()
    rmse_all = results['rmse'].values.tolist()

    return domain, evidence_all, train_time_all, rmse_all


def load_data(datafile):
    data = pd.read_csv(datafile)
    data = np.array(data)
    data = data.astype(float)
    dim = data.shape[1]
    X = data[:, 0:dim - 1]
    y = data[:, -1].reshape(-1, 1)
    return X, y


def random_selection(num, randseed):
    np.random.seed(randseed)
    inds = np.arange(len(domain))
    np.random.shuffle(inds)
    rand_ker = np.array(domain)[inds[0:num]]
    rand_elbo = np.array(evidence_all)[inds[0:num]]
    rand_rmse = np.array(rmse_all)[inds[0:num]]
    rand_train_time = np.array(train_time_all)[inds[0:num]]

    best_ind = []

    for i in range(num):
        best_ind.append(np.argmax(rand_elbo[0:i + 1]))

    record_best_ker = rand_ker[best_ind]
    record_elbo = rand_elbo[best_ind]
    record_rmse = rand_rmse[best_ind]

    obs_time = []
    acc_time = []
    for ker in rand_ker:
        obs_time.append(get_time(ker))
        acc_time.append(sum(obs_time))

    return record_best_ker, record_elbo, record_rmse, obs_time, acc_time


def random_selection_m(iternum, randnum):
    avg_time = np.mean(np.array(train_time_all))

    best_ker_elbos = []
    best_ker_rmses = []
    for i in range(randnum):
        _, elbos, rmses, _, _ = random_selection(iternum, i)
        best_ker_elbos.append(elbos)
        best_ker_rmses.append(rmses * datastd)

    best_ker_elbos = np.array(best_ker_elbos)
    avg_elbos = np.mean(best_ker_elbos, 0)
    avg_elbos_std = np.std(best_ker_elbos, 0)
    avg_rmses = np.mean(best_ker_rmses, 0)
    avg_rmse_std = np.std(best_ker_rmses, 0)
    acc_time = avg_time * np.array(range(iternum)) + avg_time
    return acc_time, avg_elbos, avg_rmses, avg_elbos_std, avg_rmse_std


def plot_rmse_cmp(elbo_file_name, bayesian_file_name, bo_result_filename, datastd, gpu_num=5):
    x = plt.cm.get_cmap('tab10')
    colors = x.colors[1:5]
    blue = sns.color_palette("Set2")[2]
    with open(bo_result_filename, 'rb') as fin:
        res = pickle.load(fin)
    obs_time = res["obs_time"]
    bo_time = res["bo_time"]
    bomins = res["best_ker_seen_elbo"]
    best_ker_seen = res['best_ker_seen']
    best_ker_rmse = []
    for ker in best_ker_seen:
        best_ker_rmse.append(get_model_rmse(ker))
    total_time = [max(obs_time[0:4]) + bo_time[i] + sum(obs_time[4:i + 4]) for i in range(len(bomins))]
    best_ker_rmse = datastd * np.array(best_ker_rmse)
    acc_time, avg_elbos, avg_rmses, avg_elbos_std, avg_rmse_std = random_selection_m(45, 50)

    bayesian_results = pd.read_csv(bayesian_file_name)
    bayesian_time = bayesian_results['time'].values / gpu_num
    bayesian_single = bayesian_results['single'].values * datastd
    bayesian_bma = bayesian_results['Bayesian'].values * datastd

    f = plt.figure(figsize=(4.2, 3))
    err = plt.errorbar(acc_time, avg_rmses, avg_rmse_std, label='Random', color='cornflowerblue')
    err[-1][0].set_linestyle('-.')
    plt.plot(total_time, best_ker_rmse, label='BO', color=colors[0])
    # plt.plot(acc_time, avg_rmses, label='random')
    plt.plot(bayesian_time, bayesian_single, label='VBKS-s', color=colors[2])
    plt.plot(bayesian_time, bayesian_bma, label='VBKS', color=colors[1])

    plt.xlabel('Time elapsed (s)', fontsize=20)
    plt.ylabel('RMSE', fontsize=20)
    plt.legend(fontsize=11)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('VBKS vs BO', fontsize=20)
    #         plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
    result_fig = elbo_file_name[0:-4] + '_VBKSvsBOvsRand' + '_rmse.pdf'
    plt.savefig(result_fig, bbox_inches='tight', pad_inches=0)
    print('Results saved in:')
    print(os.path.dirname(elbo_file_name))
    return best_ker_seen


def plot_rmse_cmp_multi(bayesian_file_name, bo_result_filename, datastd, gpu_num=5):
    x = plt.cm.get_cmap('tab10')
    colors = x.colors[1:5]
    with open(bo_result_filename, 'rb') as fin:
        res = pickle.load(fin)
    obs_time = res["obs_time"]
    bo_time = res["bo_time"]
    zero = np.array([0])
    bo_time = np.concatenate((zero, bo_time))
    bomins = res["best_ker_seen_elbo"]
    best_ker_seen = res['best_ker_seen']
    best_ker_rmse = []
    for ker in best_ker_seen:
        best_ker_rmse.append(get_model_rmse(ker))
    total_time = [max(obs_time[0:4]) + bo_time[i] + sum(obs_time[4:i + 4]) for i in range(len(bomins))]
    best_ker_rmse = datastd * np.array(best_ker_rmse)

    acc_time, avg_elbos, avg_rmses, avg_elbos_std, avg_rmse_std = random_selection_m(45, 50)

    bayesian_results = pd.read_csv(bayesian_file_name)
    bayesian_time = bayesian_results['time'].values / gpu_num
    bayesian_single = bayesian_results['single_mean'].values * datastd
    bayesian_bma = bayesian_results['Bayesian_mean'].values * datastd

    bayesian_single_std = bayesian_results['single_std'].values * datastd
    bayesian_bma_std = bayesian_results['Bayesian_std'].values * datastd

    f = plt.figure(figsize=(4.2, 3))
    err = plt.errorbar(acc_time, avg_rmses, avg_rmse_std, label='Random', color='cornflowerblue')
    err[-1][0].set_linestyle('-.')
    plt.plot(total_time, best_ker_rmse, label='BO', color=colors[0])

    plt.errorbar(bayesian_time, bayesian_single, bayesian_single_std, label='VBKS-s',
                 color=colors[2])
    plt.errorbar(bayesian_time, bayesian_bma, bayesian_bma_std, label='VBKS', color=colors[1])

    plt.xlabel('Time elapsed (s)', fontsize=20)
    plt.ylabel('RMSE', fontsize=20)
    plt.legend(fontsize=11)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('VBKS vs BO', fontsize=22)
    result_fig = bayesian_file_name[0:-4] + '_VBKSvsBOvsRand' + '_rmse_multi.pdf'
    plt.savefig(result_fig, bbox_inches='tight', pad_inches=0)
    print('Results saved in:')
    print(os.path.dirname(bayesian_file_name))

    return


if __name__ == "__main__":
    cfg_file = '../../config/swiss_cfg_1.json'
    file_str = 'swiss_top10'
    prior_pg = 0.5
    seed = 10

    # cfg_file = '../../config/air_cfg_1.json'
    # file_str = 'temper_top10'
    # prior_pg = 0.5
    # seed = 10

    settings, paths, wf = cfg.get_settings(cfg_file, 'full')
    elbo_file_name = '../../results/' + wf + '/' + 'fulldata_summary.csv'
    domain, evidence_all, train_time_all, rmse_all = pre_trained_res(elbo_file_name)
    datafile = '../'+paths['datafile']

    datastd = 1

    _, _, wf_sub = cfg.get_settings(cfg_file, 'sub')
    bayesian_file_name = '../../results/' + wf_sub + '_rnd' + str(seed) + '/' + 'plots_res/' + file_str + "_RMSE_comparison_" + 'ss' + str(
        1000) + '_p' + str(prior_pg) + "_normalized.csv"
    subset_size = 200
    bo_res_file = '../../results/' + wf + '/' + 'bo_subsize' + str(subset_size) + '.pkl'
    plot_rmse_cmp(elbo_file_name, bayesian_file_name, bo_res_file, datastd)

    # ==averaging multiple runs:
    # saving_folder = '../../results/multiple_results/'
    # if not os.path.exists(saving_folder):
    #     os.mkdir(saving_folder)
    # random_seeds = [1, 2, 3]
    # result_folders = ['../../results/' + wf_sub + '_rnd' + str(seed) + '/' for seed in random_seeds]
    # bayesian_file_name = gtr.get_bks_res_multi_seeds(result_folders, file_str, 1000, 0.5, saving_folder)
    # plot_rmse_cmp_multi(bayesian_file_name, bo_res_file, datastd)
