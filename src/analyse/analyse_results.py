import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('')))  # relative directory doesn't work
from analyse import plot_rmse as pr, kern_set_prob as lk, BMA, show_pk as sp, iter_prob as ip, ana_utils
import numpy as np
from utils import training_utils as tu
from utils import parse_cfg as cfg


# --- Plot RMSE. Large kern set---
def plot_rmse(cfg_file, random_seed, file_str='top10'):
    settings, paths, working_folder = cfg.get_settings(cfg_file, 'sub')
    data_to_use_size = settings['data_to_use']
    minibatch_size = settings['minibatch_size']
    num_batch = int(settings['num_of_batch_per_subset'])
    num_subset = int(settings['num_of_subset'])
    batch_sizes = np.array(range(num_subset)) * num_batch + num_batch

    # save and plot the normalized results here
    datastd = 1
    working_folder = '../results/' + working_folder + '_rnd' + str(random_seed) + '/'
    samplesize = 1000
    prior_pg = 0.5
    # --- prob, BMA, plot ---
    for s in batch_sizes:
        lk.compute_pk(s, file_str, samplesize, prior_pg, top_n=10, multi_ker_str=None,
                      working_folder=working_folder, normalized=True)

    for s in batch_sizes:
        BMA.subset_BMA(working_folder, file_str, s, samplesize, prior_pg, True)

    pr.gather_and_plot_rmse(working_folder, batch_sizes, file_str, samplesize, prior_pg, minibatch_size,
                            data_to_use_size,
                            datastd, normalized=True)


def compute_pk_large(cfg_file, random_seed, file_str='rank3'):
    settings, paths, working_folder = cfg.get_settings(cfg_file, 'sub')
    num_batch = int(settings['num_of_batch_per_subset'])
    num_subset = int(settings['num_of_subset'])
    batch_sizes = np.array(range(num_subset)) * num_batch + num_batch
    working_folder = '../results/' + working_folder + '_rnd' + str(random_seed) + '/'
    multi_ker_str = ana_utils.get_top_kern_cmb(batch_sizes, 3, working_folder)
    print(multi_ker_str)
    samplesize = 1000
    prior_pg = 0.1
    for s in batch_sizes:
        lk.compute_pk(s, file_str, samplesize, prior_pg, top_n=None, multi_ker_str=multi_ker_str,
                      working_folder=working_folder, normalized=True, scaler=2)
    data_to_use_size = settings['data_to_use']
    minibatch_size = settings['minibatch_size']
    sp.line_plot_pk(working_folder, file_str, samplesize, prior_pg, batch_sizes, True, minibatch_size, data_to_use_size)
    return

# --- Plot p(k), small kern set ---
# ---------- specify kernels, small_kern_set_prob ---------
def plot_pk_subset(cfg_file, random_seed):
    settings, paths, working_folder = cfg.get_settings(cfg_file, 'sub')
    data_to_use_size = settings['data_to_use']
    minibatch_size = settings['minibatch_size']
    num_batch = int(settings['num_of_batch_per_subset'])
    num_subset = int(settings['num_of_subset'])
    batch_sizes = np.array(range(num_subset)) * num_batch + num_batch
    working_folder = '../results/' + working_folder + '_rnd' + str(random_seed) + '/'
    datastd = 1

    # multi_ker_str = ['s', 'r', 'p', 'p*r+s', 'p*r*r', 'r+r*p', 'p*r', 'p+r*p']
    multi_ker_str = ['s', 'r', 'p', 'p*r+s', 'p*r', 'p+r*p', 'p*p*p', 's*s+s', 'p+p*r', 'r*r+s', 's*s+r', 'p+p*p']
    file_str = 'group2'
    samplesize = 2000
    prior = 0.1
    # compute pk
    for s in batch_sizes:
        lk.compute_pk(s, file_str, samplesize, prior, top_n=None, multi_ker_str=multi_ker_str,
                      working_folder=working_folder, normalized=True)
    # compute rmse
    for s in batch_sizes:
        BMA.subset_BMA(working_folder, file_str, s, samplesize, prior, True)

    # save rmse and plot
    pr.gather_and_plot_rmse(working_folder, batch_sizes, file_str, samplesize, prior, minibatch_size, data_to_use_size,
                            datastd, normalized=True)
    # plot_single_pk
    sp.line_plot_pk(working_folder, file_str, samplesize, prior, batch_sizes, True, minibatch_size, data_to_use_size)
    sp.bar_plot_pk(working_folder, file_str, samplesize, prior, batch_sizes, True, 'subset')


# --------- full data -------------
def plot_pk_full():
    working_folder = "../results/swiss_full_mb128_inithyperrerun/"
    # multi_ker_str = ['s', 'r', 'p', 'p*r+s', 'p*r*r', 'r+r*p', 'p*r', 'p+r*p']
    multi_ker_str = ['s', 'r', 'p', 'p*r+s', 'p*r', 'p+r*p', 'p*p*p', 's*s+s', 'p+p*r', 'r*r+s', 's*s+r', 'p+p*p']
    file_str = 'group2_miditer'
    logger_range = 10
    size_sample = 2000
    prior_pg = 0.1
    scaler = 800
    for i in range(20):
        ip.iter_pk(logger_range, (i + 1) * 100, file_str, size_sample, prior_pg, scaler, top_n=None,
                   multi_ker_str=multi_ker_str, working_folder=working_folder, normalized=True, run_id=2)

    locations = np.arange(100, 2000, 100)
    sp.bar_plot_pk(working_folder, file_str, size_sample, prior_pg, locations, True, 'full')

seed = 10
plot_rmse('../config/swiss_cfg_1.json', seed, file_str='swiss_top10')
compute_pk_large('../config/swiss_cfg_1.json', seed, 'rank3')

# seed = 10
# plot_rmse('../config/air_cfg_1.json', seed, file_str='temper_top10')
# compute_pk_large('../config/air_cfg_1.json', seed, 'rank3')
