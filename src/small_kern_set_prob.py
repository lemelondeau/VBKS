"""
@Tong
04-07-2019
specify a kernel set
get p(k) vs subsets size and p(k) vs training time using full data

"""
import os
import paths

# os.environ['R_HOME'] = '/home/tengtong/miniconda3/envs/py36/lib/R'
os.environ['R_HOME'] = paths.r_home
import pandas as pd
import training_wrap_inputs as twi
from analyse import kern_set_prob as lk
from analyse import iter_prob as ipk

"""
saving structure:
working_folder
for each subset size:
-training setting
-training results for all kernels(logger, elbos, y_pred, rmse)
-kernel names
-test data
-intermediate data for computing p(k)
-prob file
"""


def subset_pk(datafile, file_str, size_sample, prior_pg, settings, train_or_not=True, working_folder="first_test_pk/",
              normalized=False, multi_ker_init_dict=None):
    """
    choose the largest elbos for each kernel
    Attention: this should match rmse in BMA.py
    :param datafile:
    :param file_str:
    :param size_sample:
    :param prior_pg:
    :param settings:
    :param train_or_not: if False, don;t need to train again. Use existing results to compute the prob
    :param working_folder:
    :param normalized:
    :param multi_ker_init_dict:
    :return:
    """

    # first get the settings
    subset_batchnum_arr = settings['subset_batchnum_arr']
    reuse_batchnum_arr = settings['reuse_batchnum_arr']
    multi_ker_str = settings['multi_ker_str']
    if not os.path.exists(working_folder):
        os.mkdir(working_folder)
    settings.pop('subset_batchnum_arr', None)
    settings.pop('reuse_batchnum_arr', None)
    if train_or_not:
        df = pd.DataFrame(settings)
        df.to_csv(working_folder + file_str + "_settings.csv")

    for i, s in enumerate(subset_batchnum_arr):
        if train_or_not:
            # train with subsets
            twi.train_one_subset(datafile, settings, s, working_folder, multi_ker_str, multi_ker_init_dict,
                                 reuse_batchnum_arr[i], 'trained_ker.txt')

        lk.compute_pk(s, file_str, size_sample, prior_pg, top_n=None, multi_ker_str=multi_ker_str,
                      working_folder=working_folder, normalized=normalized)
    return


def fulldata_pk(datafile, file_str, size_sample, prior_pg, settings, train_or_not=True, working_folder="first_test_pk/",
                normalized=False, multi_ker_init_dict=None):
    start_loc = settings['start_loc']
    logger_range = settings['logger_range']
    scaler = settings['scaler']
    run_id = settings['run_id']
    multi_ker_str = settings['multi_ker_str']
    settings.pop('multi_ker_init_dict', None)
    if train_or_not:
        df = pd.DataFrame(settings)
        df.to_csv(working_folder + file_str + "_settings.csv")

    # logger_range less than interval
    if train_or_not:
        twi.train_fulldata(datafile, settings, multi_ker_str, multi_ker_init_dict, working_folder)
    ipk.iter_pk(logger_range, start_loc, file_str, size_sample, prior_pg, scaler, top_n=None,
                multi_ker_str=multi_ker_str, working_folder=working_folder, normalized=normalized, run_id=run_id)

    return
