import os
import paths
# os.environ['R_HOME'] = '/home/tengtong/miniconda3/envs/py36/lib/R'
os.environ['R_HOME'] = paths.r_home
import numpy as np
from analyse import ana_utils


# train with fulldata, compute kernel posterior every a few iterations
def iter_pk(logger_range, start_loc, file_str, size_sample, prior_pg, scaler,
            top_n=None, multi_ker_str=None,
            working_folder="iter_pk_test/", normalized=True, run_id=None):
    """
    :param logger_range: get the average elbo over a piece of learning curve
    :param start_loc: staring location on the learning curve, if <=0 use converged results
    :param file_str: an identifier for the results
    :param size_sample:
    :param prior_pg:
    :param scaler:
    :param top_n:
    :param multi_ker_str:
    :param working_folder:
    :param normalized:
    :param run_id:
    :return:
    """
    # read existing results
    valid_ker, elbos_logger_multi_ker, elbos_multi_ker = ana_utils.get_valid_ker_and_best_elbos_logger(working_folder,
                                                                                                       run_id)
    if not os.path.exists(working_folder + 'ana_res/'):
        os.makedirs(working_folder + 'ana_res')
    # todo: scaler
    # compute p(k) in the middle of training
    if start_loc > 0:
        mid_elbos = ana_utils.elbo_estimate_fulldata(start_loc, logger_range, elbos_logger_multi_ker)
        if normalized:
            mid_elbos = mid_elbos / scaler
            normalized_str = '_normalized'
        else:
            normalized_str = ''

        r_read_file_name = working_folder + file_str + "_elbos_midpoint" + str(start_loc) + normalized_str + ".csv"
        prob_result_file = ana_utils.naming_iter(file_str, start_loc, size_sample, prior_pg, normalized, 'prob', 'mid')

        ana_utils.compute_prob_wrap(valid_ker, mid_elbos, top_n, multi_ker_str, size_sample, prior_pg, r_read_file_name,
                                    prob_result_file, working_folder)

    # compute p(k), full data, converged
    # Attention: this should match rmse in BMA.py
    if start_loc <= 0:
        elbos = np.array(elbos_multi_ker)
        if normalized:
            elbos = elbos / scaler
            normalized_str = '_normalized'
        else:
            normalized_str = ''
        r_read_file_name = working_folder + file_str + "_elbos_converged" + normalized_str + ".csv"
        prob_result_file = ana_utils.naming_iter(file_str, 0, size_sample, prior_pg, normalized, 'prob', 'converged')
        ana_utils.compute_prob_wrap(valid_ker, elbos, top_n, multi_ker_str, size_sample, prior_pg, r_read_file_name,
                                    prob_result_file, working_folder)


# os.chdir('/home/tengtong/BKS/src/')  # set working folder
os.chdir(paths.cwd)
