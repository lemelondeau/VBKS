"""
@Tong
2019-07-22

"""
# after distributing kernels to different gpus, detecting whether training is finished
# gather the training results to compute kernel posterior
import os
import paths

# os.environ['R_HOME'] = '/home/tengtong/miniconda3/envs/py36/lib/R'
os.environ['R_HOME'] = paths.r_home
import numpy as np
import pandas as pd
from analyse import compute_prob_use_r as cpr
from analyse import ana_utils


def compute_pk(subset_batchnum, res_str, size_sample, prior_pg,
               top_n=None,
               multi_ker_str=None,
               working_folder="test_large_kern_set_reuse/",
               normalized=False, scaler=2):
    """
    compute the probability of kernels for a SUBSET
    don't necessarily need to be a LARGE kernel set
    :param subset_batchnum: specify the subset size to work with
    :param res_str: a string to label the result
    :param size_sample: number of samples used for computing the probability
    :param prior_pg: prior of p(g)
    :param top_n: compute the pk using only top n kernels. Use all kernel if None.
    :param multi_ker_str: the kernels. Use all kernels or top_n kernels if None.
    :param working_folder: the working folder path. If None, use all valid_ker in the folder.
    :param normalized: whether to normaliza the ELBO for computing the probability
    :param scaler: to make the absolute elbo roughly smaller than a certain value, this is for computing the prob
    :return:
    """
    valid_ker, elbos = ana_utils.get_valid_ker_and_elbos(working_folder, subset_batchnum)
    elbos = np.array(elbos)
    if normalized:
        elbos = elbos / subset_batchnum * scaler
        normalized_str = '_normalized'
    else:
        normalized_str = ''
    max_elbos = np.amax(elbos, axis=1)
    max_elbos = np.array(max_elbos).reshape(-1)

    # save as csv file, for computing p(k)
    # "kernels" "L_i"
    r_read_file_name = working_folder + res_str + "_elbos_size" + str(subset_batchnum) + normalized_str + ".csv"
    prob_result_file = ana_utils.naming(res_str, subset_batchnum, size_sample, prior_pg, normalized, 'prob')
    if not os.path.exists(working_folder + 'ana_res/'):
        os.makedirs(working_folder + 'ana_res')

    data = {'kernels': valid_ker,
            'L_i': max_elbos}
    df = pd.DataFrame(data)
    df = df.sort_values(['L_i'], ascending=False)
    print(df)
    if top_n is not None:
        multi_ker_str = df['kernels'][0:top_n]
    if multi_ker_str is not None:
        df = df.loc[df['kernels'].isin(list(set(valid_ker) & set(multi_ker_str)))]
    print(df)
    print(r_read_file_name)
    df.to_csv(r_read_file_name, index=None)

    cwd = os.getcwd()
    # compute p(k)
    cpr.compute_prob(size_sample, prior_pg, r_read_file_name, prob_result_file, working_folder)
    # the path will be changed when calling R, change back to the original one
    os.chdir(cwd)


# set working folder back
# os.chdir('/home/tengtong/BKS/src/')
os.chdir(paths.cwd)
