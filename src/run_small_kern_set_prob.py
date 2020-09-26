"""
subset p(k)
"""
from utils.use_tensorflow import use_gpu
import small_kern_set_prob as sksp
import sys
import numpy as np
import utils.get_init_dict as gidc

# gpu_id = int(sys.argv[1])
# use_gpu(gpu_id)
# subset_batchnum = int(sys.argv[2])
# subset_batchnum_arr = np.array([subset_batchnum])
# multi_ker_str = ['p*s+p*r', 'r', 'p+p+r', 'p*r', 'p+r+s+p', 'p*s+p*r+p', 'p*s+p*r+r']
# multi_ker_str = ['p*r+s', 'r','s']
# multi_ker_str = ['p*r+s', 'r', 's', 'l', 'p*r', 'p*s', 'p*p+s', 'p*r+p', 'r+s*p', 'p*s+r', 'p*p*r']

use_gpu(4)
multi_ker_str = ['p*r+s', 'r', 's', 'l', 'p*r', 'p*p+s', 'p*r+p', 'p*p*r', 'l+r', 'r+s']
subset_batchnum_arr = np.array([20, 40])
reuse_batchnum_arr = np.array([0, 20])
data_to_use = 200000
global_u_size = 400
local_u_size = 400
iter_num = 500
minibatch_size = 512
scale = 1000
test_size = 0.1
n_init = 1
Z_fixed = True  # Z_fixed: whether to use same Z for different subsets
Z_reused = True
lik_reused = True
ker_bracket = None
random_state = 2019
sub_mode = 'random'
save_logger = False

datafile = '../datasets/swissgrid.csv'
settings = {'subset_batchnum_arr': subset_batchnum_arr, 'reuse_batchnum_arr': reuse_batchnum_arr,
            'multi_ker_str': multi_ker_str, 'data_to_use': data_to_use,
            'global_u_size': global_u_size,
            'local_u_size': local_u_size, 'iter_num_subset': iter_num, 'minibatch_size': minibatch_size, 'scale': scale,
            'test_size': test_size, 'n_init': n_init, 'Z_fixed': Z_fixed, 'Z_reused': Z_reused,
            'lik_reused': lik_reused, 'random_s': random_state, 'sub_mode': sub_mode, 'ker_bracket': ker_bracket,
            'save_logger': save_logger}

# mk = gidc.get_best_init(multi_ker_str, "../results/swiss_200k_mb128_depth3_bracket/")
# mk = None
mk = gidc.get_saved_init(multi_ker_str, '../results/swiss_init_14/')
# print(mk.keys())
sksp.subset_pk(datafile, 'BMA_subsets_10k', 1000, 0.5, settings, train_or_not=True,
               working_folder="../results/best_init_pk_group2_8/", normalized=True,
               multi_ker_init_dict=mk)
sksp.subset_pk(datafile, 'BMA_subsets_10k', 1000, 0.3, settings, train_or_not=False,
               working_folder="../results/best_init_pk_group2_8/", normalized=True,
               multi_ker_init_dict=mk)
