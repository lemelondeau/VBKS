import utils.ker_construction as kc
import pandas as pd
import numpy as np
import os
import pickle

ker_file = 'kernelstring3.csv'
ker_str = pd.read_csv(ker_file)
ker_str = np.array(ker_str)
res_dir = '../config/swiss_init_1/'
bracket = False
dim = 1
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
for kstr in ker_str:
    ker = kc.str2ker(kstr[0], dim, False, True)
    ker_values = ker.read_values()
    save_file_name = res_dir + kstr[0] + '.pkl'
    with open(save_file_name, 'wb') as fout:
        pickle.dump(ker_values, fout)
