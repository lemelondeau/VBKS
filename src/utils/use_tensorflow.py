import os
import gpflow
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def use_gpu(id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use, usually either "0" or "1", "2' "3" "4"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(id)
    # without using any gpu
    # os.environ["CUDA_VISIBLE_DEVICES"]=""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # less memeory
    gpflow.reset_default_session(config=config)
