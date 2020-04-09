import os
import tables

import numpy as np
import tensorflow as tf

from params import *
from sklearn.preprocessing import MinMaxScaler

np.random.seed(13298)

def main():
    S = MinMaxScaler()
    modes = {"train": [], "test": [], "valid": []}
    
    in_path = "../../external/physionet/eeg_data_temples2.h5"
    with tables.open_file(in_path) as h5_file:
        for node in h5_file.walk_nodes("/", "CArray"):
            m = np.random.choice(["train","test","valid"], p=[0.7, 0.1, 0.2])
            modes[m].append(node._v_pathname)
            if m == "train":
                S.partial_fit(node.read()[:,:-1])

    for m in modes:
        hdf_to_tfrecord(modes[m], in_path, SEG_N, S, m)


def hdf_to_tfrecord(node_list, in_path, window_size, S, mode):
    dirname = "../../temp/vae/{}".format(mode)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    with tables.open_file(in_path) as h5_file:
        for node in node_list:
            filename = "{}/{}.tfrecord".format(dirname, node.split("/")[-1])
            with tf.io.TFRecordWriter(filename) as writer:
                data = h5_file.get_node(node).read()
                X = S.transform(data[:,:-1])
            
                for ix in range(window_size, X.shape[0], window_size):
                    Xw = X[ix-window_size:ix,:]
                    y = np.any(data[:,-1][ix-window_size:ix])
                    example_proto = serialize_example(Xw, y)
                    writer.write(example_proto)


def serialize_example(X, y):
    feature = {
        "channels": _bytes_feature(X),
        "label":    _int64_feature(y)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def _bytes_feature(array):
    value = tf.io.serialize_tensor(array.astype(np.float32))
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor    
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__=="__main__":
    main()