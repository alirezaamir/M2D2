import pyedflib
import tables
import numpy as np
from sklearn.preprocessing import  scale
from utils.params import *
import os
import pickle
import json
from scipy import signal

np.random.seed(13298)
FS = 256


def get_TxFx_channels(dict_list):
    T7F7 = 0
    T8F8 = 0
    for idx, channel in enumerate(dict_list):
        if channel['label'] == 'F7-T7':
            T7F7 = idx
        elif channel['label'] == 'F8-T8':
            T8F8 = idx
    return T7F7, T8F8


def compare2hdf(signals):
    in_path = "../../input/eeg_data_temples2.h5"
    db_signal = []
    with tables.open_file(in_path) as h5_file:
        for node in h5_file.walk_nodes("/", "CArray"):
            if node._v_name == 'chb01_03':
                db_signal = node.read()[:, :]
                print("Length: {}".format(len(node.read()[:, :-1])))

    x = db_signal[:, 1]
    y = signals[0][1:-1]
    r = np.corrcoef(x, y)
    print("correlation: {}".format(r))

    print(np.where(db_signal[:, -1] > 0))


def save_pickle(data, dirname, record_name, window_size):
    filename = "{}/{}.pickle".format(dirname, record_name.split('/')[-1][:-4])
    output_dict = {"X": [], "y": []}
    # X = S.transform(data[:, :-1])
    X = data[:, :-1]
    sos = signal.butter(3, 50, fs=FS, btype="lowpass", output="sos")
    X = signal.sosfilt(sos, X, axis=1)
    X_normal = scale(X, axis=0)
    with open(filename, 'wb') as pickle_file:
        for ix in range(window_size, X.shape[0], window_size):
            Xw = X_normal[ix - window_size:ix, :]
            print("Xw std:{}, mean:{}".format(np.std(Xw), np.mean(Xw)))
            y = 0 if np.sum(data[:, -1][ix - window_size:ix]) == 0 else 1
            output_dict["X"].append(Xw)
            output_dict["y"].append(y)
        pickle.dump(output_dict, pickle_file)


def read_edf_file(record_name, seizure_list):
    print("Record: {}".format(record_name))
    edf_filename = '../input/chbmit/1.0.0/{}'.format(record_name)
    signals, signal_headers, header = pyedflib.highlevel.read_edf(edf_filename)
    T7F7, T8F8 = get_TxFx_channels(signal_headers)
    if T7F7 == 0 and T8F8 == 0:
        np.zeros((0,3))
    print("Reading {}".format(record_name))
    channel1 = np.expand_dims(signals[T7F7], axis=1)
    channel2 = np.expand_dims(signals[T8F8], axis=1)
    eeg_data = np.concatenate((channel1, channel2), axis=1)

    length = eeg_data.shape[0]
    label = np.zeros((length, 1))
    if record_name in seizure_list:
        for start, end in seizure_list[record_name]:
            label[start * FS: end*FS] = 1
            print("Start: {}, End: {}".format(start, end))
    eeg_data = np.concatenate((eeg_data, label), axis=1)
    return eeg_data


def main():
    # S = MinMaxScaler()
    seizure_list = json.load(open("../../input/seizures.json"))
    modes = {"train": [], "test": [], "valid": []}
    with open('../../input/chbmit/1.0.0/RECORDS', 'r') as f:
        lines = f.readlines()
        records = lines

    for rec in records:
        record_name = rec.replace('\n', '')
        m = np.random.choice(["train", "valid"], p=[0.9, 0.1])
        modes[m].append(record_name)
        # eeg_data = read_edf_file(record_name, seizure_list)
        # if m == "train":
        #     S.partial_fit(eeg_data[:, :-1])

    for m in modes:
        dirname = "../temp/vae_mmd_data/{}/{}/{}".format(SEG_N, "full_normal", m)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        for record_name in modes[m]:
            data = read_edf_file(record_name, seizure_list)
            save_pickle(data, dirname, record_name, SEG_N)


if __name__ == '__main__':
    main()

