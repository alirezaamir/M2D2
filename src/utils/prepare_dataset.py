import pyedflib
import tables
import numpy as np
from sklearn.preprocessing import  scale
from utils.params import *
import os
import pickle
import json
from scipy import signal
from sklearn.model_selection import train_test_split
import pandas as pd

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


def epilepsiae():
    for pat in pat_list:
        for label in ['non_seiz', 'seiz']:
            data = pickle.load(open('../../input/epilepsiae/{}/{}_{}.pickle'.format(label, pat, label), 'rb'))
            X = data[label]

            sos = signal.butter(3, 50, fs=FS, btype="lowpass", output="sos")
            X = signal.sosfilt(sos, X, axis=1)
            X = scale(X, axis=1)
            X = np.reshape(X, newshape=(-1, SEG_N, 2))

            Xw = {}
            Xw["train"], Xw["valid"] = train_test_split(X, test_size = 0.1, random_state = 42)

            output_dict = {"X": [], "y": []}
            for mode in ["train", "valid"]:
                dirname = "../../temp/vae_mmd_data/{}/{}/{}".format(SEG_N, "epilepsia_normal", mode)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                filename = "{}/{}.pickle".format(dirname, pat)
                with open(filename, 'wb') as pickle_file:
                    output_dict["X"]= Xw[mode]
                    if label == 'non_seiz':
                        output_dict["y"] = np.zeros(shape=(Xw[mode].shape[0]))
                    else:
                        output_dict["y"] = np.ones(shape=Xw[mode].shape[0])
                    pickle.dump(output_dict, pickle_file)


def epilepsiae_seizures():
    window_size = SEG_N
    dirname = "../../temp/vae_mmd_data/{}/{}".format(SEG_N, "epilepsiae_seizure")
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open("../../input/GAN_data/seizure.json") as json_file:
        seizure_files = json.load(json_file)
        for pat in pat_list:
            root_path = "../../input/GAN_data/{}".format(pat)
            for filename in seizure_files[pat]:
                data = []
                for channel in ["T3F7", "T4F8"]:
                    xlsx_file = "{}/{}_{}.xlsx".format(root_path, channel, filename)
                    dfs = pd.read_excel(xlsx_file, engine='openpyxl')
                    X = dfs.Var1
                    sos = signal.butter(3, 50, fs=FS, btype="lowpass", output="sos")
                    X = signal.sosfilt(sos, X, axis=0)
                    data.append(scale(X, axis=0))
                data = np.transpose(data)
                X_normal = np.asarray(data)

                out_filename = "{}/{}.pickle".format(dirname, filename)
                output_dict = {"X": [], "y": []}
                with open(out_filename, 'wb') as pickle_file:
                    for ix in range(window_size, X_normal.shape[0], window_size):
                        Xw = X_normal[ix - window_size:ix, :]
                        y = 0 if np.sum(dfs.Var2[ix - window_size:ix]) == 0 else 1
                        output_dict["X"].append(Xw)
                        output_dict["y"].append(y)
                    print("Filename: {}, X shape:{}".format(filename, np.array(output_dict["X"]).shape))
                    pickle.dump(output_dict, pickle_file)


def get_epilepsiae_non_seizure(test_patient, state_len):
    with open("../input/GAN_data/seizure.json") as json_file:
        seizure_files = json.load(json_file)
        root_path = "../input/GAN_data/{}".format(test_patient)
        all_filenames = [x for x in os.listdir(root_path) if x.split('.')[0][5:] not in seizure_files[test_patient]]
        random_filenames = np.random.permutation(all_filenames)
        for filename in random_filenames:
            random_filename = filename.split('.')[0][5:]
            data = []
            for channel in ["T3F7", "T4F8"]:
                xlsx_file = "{}/{}_{}.xlsx".format(root_path, channel, random_filename)
                dfs = pd.read_excel(xlsx_file, engine='openpyxl')
                X = dfs.Var1
                sos = signal.butter(3, 50, fs=FS, btype="lowpass", output="sos")
                X = signal.sosfilt(sos, X, axis=0)
                data.append(scale(X, axis=0))
            data = np.transpose(data)
            X_normal = np.asarray(data)


            dirname = "../temp/vae_mmd_data/{}/{}".format(SEG_N, "epilepsiae_non_seizure")
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            out_filename = "{}/{}.pickle".format(dirname, random_filename)
            output_dict = {"X": [], "y": []}
            with open(out_filename, 'wb') as pickle_file:
                for ix in range(SEG_N, X_normal.shape[0], SEG_N):
                    Xw = X_normal[ix - SEG_N:ix, :]
                    y = 0 if np.sum(dfs.Var2[ix - SEG_N:ix]) == 0 else 1
                    output_dict["X"].append(Xw)
                    output_dict["y"].append(y)
                print("Filename: {}, X shape:{}".format(random_filename, np.array(output_dict["X"]).shape))
                pickle.dump(output_dict, pickle_file)
            x = np.array(output_dict["X"])
            if x.shape[0] < state_len:
                continue

            start_point = x.shape[0] // 2 - state_len // 2
            end_point = start_point + state_len
            x = x[start_point:end_point, :, :]

            return np.expand_dims(x, 0)


if __name__ == '__main__':
    # main()
    epilepsiae_seizures()
    # get_epilepsiae_non_seizure('pat_102')
