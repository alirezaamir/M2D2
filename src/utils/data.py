import numpy as np
import pickle
import pandas as pd
import os
from utils.params import SEG_N


def dataset_training(mode, test_patient, all_filenames, max_len=899, state_len=300):
    X_total = []
    y_total = []
    seizure_len = []

    for filename in all_filenames[mode][test_patient]:
        file_pat = int(filename.split('/')[-1][3:5])

        with open(filename, "rb") as pickle_file:
            data = pickle.load(pickle_file)
            x = np.array(data["X"])
            y = np.array(data["y"])
            if np.sum(y) == 0:
                continue
            seizure_len.append(np.sum(y))
            y = np.expand_dims(y, -1)
            if x.shape[0] == max_len:
                x_selected = x
                y_selected = y
            elif x.shape[0] < max_len:
                diff = max_len - x.shape[0]
                x = np.pad(x, pad_width=[(0, diff), (0, 0), (0, 0)], constant_values=0)
                x_selected = x
                y = np.pad(y, pad_width=[(0, diff), (0, 0)], constant_values=0)
                y_selected = y
            elif x.shape[0] > max_len:
                for start in range(0, x.shape[0] - max_len, (max_len//4)):
                    end = start + max_len
                    if np.sum(y[start:end]) == 0:
                        continue
                    x_selected = x[start:end, :, :]
                    y_selected = y[start:end]

            if state_len is None:
                X_total.append(x_selected)
                y_total.append(y_selected)
            else:
                x_edge = get_non_seizure_signal(file_pat, state_len)
                x_edge = np.squeeze(x_edge, axis=0)
                concatenated = np.concatenate((x_edge,x_selected, x_edge), axis=0)
                X_total.append(concatenated)

                y_true_section = np.concatenate((np.zeros((state_len, 1), dtype=np.float32),
                                                 y_selected,
                                                 np.zeros((state_len, 1), dtype=np.float32)))
                y_total.append(y_true_section)

    # balance_ratio = max_len / np.mean(seizure_len)
    balance_ratio = 1
    print("Balanced Ratio : {}".format(balance_ratio))
    return np.asarray(X_total), np.asarray(y_total) * balance_ratio


def get_non_seizure_signal(test_patient, state_len, root = '..'):
    sessions = build_dataset_pickle(test_patient, root=root)
    sessions_permuted = np.random.permutation([s for s in sessions.keys()])
    for node in sessions_permuted:
        patient_num = int(node[3:5])
        if test_patient != patient_num:
            continue

        x = sessions[node]['data']
        y_true = sessions[node]['label']

        if np.sum(y_true) != 0:
            continue

        if x.shape[0] < state_len:
            continue
        print("Non Seizure node for initialization : {}".format(node))
        start_point = x.shape[0] // 2 - state_len // 2
        end_point = start_point + state_len
        x = x[start_point:end_point, :, :]

        return np.expand_dims(x, 0)
    return np.random.randn(state_len)


def get_epilepsiae_non_seizure(test_patient, state_len, root='..'):
    dirname = "{}/temp/vae_mmd_data/{}/{}/{}".format(root, SEG_N, "epilepsiae_non_seizure", test_patient)
    all_filenames = ["{}/{}".format(dirname, x) for x in os.listdir(dirname)]
    random_filenames = np.random.permutation(all_filenames)
    for filename in random_filenames:
        with open(filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            x = np.array(data["X"])
            if x.shape[0] < state_len:
                continue

            start_point = x.shape[0] // 2 - state_len // 2
            end_point = start_point + state_len
            x = x[start_point:end_point, :, :]

            return np.expand_dims(x, 0)


def make_train_label_bbox(y_true):
    y_true = np.squeeze(y_true)
    y_non_zero = np.where(y_true > 0, 1, 0)
    y_diff = np.diff(y_non_zero)
    start_points = np.where(y_diff > 0)[0]
    stop_points = np.where(y_diff < 0)[0]
    middle_points = (start_points + stop_points) // 2
    seizure_len = stop_points - start_points
    longest_seizure = np.argmax(seizure_len)
    box_mean = middle_points[longest_seizure] / y_true.shape[0]
    return np.asarray(box_mean).astype('float32')


def make_train_label_classification(y_true):
    y_true = np.squeeze(y_true)
    y_non_zero = np.where(y_true > 0, 1, 0)
    y_diff = np.diff(y_non_zero)
    start_points = np.where(y_diff > 0)[0]
    stop_points = np.where(y_diff < 0)[0]
    middle_points = (start_points + stop_points) // 2
    seizure_len = stop_points - start_points
    longest_seizure = np.argmax(seizure_len)
    y_class = middle_points[longest_seizure]
    return np.asarray(y_class).astype('float32')


def get_epilepsiae_seizures(mode, test_patient, dirname, max_len=899, state_len = 300):
    X_total = []
    y_total = []
    mode_dirname = "{}/{}".format(dirname, mode)
    all_filenames = ["{}/{}".format(mode_dirname, x) for x in os.listdir(mode_dirname) if
                     not x.startswith(str(test_patient))]
    for filename in all_filenames:
        file_pickle_name = filename.split('/')[-1]
        file_pat = "pat_{}".format(file_pickle_name.split('_')[1])
        with open(filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            x = np.array(data["X"])
            y = np.array(data["y"])

            y = np.expand_dims(y, -1)
            if x.shape[0] == max_len:
                x_selected = x
                y_selected = y
            elif x.shape[0] < max_len:
                diff = max_len - x.shape[0]
                x = np.pad(x, pad_width=[(0, diff), (0, 0), (0, 0)], constant_values=0)
                y = np.pad(y, pad_width=[(0, diff), (0, 0)], constant_values=0)
                x_selected = x
                y_selected = y
            elif x.shape[0] > max_len:
                for i in range(x.shape[0] // max_len):
                    start = i * max_len
                    end = (i + 1) * max_len
                    if np.sum(y[start:end]) == 0:
                        continue
                    x_selected = x[start:end, :, :]
                    y_selected = y[start:end, :]
            if state_len is None:
                X_total.append(x_selected)
                y_total.append(y_selected)
            else:
                x_edge = get_epilepsiae_non_seizure(file_pat, state_len)
                x_edge = np.squeeze(x_edge, axis=0)
                concatenated = np.concatenate((x_edge, x_selected, x_edge), axis=0)
                X_total.append(concatenated)

                y_true_section = np.concatenate((np.zeros((state_len, 1), dtype=np.float32),
                                                 y_selected,
                                                 np.zeros((state_len, 1), dtype=np.float32)))
                y_total.append(y_true_section)
    return np.asarray(X_total), np.asarray(y_total)


def get_epilepsiae_test(test_patient, root='..'):
    dataset = {}
    for mode in ["train", "valid"]:
        dirname = "{}/temp/vae_mmd_data/1024/epilepsiae_seizure/{}".format(root, mode)
        filenames = ["{}/{}".format(dirname, x) for x in os.listdir(dirname) if x.startswith(test_patient)]
        for filename in filenames:
            with open(filename, "rb") as pickle_file:
                pickle_name = filename.split('/')[-1]
                name = pickle_name.split('.')[0]
                data = pickle.load(pickle_file)
                x = np.array(data["X"])
                y = np.array(data["y"])
                dataset[name] = {'data': x, 'label': y}

    return dataset


def get_new_conv_w(state_len, N=6, state_dim=7):
    new_conv_weight = np.zeros((N * 2 + 1, 2 * state_dim, N+1), dtype=np.float)

    for ch in range(N+1):
        w_len = (ch * 2 + 1)
        for i in range(ch * 2 + 1):
            new_conv_weight[N + i - ch, ch * 2 - i, ch] = 1.0 / (w_len * w_len)
            new_conv_weight[N + i - ch, state_dim + i, ch] = 1.0 / (w_len * w_len)

            new_conv_weight[N + i - ch, state_dim - 1, ch] = -1.0 / (w_len * state_len)
            new_conv_weight[N + i - ch, 2 * state_dim - 1, ch] = -1.0 / (w_len * state_len)

    return [new_conv_weight]


def get_seizure_point_from_label(y_true):
    y_non_zero = np.where(y_true > 0, 1, 0)
    y_non_zero = np.concatenate((y_non_zero, [0]))
    # For sections which have seizure at the end or start of the section
    y_non_zero = np.concatenate(([0], y_non_zero,))
    y_diff = np.diff(y_non_zero)
    start_points = np.where(y_diff > 0)[0]
    stop_points = np.where(y_diff < 0)[0]

    accepted_points = []
    for start, stop in zip(start_points, stop_points):
        accepted_points += range(start, stop)
    return accepted_points


def build_dataset_pickle(test_patient, root='..'):
    dataset = {}
    for mode in ["train" , "valid"]:
        dirname = "{}/temp/vae_mmd_data/{}/full_normal/{}".format(root, SEG_N, mode)
        filenames = ["{}/{}".format(dirname, x) for x in os.listdir(dirname) if x.startswith("chb{:02d}".format(test_patient))]
        for filename in filenames:
            with open(filename, "rb") as pickle_file:
                pickle_name = filename.split('/')[-1]
                name = pickle_name[:-7]
                data = pickle.load(pickle_file)
                x = np.array(data["X"])
                y = np.array(data["y"])
                dataset[name] = {'data': x, 'label': y}

    return dataset