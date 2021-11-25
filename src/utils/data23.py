import numpy as np
import pickle
import pandas as pd
import os
import json


def dataset_training(mode, test_patient, dirname):
    ictal = []
    inter_ictal = []
    non_ictal = []
    filepath = "{}/{}".format(dirname, mode)
    filenames = ["{}/{}".format(filepath, x) for x in os.listdir(filepath) if
                    x.startswith("chb{:02d}".format(test_patient))]
    seizure_list = json.load(open("../../input/seizures.json"))
    print(seizure_list)
    for filename in filenames:
        name = filename.split('/')[-1]
        name = name.split('.')[0]
        pat = int(filename.split('/')[-1][3:5])
        edf_name = "chb{:02d}/{}.edf".format(pat, name)
        print(edf_name)

        with open(filename, "rb") as pickle_file:
            data = pickle.load(pickle_file)
            x = np.array(data["X"])
            del data
            # y = np.array(data["y"])

            for start_ictal, end_ictal in seizure_list[edf_name]:
                for start_index in range(start_ictal * 256, (end_ictal - 4) * 256 + 1, 32):
                    if np.random.rand() < 1:
                        ictal.append(x[start_index:start_index+1024, :])
                    # print("Ictal: {}, {}".format(start_index, start_index+1024))

                start_inter_ictal = max(start_ictal-150, 0)
                end_inter_ictal = min(end_ictal+150, x.shape[0]//256)
                print("Inter ictal Start :{} Stop :{}".format(start_inter_ictal, end_inter_ictal))
                for start_index in range(start_inter_ictal * 256, (end_inter_ictal - 4) * 256 + 1, 64):
                    if (start_ictal - 4)*256 <= start_index <= end_ictal*256:
                        continue
                    inter_ictal.append(x[start_index:start_index+1024, :])
                    # print("Inter-Ictal: {}, {}".format(start_index, start_index + 1024))

                for start_index in range(0, x.shape[0] - 1024, 1024): # TODO: this part repeats for every ictal!
                    if (start_inter_ictal-4)*256 <= start_index <= end_inter_ictal*256:
                        continue
                    non_ictal.append(x[start_index:start_index+1024, :])
                    # print("Non-Ictal: {}, {}".format(start_index, start_index + 1024))
            print("Ictal: {}".format(len(ictal)))
            print("Inter-Ictal: {}".format(len(inter_ictal)))
            print("Non-Ictal: {}".format(len(non_ictal)))

    print("Ictal: {}".format(len(ictal)))
    print("Inter-Ictal: {}".format(len(inter_ictal)))
    print("Non-Ictal: {}".format(len(non_ictal)))
    return np.asarray(ictal), np.asarray(inter_ictal), np.asarray(non_ictal)


def get_non_seizure(test_patient, root='../..'):
    non_ictal = []
    filepath = "{}/temp/vae_mmd_data/23channel/non_ictal".format(root)
    filenames = ["{}/{}".format(filepath, x) for x in os.listdir(filepath) if
                    x.startswith("chb{:02d}".format(test_patient))]
    for filename in filenames:
        name = filename.split('/')[-1]
        name = name.split('.')[0]
        pat = int(filename.split('/')[-1][3:5])

        with open(filename, "rb") as pickle_file:
            data = pickle.load(pickle_file)
            x = np.array(data["X"])
            del data

            for start_index in range(0, x.shape[0] - 1024, 1024):
                non_ictal.append(x[start_index:start_index+1024, :])

    print("Non-Ictal: {}".format(len(non_ictal)))
    return np.asarray(non_ictal), np.zeros((len(non_ictal)))


def get_balanced_data(test_patient, ictal_ratio = 1.0, inter_ratio =1.0, non_ratio = 1.0):
    dir_path = '../input/chbmit_overlapped'
    ictal = np.zeros((0, 1024, 23))
    inter_ictal = np.zeros((0, 1024, 23))
    non_ictal = np.zeros((0, 1024, 23))
    x_total = {"ictal": ictal, "inter_ictal": inter_ictal, "non_ictal": non_ictal}
    ratio = {"ictal": ictal_ratio, "inter_ictal": inter_ratio, "non_ictal": non_ratio}
    for pat in np.random.permutation([12, 1, 2, 15, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23]):
        if pat == test_patient:
            continue

        for mode in ["ictal", "inter_ictal", "non_ictal"]:
            for i in range(10):
                # if np.random.rand() > ratio[mode]:
                #     continue
                pickle_file = open("{}/{}/{}_{}.pickle".format(dir_path, mode, pat, i), "rb")
                data = pickle.load(pickle_file)
                length = data.shape[0]
                random_indices = np.random.choice(length, size=int(length * ratio[mode]), replace=False)
                x_total[mode] = np.concatenate((x_total[mode], data[random_indices]))
                pickle_file.close()
        # if x_total["ictal"].shape[0] +  x_total["inter_ictal"].shape[0]  + x_total["non_ictal"].shape[0] > 5000:
        #     break

    X = np.concatenate((x_total["ictal"], x_total["inter_ictal"], x_total["non_ictal"]), axis=0)
    label = np.concatenate((np.ones(x_total["ictal"].shape[0]),
                            np.zeros(x_total["inter_ictal"].shape[0] + x_total["non_ictal"].shape[0])))
    print("Train shape: {}".format(X.shape))
    return X, label


def get_test_overlapped(test_patient, root=''):
    dir_path = root + '../input/chbmit_overlapped/{}.pickle'
    pickle_file = open(dir_path.format(test_patient), "rb")
    data = pickle.load(pickle_file)
    X = np.concatenate((data["ictal"], data["inter_ictal"], data["non_ictal"]), axis=0)
    label = np.concatenate((np.ones(data["ictal"].shape[0]), np.zeros(data["inter_ictal"].shape[0] + data["non_ictal"].shape[0])))
    pickle_file.close()
    return X, label


def get_test_data(test_patient, root=''):
    filepath = root+'../temp/vae_mmd_data/23channel/train'
    filenames = ["{}/{}".format(filepath, x) for x in os.listdir(filepath) if
                 x.startswith("chb{:02d}".format(test_patient))]
    x_total = []
    y_total = []
    for filename in filenames:
        with open(filename, "rb") as pickle_file:
            data = pickle.load(pickle_file)
            x = np.array(data["X"])
            y = np.array(data["y"])
            for start_index in range(0, x.shape[0] - 1024, 1024):
                x_total.append(x[start_index:start_index + 1024, :])
                y_total.append(0 if np.sum(y[start_index:start_index + 1024]) == 0 else 1)
                # print("Non-Ictal: {}, {}".format(start_index, start_index + 1024))

    return np.asarray(x_total), np.asarray(y_total)


def split_files():
    for pat in range(1, 24):
        dir_path = '../input/chbmit_overlapped'
        pickle_file = open("{}/{}.pickle".format(dir_path, pat), "rb")
        data = pickle.load(pickle_file)
        for mode in ["ictal", "inter_ictal", "non_ictal"]:
            x = np.random.permutation(data[mode])
            length = x.shape[0] // 10
            for i in range(10):
                with open("{}/{}/{}_{}.pickle".format(dir_path, mode, pat, i), "wb") as write_pickle:
                    pickle.dump(x[length*i: length*(i+1)], write_pickle)
        pickle_file.close()


if __name__ == '__main__':
    # X_total = np.zeros((0, 1024, 23))
    # y_total = np.zeros((0,))
    # for pat in range(1,24):
    #     X, y = get_balanced_data(test_patient=pat, ictal_ratio=0.2, inter_ratio=0.2, non_ratio=0.1)
    #     print("X shape: {}, y: {}".format(X.shape, y.shape))
    #     X_total = np.concatenate((X_total, X))
    #     y_total = np.concatenate((y_total, y))
    #     data_dict = {"ictal" : ictal, "inter_ictal": inter_ictal, "non_ictal": non_ictal}
    #     print("ictal shape : {}".format(ictal.shape))
    #     with open("../input/chbmit_overlapped/ictal/{}.pickle".format(pat), "wb") as pickle_file:
    #         pickle.dump(data_dict, pickle_file)
    X, y = get_balanced_data(test_patient=1, ictal_ratio= 0.2, inter_ratio=0.2, non_ratio=0.1)

    print("X shape: {}, y: {}".format(X.shape, y.shape))
    # split_files()
