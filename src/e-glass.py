import numpy as np
import scipy.signal, scipy.integrate
from utils.data import dataset_training
from training import get_all_filenames
from sklearn.ensemble import RandomForestClassifier
import pywt
import antropy
import pickle
import matplotlib.pyplot as plt

from utils.data import get_epilepsiae_test
from utils.params import pat_list

import matplotlib
matplotlib.use('TkAgg')


def sh_ren_ts_entropy(x, a, q):
    p, bin_edges = np.histogram(x)
    p = p / np.sum(p)
    p = p[np.where(p > 0)]  # to exclude log(0)
    shannon_en = - np.sum(p * np.log2(p))
    renyi_en = np.log2(np.sum(pow(p, a))) / (1 - a)
    tsallis_en = (1 - np.sum(pow(p, q))) / (q - 1)
    return (shannon_en, renyi_en, tsallis_en)


def bandpower(x, fs, fmin, fmax):
    f, Pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return scipy.integrate.trapz(Pxx[ind_min: ind_max + 1], f[ind_min: ind_max + 1])


def sampen(m, r, L):
    epsilon = 0.001
    N = len(L)
    B = 0.0
    A = 0.0
    # Split time series and save all templates of length m
    xmi = np.array([L[i: i + m] for i in range(N - m)])
    xmj = np.array([L[i: i + m] for i in range(N - m + 1)])
    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])
    # Similar for computing A
    m += 1
    xm = np.array([L[i: i + m] for i in range(N - m + 1)])
    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
    return -np.log((A + epsilon) / (B + epsilon))


def sampen2(dim, r, data):
    epsilon = 0.001
    N = len(data)
    correl = np.zeros(2)
    dataMat = np.zeros((dim + 1, N - dim))
    for i in range(dim + 1):

        dataMat[i, :] = data[i: N - dim + i]

    for m in range(dim, dim + 2):
        count = np.zeros(N - dim)
        tempMat = dataMat[0:m, :]

        for i in range(N - m):
            # calculate distance, excluding self - matching case
            dist = np.max(np.abs(tempMat[:, i + 1: N - dim] - np.tile(tempMat[:, i], ((N - dim - i - 1), 1)).T), axis=0)
            D = (dist < r)
            count[i] = np.sum(D) / (N - dim - 1)

        correl[m - dim] = np.sum(count) / (N - dim)

    saen = np.log((correl[0] + epsilon) / (correl[1] + epsilon))
    return saen


def get_entropy_features(data, samplFreq):
    spectral_entropy = antropy.spectral_entropy(data, sf=samplFreq, normalize=True)
    spectral_entropy = 0 if np.isnan(spectral_entropy) else spectral_entropy
    aprox_entropy = antropy.app_entropy(data)
    aprox_entropy = 0 if np.isnan(aprox_entropy) else aprox_entropy
    higuchi = antropy.higuchi_fd(data)
    higuchi = 0 if np.isnan(higuchi) else higuchi
    return [spectral_entropy, aprox_entropy, higuchi]


def calculateMLfeatures(data, samplFreq):
    ''' function that calculates various features relevant for epileptic seizure detection
    from paper: D. Sopic, A. Aminifar, and D. Atienza, e-Glass: A Wearable System for Real-Time Detection of Epileptic Seizures, 2018
    at the bottom of function choose whether 45 or 54 features are used
    data is a 1D array representing data window from which to extract features
    '''
    # some parameters
    DWTfilterName = 'db4'  # 'sym5'
    DWTlevel = 7
    n1 = 2  # num dimensions for sample entropy
    r1 = 0.2  # num of STD for sample entropy
    r2 = 0.35  # num of STD for sample entropy
    a = 2  # param for shannon, renyi and tsallis enropy
    q = 2  # param for shannon, renyi and tsallis enropy

    # DWT
    coeffs = pywt.wavedec(data, DWTfilterName, level=DWTlevel)
    a7, d7, d6, d5, d4, d3, d2, d1 = coeffs

    # sample entropy
    samp_1_d7_1 = sampen2(n1, r1 * np.std(d7), d7)
    samp_1_d6_1 = sampen2(n1, r1 * np.std(d6), d6)
    samp_2_d7_1 = sampen2(n1, r2 * np.std(d7), d7)
    samp_2_d6_1 = sampen2(n1, r2 * np.std(d6), d6)

    # permutation entropy
    # perm_d7_3 = perm_entropy(d7, order=3, delay=1, normalize=False)
    # perm_d7_5 = perm_entropy(d7, order=5, delay=1, normalize=False)
    # perm_d7_7 = perm_entropy(d7, order=7, delay=1, normalize=False)
    # perm_d6_3 = perm_entropy(d6, order=3, delay=1, normalize=False)
    # perm_d6_5 = perm_entropy(d6, order=5, delay=1, normalize=False)
    # perm_d6_7 = perm_entropy(d6, order=7, delay=1, normalize=False)
    # perm_d5_3 = perm_entropy(d5, order=3, delay=1, normalize=False)
    # perm_d5_5 = perm_entropy(d5, order=5, delay=1, normalize=False)
    # perm_d5_7 = perm_entropy(d5, order=7, delay=1, normalize=False)
    # perm_d4_3 = perm_entropy(d4, order=3, delay=1, normalize=False)
    # perm_d4_5 = perm_entropy(d4, order=5, delay=1, normalize=False)
    # perm_d4_7 = perm_entropy(d4, order=7, delay=1, normalize=False)
    # perm_d3_3 = perm_entropy(d3, order=3, delay=1, normalize=False)
    # perm_d3_5 = perm_entropy(d3, order=5, delay=1, normalize=False)
    # perm_d3_7 = perm_entropy(d3, order=7, delay=1, normalize=False)

    perm_d7_3 = antropy.perm_entropy(d7, order=3, delay=1, normalize=True)
    perm_d7_5 = antropy.perm_entropy(d7, order=5, delay=1, normalize=True)
    perm_d7_7 = antropy.perm_entropy(d7, order=7, delay=1, normalize=True)
    perm_d6_3 = antropy.perm_entropy(d6, order=3, delay=1, normalize=True)
    perm_d6_5 = antropy.perm_entropy(d6, order=5, delay=1, normalize=True)
    perm_d6_7 = antropy.perm_entropy(d6, order=7, delay=1, normalize=True)
    perm_d5_3 = antropy.perm_entropy(d5, order=3, delay=1, normalize=True)
    perm_d5_5 = antropy.perm_entropy(d5, order=5, delay=1, normalize=True)
    perm_d5_7 = antropy.perm_entropy(d5, order=7, delay=1, normalize=True)
    perm_d4_3 = antropy.perm_entropy(d4, order=3, delay=1, normalize=True)
    perm_d4_5 = antropy.perm_entropy(d4, order=5, delay=1, normalize=True)
    perm_d4_7 = antropy.perm_entropy(d4, order=7, delay=1, normalize=True)
    perm_d3_3 = antropy.perm_entropy(d3, order=3, delay=1, normalize=True)
    perm_d3_5 = antropy.perm_entropy(d3, order=5, delay=1, normalize=True)
    perm_d3_7 = antropy.perm_entropy(d3, order=7, delay=1, normalize=True)

    # shannon renyi and tsallis entropy
    (shannon_en_sig, renyi_en_sig, tsallis_en_sig) = sh_ren_ts_entropy(data, a, q)
    (shannon_en_d7, renyi_en_d7, tsallis_en_d7) = sh_ren_ts_entropy(d7, a, q)
    (shannon_en_d6, renyi_en_d6, tsallis_en_d6) = sh_ren_ts_entropy(d6, a, q)
    (shannon_en_d5, renyi_en_d5, tsallis_en_d5) = sh_ren_ts_entropy(d5, a, q)
    (shannon_en_d4, renyi_en_d4, tsallis_en_d4) = sh_ren_ts_entropy(d4, a, q)
    (shannon_en_d3, renyi_en_d3, tsallis_en_d3) = sh_ren_ts_entropy(d3, a, q)

    # band power
    p_tot = bandpower(data, samplFreq, 0, 45)
    p_dc = bandpower(data, samplFreq, 0, 0.5)
    p_mov = bandpower(data, samplFreq, 0.1, 0.5)
    p_delta = bandpower(data, samplFreq, 0.5, 4)
    p_theta = bandpower(data, samplFreq, 4, 8)
    p_alfa = bandpower(data, samplFreq, 8, 13)
    p_middle = bandpower(data, samplFreq, 12, 13)
    p_beta = bandpower(data, samplFreq, 13, 30)
    p_gamma = bandpower(data, samplFreq, 30, 45)
    p_dc_rel = p_dc / p_tot
    p_mov_rel = p_mov / p_tot
    p_delta_rel = p_delta / p_tot
    p_theta_rel = p_theta / p_tot
    p_alfa_rel = p_alfa / p_tot
    p_middle_rel = p_middle / p_tot
    p_beta_rel = p_beta / p_tot
    p_gamma_real = p_gamma / p_tot

    # all features from the paper - 54 features
    featuresAll= [samp_1_d7_1, samp_1_d6_1, samp_2_d7_1, samp_2_d6_1, perm_d7_3, perm_d7_5, perm_d7_7, perm_d6_3, perm_d6_5, perm_d6_7,   perm_d5_3, perm_d5_5,
             perm_d5_7, perm_d4_3, perm_d4_5, perm_d4_7, perm_d3_3, perm_d3_5, perm_d3_7, shannon_en_sig, renyi_en_sig, tsallis_en_sig, shannon_en_d7, renyi_en_d7, tsallis_en_d7,
             shannon_en_d6, renyi_en_d6, tsallis_en_d6, shannon_en_d5, renyi_en_d5, tsallis_en_d5, shannon_en_d4, renyi_en_d4, tsallis_en_d4, shannon_en_d3, renyi_en_d3, tsallis_en_d3,
             p_tot, p_dc, p_mov, p_delta, p_theta, p_alfa, p_middle, p_beta, p_gamma, p_dc_rel, p_mov_rel, p_delta_rel, p_theta_rel, p_alfa_rel, p_middle_rel, p_beta_rel, p_gamma_real]
    # here I exclude features that are not normalized - 45 features
    # featuresAll = [samp_1_d7_1, samp_1_d6_1, samp_2_d7_1, samp_2_d6_1, perm_d7_3, perm_d7_5, perm_d7_7, perm_d6_3,
    #                perm_d6_5, perm_d6_7, perm_d5_3, perm_d5_5,
    #                perm_d5_7, perm_d4_3, perm_d4_5, perm_d4_7, perm_d3_3, perm_d3_5, perm_d3_7, shannon_en_sig,
    #                renyi_en_sig, tsallis_en_sig, shannon_en_d7, renyi_en_d7, tsallis_en_d7,
    #                shannon_en_d6, renyi_en_d6, tsallis_en_d6, shannon_en_d5, renyi_en_d5, tsallis_en_d5, shannon_en_d4,
    #                renyi_en_d4, tsallis_en_d4, shannon_en_d3, renyi_en_d3, tsallis_en_d3,
    #                p_dc_rel, p_mov_rel, p_delta_rel, p_theta_rel, p_alfa_rel, p_middle_rel, p_beta_rel, p_gamma_real]
    return (featuresAll)


def prepare_data(test_patient):
    all_filenames = get_all_filenames(entire_dataset=True)
    mode = 'train'
    features_dict = {}
    for filename in all_filenames[mode][test_patient]:

        file_pat = filename.split('/')[-1][:-7]
        print(file_pat)

        with open(filename, "rb") as pickle_file:
            data = pickle.load(pickle_file)
            x = np.array(data["X"])
            y = np.array(data["y"])
            if np.sum(y) == 0:
                continue
            print("X Shape : {}".format(x.shape))
            features = np.zeros((x.shape[0], 108))
            for sample in range(x.shape[0]):
                features[sample, :54] = calculateMLfeatures(x[sample,:,0], samplFreq=256)
                features[sample, 54:] = calculateMLfeatures(x[sample,:,1], samplFreq=256)
            features_dict[file_pat] = features

    pickle.dump(features_dict, open("../test_code/Features_Eglass_valid.pickle", "wb"))


def prepare_epilepsiae():
    features_dict = {}
    for pat in pat_list:
        dataset = get_epilepsiae_test(test_patient=pat)
        for pat_name in dataset.keys():
            x = np.array(dataset[pat_name]["data"])
            y = np.array(dataset[pat_name]["label"])
            if np.sum(y) == 0:
                continue
            print("X Shape : {}".format(x.shape))
            features = np.zeros((x.shape[0], 108))
            for sample in range(x.shape[0]):
                features[sample, :54] = calculateMLfeatures(x[sample, :, 0], samplFreq=256)
                features[sample, 54:] = calculateMLfeatures(x[sample, :, 1], samplFreq=256)
            features_dict[pat_name] = features

    pickle.dump(features_dict, open("../test_code/Features_Eglass_epilepsiae.pickle", "wb"))


def prepare_pickle_files():
    new_features_dict = {}
    features_dict = pickle.load(open("../test_code/Features_Eglass_epilepsiae.pickle", "rb"))
    for pat in pat_list:
        dataset = get_epilepsiae_test(test_patient=pat)
        for pat_name in dataset.keys():
            y = np.array(dataset[pat_name]["label"])
            if np.sum(y) == 0:
                continue
            new_features_dict[pat_name] = {"X": features_dict[pat_name], "y": y}

    pickle.dump(new_features_dict, open("../test_code/Features_Eglass_new_epilepsiae.pickle", "wb"))


def add_entropy_features():
    features_dict = pickle.load(open("../test_code/Features_Eglass_chb.pickle", "rb"))
    print(features_dict)
    all_filenames = get_all_filenames(entire_dataset=True)
    mode = 'train'
    new_features_dict = {}
    for test_patient in ["-1"]:
        for filename in all_filenames[mode][test_patient]:

            file_pat = filename.split('/')[-1][:-7]
            print(file_pat)

            with open(filename, "rb") as pickle_file:
                data = pickle.load(pickle_file)
                x = np.array(data["X"])
                y = np.array(data["y"])
                if np.sum(y) == 0:
                    continue

                features = np.zeros((x.shape[0], 57*2))
                for idx, sample in enumerate(range(x.shape[0])):
                    features[sample, :54] = features_dict[file_pat]["X"][idx, :54]
                    features[sample, 54:57] = get_entropy_features(x[sample, :, 0], samplFreq=256)
                    features[sample, 57:57+54] = features_dict[file_pat]["X"][idx, 54:]
                    features[sample, 57+54:] = get_entropy_features(x[sample, :, 1], samplFreq=256)
                new_features_dict[file_pat] = {"X": features, "y": y}
                print("pat {} new features: {}, {}, {}".format(file_pat, np.mean(features[:, 54]), np.mean(features[:, 55]), np.mean(features[:, 56])))
            # new_features_dict[pat_name] = {"X": features_dict[pat_name], "y": y}
            # print("Shape: {}".format(features_dict[pat_name]["X"].shape))

    pickle.dump(new_features_dict, open("../test_code/Features_Eglass_entropy_chb.pickle", "wb"))


def classify():
    features_dict = pickle.load(open("../test_code/Features_Eglass_entropy_chb.pickle", "rb"))
    middle_diff = []
    for test_patient in range(1,24):#pat_list: #range(1,24):
        train_files = [x for x in features_dict.keys() if not x.startswith("chb{:02d}".format(test_patient))]
        # train_files = [x for x in features_dict.keys() if not x.startswith("{}".format(test_patient))]
        test_files = [x for x in features_dict.keys() if x.startswith("chb{:02d}".format(test_patient))]
        # test_files = [x for x in features_dict.keys() if x.startswith("{}".format(test_patient))]
        train_data = np.zeros((0,114))
        train_label = np.zeros((0,))
        for pat_file in train_files:
            train_data = np.concatenate((train_data, features_dict[pat_file]['X']))
            train_label = np.concatenate((train_label, features_dict[pat_file]['y']))

        print("Train : {}".format(train_data.shape))
        print("Train : {}".format(train_label.shape))
        np.nan_to_num(train_data, copy=False)

        non_seizure_index = np.where(train_label == 0)[0]
        seizure_index = np.where(train_label != 0)[0]
        seizure_count = seizure_index.shape[0]
        rf = RandomForestClassifier(n_estimators=200)

        for iter in range(20):
            print ("Iteration : {}".format(iter))
            non_seizure_index = np.random.permutation(non_seizure_index)
            non_seizure_index_balanced = non_seizure_index[:seizure_count]
            train_data_balanced = np.concatenate((train_data[seizure_index], train_data[non_seizure_index_balanced]))
            train_label_balanced = np.concatenate((np.ones(seizure_count), np.zeros(seizure_count)))

            idx = np.random.permutation(2 * seizure_count)
            train_data_balanced = train_data_balanced[idx]
            train_label_balanced = train_label_balanced[idx]

            rf.fit(train_data_balanced, train_label_balanced)
        pickle.dump(rf, open("../test_code/entropy_models/model_{}.pickle".format(test_patient), "wb"))

        for pat_file in test_files:
            test_data = features_dict[pat_file]['X']
            np.nan_to_num(test_data, copy=False)
            test_label = features_dict[pat_file]['y']

            y_non_zero = np.where(test_label > 0, 1, 0)
            y_non_zero = np.concatenate((y_non_zero, [0]))
            # For sections which have seizure at the end or start of the section
            y_non_zero = np.concatenate(([0], y_non_zero,))
            y_diff = np.diff(y_non_zero)
            start_points = np.where(y_diff > 0)[0]
            stop_points = np.where(y_diff < 0)[0]

            accepted_points = []
            for start, stop in zip(start_points, stop_points):
                accepted_points += range(start, stop)

            predict = rf.predict_proba(test_data)
            rf_max = np.argmax(predict[:,1])
            t_diff = np.abs(accepted_points - rf_max)
            middle_diff.append(np.min(t_diff))
            print("Pat : {} - Time diff: {} ".format(pat_file, np.min(t_diff)))

    print(middle_diff)


def classify_epilepsiae():
    features_dict = pickle.load(open("../test_code/Features_Eglass_entropy_chb.pickle", "rb"))
    middle_diff = []
    train_files = features_dict.keys()
    train_data = np.zeros((0, 114))
    train_label = np.zeros((0,))
    for pat_file in train_files:
        train_data = np.concatenate((train_data, features_dict[pat_file]['X']))
        train_label = np.concatenate((train_label, features_dict[pat_file]['y']))

    np.nan_to_num(train_data, copy=False)

    non_seizure_index = np.where(train_label == 0)[0]
    seizure_index = np.where(train_label != 0)[0]
    seizure_count = seizure_index.shape[0]
    rf = RandomForestClassifier(n_estimators=200)

    for iter in range(20):
        print ("Iteration : {}".format(iter))
        non_seizure_index = np.random.permutation(non_seizure_index)
        non_seizure_index_balanced = non_seizure_index[:seizure_count]
        train_data_balanced = np.concatenate((train_data[seizure_index], train_data[non_seizure_index_balanced]))
        train_label_balanced = np.concatenate((np.ones(seizure_count), np.zeros(seizure_count)))

        idx = np.random.permutation(2 * seizure_count)
        train_data_balanced = train_data_balanced[idx]
        train_label_balanced = train_label_balanced[idx]

        rf.fit(train_data_balanced, train_label_balanced)

    pickle.dump(rf, open("../test_code/models/model_chb_{}.pickle".format(-1), "wb"))
    features_dict = pickle.load(open("../test_code/Features_Eglass_entropy_epilepsiae.pickle", "rb"))

    test_files = features_dict.keys()
    for pat_file in test_files:
        test_data = features_dict[pat_file]['X']
        np.nan_to_num(test_data, copy=False)
        test_label = features_dict[pat_file]['y']

        y_non_zero = np.where(test_label > 0, 1, 0)
        y_non_zero = np.concatenate((y_non_zero, [0]))
        # For sections which have seizure at the end or start of the section
        y_non_zero = np.concatenate(([0], y_non_zero,))
        y_diff = np.diff(y_non_zero)
        start_points = np.where(y_diff > 0)[0]
        stop_points = np.where(y_diff < 0)[0]

        accepted_points = []
        for start, stop in zip(start_points, stop_points):
            accepted_points += range(start, stop)

        predict = rf.predict_proba(test_data)
        rf_max = np.argmax(predict[:, 1])
        t_diff = np.abs(accepted_points - rf_max)
        middle_diff.append(np.min(t_diff))
        print("Pat : {} - Time diff: {} ".format(pat_file, np.min(t_diff)))
    print(middle_diff)


def inference(test_patient, train_dataset, test_dataset = 'chb'):
    rf = pickle.load(open("../test_code/models/{}/model_{}.pickle".format(train_dataset, test_patient), "rb"))
    features_dict = pickle.load(open("../test_code/Features_Eglass_{}.pickle".format(test_dataset), "rb"))

    # if test_patient == -1:
    test_files = features_dict.keys()  #
    # else:
    # test_files = [x for x in features_dict.keys() if x.startswith("chb{:02d}".format(test_patient))]
        # test_files = [x for x in features_dict.keys() if x.startswith("{}".format(test_patient))]

    middle_diff = {}
    for pat_file in test_files:
        middle_diff[pat_file] = {}
        for t_duration in [0]:#[0, 4, 8, 15, 37, 75]:
            start_remove = []
            stop_remove = []
            for attempt in range(1):
                test_data = features_dict[pat_file]['X']
                np.nan_to_num(test_data, copy=False)
                test_label = features_dict[pat_file]['y']

                X_section = test_data
                for start, stop in zip(start_remove, stop_remove):
                    X_section = np.delete(X_section, range(start, stop), axis=0)

                y_true_section = test_label
                for start, stop in zip(start_remove, stop_remove):
                    y_true_section = np.delete(y_true_section, range(start, stop), axis=0)

                if np.sum(y_true_section) == 0:
                    middle_diff[pat_file]["{}_{}".format(t_duration, attempt)] = -1
                    continue

                y_non_zero = np.where(y_true_section > 0, 1, 0)
                y_non_zero = np.concatenate((y_non_zero, [0]))
                # For sections which have seizure at the end or start of the section
                y_non_zero = np.concatenate(([0], y_non_zero,))
                y_diff = np.diff(y_non_zero)
                start_points = np.where(y_diff > 0)[0]
                stop_points = np.where(y_diff < 0)[0]

                accepted_points = []
                for start, stop in zip(start_points, stop_points):
                    accepted_points += range(start, stop)

                predict = rf.predict_proba(X_section)
                rf_max = np.argmax(predict[:, 1])

                node_diff = []
                for idx in range(predict.shape[0]):
                    t_diff = np.abs(np.subtract(accepted_points, idx))
                    # LOG.info("Time diff : {}".format(np.min(t_diff)))
                    node_diff.append((predict[idx, 1], np.min(t_diff)))
                middle_diff[pat_file] = node_diff

                # start_detected_point = max(rf_max - t_duration, 0)
                # start_remove.append(start_detected_point)
                # stop_detected_point = min(rf_max + t_duration + 1, X_section.shape[0])
                # stop_remove.append(stop_detected_point)

                # t_diff = np.abs(accepted_points - rf_max)
                # print("Pat : {}_ t{} _ n{} - Time diff: {} ".format(pat_file, t_duration, attempt, np.min(t_diff)))
                #
                # middle_diff[pat_file]["{}_{}".format(t_duration, attempt)] = (int(np.min(t_diff)))


    return middle_diff


if __name__ == '__main__':
    # prepare_data("-1")
    # prepare_epilepsiae()
    # classify()
    classify_epilepsiae()
    # inference(-1)
    # prepare_pickle_files()
    # middle_diff = {}
    # for pat in range(1,24):
    #     middle_diff.update(inference(pat))
    # print(middle_diff)
    # diffs = {'train': [], 'test': []}
    # for pat in range(1,24):
    #     diffs['train'].append(inference(pat,'chb', 'chb'))
    # # diffs['test'] = inference(-1, 'chb', 'new_epilepsiae')
    #
    # with open('../output/eglass_chb_loocv.pickle', 'wb') as outfile:
    #     pickle.dump(diffs, outfile)
    # print(diffs)
    # add_entropy_features()

