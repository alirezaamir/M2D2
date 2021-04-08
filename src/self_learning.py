import numpy as np
import os
import pickle
from utils.params import SEG_N, pat_hours
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from vae_mmd import build_dataset_pickle as test_dataset


TH = 0.5


def borrow_seizure():
    x_borrow = np.zeros((0, 899, 1024, 2))
    y_borrow = np.zeros((0, 899))

    pat_rand = np.random.randint(1, 24, size=3)
    for pat in pat_rand:
        dataset = test_dataset(pat)
        for session in dataset.keys():
            x_in = dataset[session]['data']
            y_in = dataset[session]['label']
            if np.sum(y_in) == 0:
                continue
            if x_in.shape[0] != 899:
                continue
            x_borrow = np.concatenate((x_borrow, np.expand_dims(x_in, 0)))
            y_borrow = np.concatenate((y_borrow, np.expand_dims(y_in, 0)))
    return x_borrow, y_borrow


def self_eval(model, test_patient):
    x_roc = []
    y_roc = []
    dirname = "../temp/vae_mmd_data/1024/epilepsiae_all/{}".format(test_patient)
    filenames = ["{}/{}_{}.pickle".format(dirname, test_patient, i) for i in range(pat_hours[test_patient]//2,
                                                                                   pat_hours[test_patient])]

    error = []
    for filename in filenames:
        with open(filename, "rb") as pickle_file:
            data = pickle.load(pickle_file)
            x = np.array(data["X"])
            y = np.array(data["y"])
            if x.shape[0] != 899:
                print("Shape: {} removed".format(x.shape[0]))
                continue
            x = np.pad(x, [(0,1),(0,0), (0,0)], mode= 'edge')
            x = np.reshape(x, (-1, 300, 1024, 2))

            y = np.pad(y, (0,1), mode='edge')
            y = np.reshape(y, (-1, 300))

            for idx in range(x.shape[0]):
                x_in = np.expand_dims(x[idx], axis=0)
                mmd_predict = model.predict(x_in, batch_size = 1)
                max_mmd = np.max(mmd_predict)
                label = 1 if np.sum(y[idx]) > 0 else 0
                x_roc.append(max_mmd)
                y_roc.append(label)

                if label == 1:
                    middle = np.median(np.where(y[idx]>0)[0])
                    err = np.abs(np.argmax(mmd_predict) - middle)
                    error.append(err)
                    print("Error : {}, predict :{}".format(err, np.argmax(mmd_predict)))

    auc = roc_auc_score(y_true=y_roc, y_score=x_roc)
    print("AUC: {}".format(auc))

    return auc, error


def get_data(test_patient):
    x_total = np.zeros((0, 899, 1024, 2))
    y_total = np.zeros(0)
    dirname = "../temp/vae_mmd_data/1024/epilepsiae_all/{}".format(test_patient)
    filenames = ["{}/{}_{}.pickle".format(dirname, test_patient, i) for i in range(1, pat_hours[test_patient] // 2)]

    for filename in filenames:
        with open(filename, "rb") as pickle_file:
            data = pickle.load(pickle_file)
            x = np.array(data["X"])
            if x.shape[0] != 899:
                continue
            x = np.expand_dims(x, 0)
            x_total = np.concatenate((x_total, x))
            y = np.array(data["y"])
            label = 1 if np.sum(y) > 0 else 0
            y_total = np.concatenate((y_total, [label]))

    return x_total, y_total


def self_train(test_patient):
    source_arch = 'vae_sup_chb'
    source_model = 'vae_interval_20min_v24'
    save_path = '../temp/vae_mmd/integrated/{}/{}/{}/model/test_{}/saved_model/'.format(SEG_N,
                                                                                        source_arch,
                                                                                        source_model,
                                                                                        '-1')
    trained_model = tf.keras.models.load_model(save_path)
    # for layer in trained_model.layers[:18]:
    #     print(layer.name)
    #     layer.trainable = False
    auc_array = []
    error_array = []
    auc, error = self_eval(trained_model, test_patient)
    auc_array.append(auc)
    error_array.append(error)

    x, y = get_data(test_patient)
    x_seizure = np.zeros((0, 899, 1024, 2))
    len_seizure = 0
    x_non_seizure = np.zeros((0, 899, 1024, 2))
    y_train = np.zeros((0, 899))
    h = 0
    for x_in, y_in in zip(x,y):
        predict = trained_model.predict(x= np.expand_dims(x_in, 0))
        if np.max(predict) > TH and np.sum(y_in) == 0:
            x_non_seizure = np.concatenate((np.expand_dims(x_in, 0), x_non_seizure))
            print("Shape : {}".format(x_non_seizure.shape))
            new_label = np.zeros(shape=(1, 899))
            y_train = np.concatenate((y_train, new_label))
        elif np.sum(y_in) > 0:
            x_seizure = np.concatenate((np.expand_dims(x_in, 0), x_seizure))
            ratio = 1 / np.max(predict)
            new_label = np.expand_dims(predict[0,:,0] * ratio, 0)
            print("Predict: {} {}".format(predict.shape, new_label.shape))
            y_train = np.concatenate((new_label, y_train))
            print("Ratio :{}, before: {}, after: {}".format(ratio, np.max(predict), np.max(predict) * ratio))
            len_seizure += 1

        x_borrow, y_borrow = borrow_seizure()
        x_train = np.concatenate((x_borrow, x_seizure, x_non_seizure))
        print("Train shape: {}".format(x_train.shape) )
        label_train = np.concatenate((y_borrow, y_train))

        if x_train.shape[0] > 5:
            trained_model.fit(x=x_train, y=label_train, batch_size=1, epochs=4)
        h += 1
        if h%5 == 0:
            auc, error = self_eval(trained_model, test_patient)
            auc_array.append(auc)
            error_array.append(error)
    print("AUC :{}".format(auc_array))


if __name__ == '__main__':
    tf.config.experimental.set_visible_devices([], 'GPU')
    self_train('pat_102')
