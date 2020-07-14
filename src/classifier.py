from  sklearn.ensemble import RandomForestClassifier
from utils import create_seizure_dataset
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

SF = 256
SEG_LENGTH = 1024
patient_samples = [3672, 1052, 3713, 2393, 2849, 7901, 1511, 2729, 3193, 3371]
patient_index = [0, 3672,  4724,  8437, 10830, 13679, 21580, 23091, 25820, 29013, 32384]


def prepare_data(Z1, mmd, label, idx):
    start_index = patient_index[idx]
    end_index = patient_index[idx+1]
    Z1_test = Z1[start_index:end_index]
    mmd_test = mmd[start_index:end_index]
    X_test = np.concatenate((Z1_test, np.expand_dims(mmd_test, axis=1)), axis=1)
    label_test = label[start_index: end_index]

    Z1_train = Z1[:start_index]
    Z1_train = np.concatenate((Z1_train, Z1[end_index:]))
    mmd_train = mmd[:start_index]
    mmd_train = np.concatenate((mmd_train, mmd[end_index:]))
    X_train = np.concatenate((Z1_train, np.expand_dims(mmd_train,axis=1)), axis=1)
    label_train = label[:start_index]
    label_train = np.concatenate((label_train, label[end_index:]))

    return X_train, label_train, X_test, label_test


def main():
    data = pickle.load(open("z1.pickle", "rb"))
    Z1, mmd, label = data["X"], data["y"], data["label"]
    for idx in range(10):  # cross_validation
        X_train, y_train , X_test, y_test = prepare_data(Z1, mmd, label, idx)
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predict)
        f1 = f1_score(y_test, predict)
        print("Patient {}: Accuracy: {:.3f}, F1 score: {:.3f}".format(idx, accuracy, f1))


if __name__ == '__main__':
    main()
