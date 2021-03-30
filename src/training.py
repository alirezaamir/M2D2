import os
import sys
import logging
import matplotlib
import json

matplotlib.use("Agg")
sys.path.append("../")

import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras.backend as K

from utils.params import SEG_N
from utils import vae_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import Callback, EarlyStopping, CSVLogger, LearningRateScheduler
from utils.params import pat_list
np.random.seed(13298)

LOG = logging.getLogger(os.path.basename(__file__))
ch = logging.StreamHandler()
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch.setFormatter(logging.Formatter(log_fmt))
ch.setLevel(logging.INFO)
LOG.addHandler(ch)
LOG.setLevel(logging.INFO)

FS = 256.0
PART_NUM = 2


def main():
    arch = sys.argv[1]
    beta = float(sys.argv[2])
    latent_dim = int(sys.argv[3])
    lr = float(sys.argv[4])
    decay = float(sys.argv[5])
    gamma = float(sys.argv[6])
    test_patient_id = int(sys.argv[7])

    # test_patient = pat_list[test_patient_id] if test_patient_id != -1 else test_patient_id
    test_patient = str(test_patient_id)

    param_str = """
    ==========================
        Arch:           {}
        Beta:           {}
        Decay:          {}
        Encoder Dim:    {}
        Learning Rate:  {}
    ==========================""".format(arch, beta, decay, latent_dim, lr)
    LOG.info("Training Model with parameters:{}".format(param_str))

    build_model = vae_model.build_ae_model
    root = "../output/vae/{}".format(arch)
    stub = "/seg_n_{}/beta_{}/latent_dim_{}/lr_{}/decay_{}/gamma_{}/test_{}"
    dirname = root + stub.format(SEG_N, beta, latent_dim, lr, decay, gamma, test_patient)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    # else:
    #     LOG.warning("Model already trained...")
    #     return

    beta = K.variable(beta)
    build_model_args = {
        "input_shape": (SEG_N, 2,),
        "enc_dimension": latent_dim,
        "beta": beta,
        "gamma": gamma,
        "optim": Adam(lr),
        "FS": FS
    }

    model, _ = build_model(**build_model_args)
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    LOG.info("Model Summary:")
    LOG.info("\n".join(model_summary))

    train_model(model, dirname, lr, decay, beta, test_patient)


def train_model(model, dirname, lr_init, decay, beta, test_patient):
    max_epochs = 5  # 200
    patience = 20
    batch_size = 32
    beta_start_epoch = 10

    history = CSVLogger(dirname + "/training.log")
    early_stopping = EarlyStopping(
        monitor="loss", patience=patience, restore_best_weights=True)
    scheduler = LearningRateScheduler(lambda x, y: lr_init * np.exp(-decay * x))
    beta_annealing = AnnealingCallback(beta, beta_start_epoch, max_epochs)

    all_filenames = get_all_filenames(entire_dataset=False)
    print(all_filenames["train"][test_patient])
    savedir = dirname + '/saved_model/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    get_dataset = build_dataset_chb
    savedir = dirname + '/saved_model/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for iter in range(12):
        train_data, train_label = get_dataset("train", test_patient, all_filenames)
        print("Shape :{}, {}".format(train_data.shape, train_label.shape))
        valid_data, valid_label = get_dataset("valid", test_patient, all_filenames)
        print("Shape :{}, {}".format(valid_data.shape, valid_label.shape))
        model.fit(x=[train_data, train_label], validation_data=[valid_data, valid_label],
                  initial_epoch=iter * max_epochs,
                  epochs=(iter + 1) * max_epochs,
                  batch_size=batch_size,
                  shuffle=True,
                  callbacks=[early_stopping, history, scheduler, beta_annealing]
        )
        model.save_weights(savedir, save_format='tf')


def build_dataset_chb(mode, test_patient, all_filenames):
    # filenames = split_list(all_filenames[mode][test_patient], wanted_parts=PART_NUM)
    filenames = np.random.permutation(all_filenames[mode][test_patient])
    number_files = len(filenames) // PART_NUM if mode == "train" else len(filenames)
    data_len = get_data_len(filenames[:number_files])
    X_total = np.zeros((data_len, SEG_N, 2))
    y_total = np.zeros((data_len,))

    last_pointer = 0
    # patient_dict = {pat_num: 0 for pat_num in range(1, 25)}
    for filename in filenames[:number_files]:
        with open(filename, "rb") as pickle_file:
            data = pickle.load(pickle_file)
            file_data_len = np.array(data["X"]).shape[0]
            X_total[last_pointer: last_pointer + file_data_len] = np.array(data["X"])
            y_total[last_pointer: last_pointer + file_data_len] = np.array(data["y"])
            last_pointer += file_data_len

    return X_total, y_total


def build_dataset_epilepsiae(mode, test_patient, _):
    X_total = np.zeros(shape=(0, SEG_N, 2))
    y_total = np.zeros(shape=(0, ))
    for label in ['non_seizure', 'seizure']:
        dirname = "../temp/vae_mmd_data/{}/epilepsiae_{}/{}".format(SEG_N, label, mode)
        filenames = ["{}/{}".format(dirname, x) for x in os.listdir(dirname) if not x.startswith("{}".format(test_patient))]
        print(filenames)
        for filename in filenames:
            with open(filename, "rb") as pickle_file:
                data = pickle.load(pickle_file)
                X_total = np.concatenate((X_total, np.array(data["X"])))
                y_total = np.concatenate((y_total, np.array(data["y"])))

    return X_total, y_total


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts)]


def get_data_len(filenames):
    pat_data_len = {"train": {1: 31812, 2: 29906, 3: 31467, 4: 129617, 5: 29670, 6: 58064, 7: 56730, 8: 17986, 9: 46872,
                              10: 41395, 11: 29482, 12: 16801, 13: 26970, 14: 21576, 15: 33273, 16: 16182, 17: 17087,
                              18: 28440, 19: 24210, 20: 21251, 21: 26818, 22: 26970, 23: 19658, 24: 18246},
                    "valid": {1: 4644, 2: 1798, 3: 2697, 4: 10804, 5: 5394, 6: 1981, 7: 3599, 8: 1798, 9: 14197,
                              10: 3603,
                              11: 1798, 12: 4501, 13: 2697, 14: 1798, 15: 2698, 16: 899, 17: 1798, 18: 3596, 19: 2697,
                              20: 3564, 21: 2697, 22: 902, 23: 4238, 24: 899}}
    # total = sum(pat_data_len[mode].values())
    # print("Total number : {}".format(total))
    total_len = 0
    with open("../input/file_len.json", "r") as json_file:
        file_data_json = json.load(json_file)
        for dirname in filenames:
            filename = dirname.split('/')[-1]
            length = file_data_json[filename]
            total_len += length
    print("Length: {}".format(total_len))
    return total_len


def get_all_filenames(entire_dataset=False):
    all_filenames = {'train': {}, 'valid': {}}
    for mode in 'train', 'valid':
        dirname = "../temp/vae_mmd_data/{}/full_normal/{}".format(SEG_N, mode)
        if entire_dataset:
            filenames = ["{}/{}".format(dirname, x) for x in os.listdir(dirname)]
            all_filenames[mode]['-1'] = filenames
            continue
        else:
            for test_patient in range(1, 25):
                filenames = ["{}/{}".format(dirname, x) for x in os.listdir(dirname) if not
                x.startswith("chb{:02d}".format(test_patient))]
                all_filenames[mode][str(test_patient)] = filenames
    return all_filenames


def build_dataset_tfrecord(mode, batch_size, test_patient):
    dirname = "../temp/vae_mmd_data/{}/LOOCV/{}".format(SEG_N, mode)
    filenames = ["{}/{}".format(dirname, x) for x in os.listdir(dirname) if
                 not x.startswith("chb{:02d}".format(test_patient))]
    print("Files: {}".format(filenames))
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_fn)
    dataset = dataset.shuffle(4096).batch(batch_size)
    return dataset


def _parse_fn(proto):
    parse_dict = {"channels": tf.io.FixedLenFeature([], tf.string),
                  "label": tf.io.FixedLenFeature([], tf.string)}
    example = tf.io.parse_single_example(proto, parse_dict)
    X = tf.io.parse_tensor(example["channels"], out_type=tf.float32)
    y = tf.io.parse_tensor(example["label"], out_type=tf.float32)
    X = tf.reshape(X, [SEG_N, 2])  # Annoying hack needed for Keras
    return X, y


class AnnealingCallback(Callback):

    def __init__(self, beta, beta_start_epoch=3, max_epochs=200):

        self.beta = beta
        self.beta_start_epoch = beta_start_epoch
        self.max_epochs = max_epochs
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.beta_start_epoch:
            beta_new = min(
                K.get_value(self.beta) + (1 / float(self.max_epochs)), 1.0)
            K.set_value(self.beta, beta_new)


if __name__ == "__main__":
    # tf.config.experimental.set_visible_devices([], 'GPU')
    main()
