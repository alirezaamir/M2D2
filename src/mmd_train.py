import numpy as np
import os
from utils import vae_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from sklearn.metrics.pairwise import polynomial_kernel
import pickle, time
import logging
import matplotlib.pyplot as plt
from utils.data import dataset_training, get_non_seizure_signal, get_epilepsiae_seizures, get_epilepsiae_test, \
    get_new_conv_w, get_epilepsiae_non_seizure, get_seizure_point_from_label
from utils.data import build_dataset_pickle as test_dataset
from training import get_all_filenames
from vae_mmd import plot_mmd
from utils.params import pat_list
from utils.data23 import get_balanced_data, get_test_data, get_test_overlapped, get_non_seizure
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib
from utils.quantization import my_quantized_model, scaled_model


LOG = logging.getLogger(os.path.basename(__file__))
ch = logging.StreamHandler()
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch.setFormatter(logging.Formatter(log_fmt))
LOG.addHandler(ch)
LOG.setLevel(logging.INFO)

SEG_LENGTH = 1024
SEQ_LEN = 899
STATE_LEN = 899
LATENT_DIM = 32
NUM_FRACTIONS_B = 5
NUM_FRACTIONS_W = 8
BITS = 8


def train_model():
    arch = '23channel'
    subdirname = "../temp/vae_mmd/integrated/{}/{}/FCN_pre_pruned".format(SEG_LENGTH, arch)
    if not os.path.exists(subdirname):
        os.makedirs(subdirname)

    for test_id in [1, 21, 22]:  # ["-1"]:  # range(30):  # range(1,24):
        # load the model
        vae_mmd_model = vae_model.get_FCN_model(state_len=STATE_LEN, latent_dim=LATENT_DIM, signal_len=SEG_LENGTH,
                                                seq_len=None, trainable_vae=True)

        print(vae_mmd_model.summary())

        vae_mmd_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy', metrics='accuracy')

        savedir = '{}/model/test_{}/saved_model/'.format(subdirname, test_id)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        for iter in range(200):
            try:
                train_data, train_label = get_balanced_data(test_id, ictal_ratio=0.03, inter_ratio=0.02, non_ratio=0.02)
                train_data = np.clip(train_data, a_min=-250, a_max=250)
                train_data = train_data / 250
                vae_mmd_model.fit(x=train_data, y=tf.keras.utils.to_categorical(train_label, 2), batch_size=32,
                                  initial_epoch= iter*5, epochs=(iter+1)*4)
                del train_data, train_label
                # test_data, test_label = get_test_data(test_id)
                # eval = vae_mmd_model.evaluate(x=test_data, y=tf.keras.utils.to_categorical(test_label))
                train_data = None
            except MemoryError:
                print("Memory Error")
                continue

        vae_mmd_model.save(savedir)


def retrain_model():
    arch = '23channel'
    subdirname = "../temp/vae_mmd/integrated/{}/{}/FCN_v1".format(SEG_LENGTH, arch)

    for test_id in range(22, 23):  # ["-1"]:  # range(30):  # range(1,24):
        load_dir = '{}/model/test_{}/saved_model/'.format(subdirname, test_id)
        vae_mmd_model = tf.keras.models.load_model(load_dir)
        print(vae_mmd_model.summary())

        scaled_model(vae_mmd_model)
        for trainable_index in [1, 3, 6, 8, 11, 13, 17]:
            print("Conv1d Before Quantization : {}".format(np.mean(vae_mmd_model.layers[6].get_weights()[0])))
            vae_mmd_model = my_quantized_model(vae_mmd_model, NUM_FRACTIONS_W, NUM_FRACTIONS_B, BITS)
            print("Conv1d After Quantization : {}".format(np.mean(vae_mmd_model.layers[6].get_weights()[0])))
            for layers_behind in range(trainable_index+1):
                vae_mmd_model.layers[layers_behind].trainable = False
            time.sleep(10)
            vae_mmd_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
                                  loss='categorical_crossentropy',
                                  metrics='accuracy')
            # for layer in vae_mmd_model.layers:
            #     print("Layer {}, Trainable {}".format(layer.name, layer.trainable))
            for _ in range(15):
                try:
                    train_data, train_label = get_balanced_data(test_id, ictal_ratio=0.03, inter_ratio=0.03,
                                                                non_ratio=0.02)
                    train_data = np.clip(train_data, a_min=-250, a_max=250) / 250.0
                    vae_mmd_model.fit(x=train_data, y=tf.keras.utils.to_categorical(train_label, 2), batch_size=32,
                                      epochs=2)
                    del train_data, train_label
                    train_data = None
                except MemoryError:
                    print("Memory Error")
                    time.sleep(5)
                    train_data = None
                    time.sleep(5)

        save_dir = '{}/model/q_{}_test_{}_v3/saved_model/'.format(subdirname, BITS, test_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        vae_mmd_model.save(save_dir)


def inference(test_patient:int, trained_model, subdirname:str, dataset='CHB'):
    """
    To evaluate the model on the test dataset
    :param test_patient: int, number of test patient. Ex, 1 -> CHB_01
    :param trained_model: TF model, trained on the training set, if None, the method will take a saved model
    :param subdirname: str, If TF model is not passed, the method takes a saved model from this address
    :param dataset: str, 'CHB' or 'Epilepsiae'
    :return: int[], distances from the seizure for every sessions
    """

    X_test, y_test = get_test_overlapped(test_patient)
    X_test = np.clip(X_test, a_min=-250, a_max=250)
    X_test = X_test/250

    # Load the trained model
    if trained_model is None:
        save_path = '{}/model/test_{}/saved_model/'.format(subdirname, test_patient)
        trained_model = tf.keras.models.load_model(save_path)
        vae_mmd_model = trained_model
    else:
        vae_mmd_model = trained_model

    # Evaluate for every sessions in the test dataset

    predicted = vae_mmd_model.predict(X_test)
    y_pred = np.argmax(predicted, axis=1)
    conf_mat = confusion_matrix(y_test, y_pred)
    print("Pat : {}\n Conf Mat : {}".format(test_patient, conf_mat))
    print("F1-score : {}".format(f1_score(y_test, y_pred)))
    f1 = [91.5, 95.5, 64, 49.5, 72.3, 28.1, 84.9, 35.6, 49.3, 80.5, 87.6, 7.1, 19.7, 0, 44.7, 22.5, 81.3, 47.1, 11.3, 37.8, 1.2, 68, 53.2]
    f1 = [72.03, 56.4, 33.5, 32.1,35.2, 0, 32.2, 21.2, 18.85, 76.84, 58.86, 7.14, 12.2, 0, 0, 4.9, 83.3, 71.2, 68.1, 7.14, 32.9, 21.9, 8.6]

    return conf_mat, f1_score(y_test, y_pred)


def get_results():
    """
    This method is only for evaluation a saved model
    """
    arch = '23channel'
    subdirname = "../temp/vae_mmd/integrated/{}/{}/FCN_pre_pruned".format(SEG_LENGTH, arch)
    conf_mat = [[0, 0],[0, 0]]
    f1_scores =[]
    for pat_id in [1,21,22]:
        pat = pat_id
        pat_conf_mat, f1 = inference(pat, None, subdirname, dataset='CHB')
        conf_mat = np.add(conf_mat, pat_conf_mat)
        f1_scores.append(f1)
    print("Total Confusion matrix: {}".format(conf_mat))
    print("F1 scores : {}".format(f1_scores))


def across_dataset():
    source_arch = 'vae_free'
    source_model = 'Epilepsiae_v62'
    subdirname = "../temp/vae_mmd/integrated/{}/across/from_{}/{}".format(SEG_LENGTH, source_arch, source_model)
    if not os.path.exists(subdirname):
        os.makedirs(subdirname)
    diffs = []
    nc = {}
    save_path = '../temp/vae_mmd/integrated/{}/{}/{}/model/test_{}/saved_model/'.format(SEG_LENGTH,
                                                                                        source_arch,
                                                                                        source_model,
                                                                                        'pat_11002')
    trained_model = tf.keras.models.load_model(save_path)
    for pat_id in range(1,24):
        pat = pat_id
        # pat = pat_list[pat_id]
        diff_pat = inference(pat, trained_model, subdirname, dataset='CHB')
        diffs += diff_pat
    print("Differences: {}\nMedian: {}\nMean: {}".format(diffs, np.median(diffs), np.mean(diffs)))
    diffs_minute = [x / 15.0 for x in diffs]
    plt.figure()
    plt.hist(diffs_minute, bins=150, range=(0, 200))
    plt.savefig("{}/hist_diff_{}.png".format(subdirname, SEQ_LEN))


if __name__ == "__main__":
    # tf.config.experimental.set_visible_devices([], 'GPU')
    # train_model()
    get_results()
    # across_dataset()
    # retrain_model()

