import numpy as np
import os
from utils import vae_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics.pairwise import polynomial_kernel
import pickle
import logging
import matplotlib.pyplot as plt
from utils.data import dataset_training
from training import get_all_filenames
from vae_mmd import build_dataset_pickle as test_dataset
from vae_mmd import plot_mmd


LOG = logging.getLogger(os.path.basename(__file__))
ch = logging.StreamHandler()
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch.setFormatter(logging.Formatter(log_fmt))
LOG.addHandler(ch)
LOG.setLevel(logging.INFO)

SF = 256
SEG_LENGTH = 1024
EXCLUDED_SIZE = 15
interval_len = 4


def main():
    arch = 'vae_unsupervised'
    beta = 1e-05
    latent_dim = 16
    lr = 0.0001
    decay = 0.5
    gamma = 0.0

    root = "../output/vae/{}/".format(arch)
    stub = "seg_n_{}/beta_{}/latent_dim_{}/lr_{}/decay_{}/gamma_{}/test_{}/saved_model/"
    build_model = vae_model.build_model
    build_model_args = {
        "input_shape": (SEG_LENGTH, 2,),
        "enc_dimension": latent_dim,
        "beta": beta,
        "gamma": 0,
        "optim": Adam(lr),
        "FS": SF
    }

    model, encoder = build_model(**build_model_args)

    print(encoder.summary())

    subdirname = "../temp/vae_mmd/integrated/{}/{}/frozen_v3_gru_free".format(SEG_LENGTH, arch)
    if not os.path.exists(subdirname):
        os.makedirs(subdirname)

    middle_diff = []
    all_filenames = get_all_filenames()
    for test_patient in range(1,25):
        train_data, train_label = dataset_training("train", test_patient, all_filenames)
        val_data, val_label = dataset_training("valid", test_patient, all_filenames)

        sessions = test_dataset(test_patient)

        # Load the specific weights for the model
        load_dirname = root + stub.format(SEG_LENGTH, beta, latent_dim, lr, decay, gamma, test_patient)
        if not os.path.exists(load_dirname):
            print("Model does not exist in {}".format(load_dirname))
            exit()
        model.load_weights(load_dirname)

        vae_mmd_model = vae_model.get_mmd_model(state_len=300, latent_dim=latent_dim, signal_len=SEG_LENGTH, trainable_vae=False)

        print(vae_mmd_model.summary())
        for layer_num in range(13):
            weights = encoder.layers[layer_num].get_weights()
            vae_mmd_model.layers[layer_num].set_weights(weights)

        print("input shape: {}".format(train_data.shape))
        train_random = np.random.randn(train_data.shape[0], train_data.shape[1], 16)
        val_random = np.random.randn(val_data.shape[0], val_data.shape[1], 16)
        vae_mmd_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        vae_mmd_model.fit(x=[train_data, train_random], y=train_label,
                          validation_data=([val_data, val_random], val_label), batch_size=1, epochs=50)

        for node in sessions.keys():   # Loop2: nodes in the dataset
            # print("node: {}".format(node))
            patient_num = int(node[3:5])
            if test_patient != patient_num:
                continue

            LOG.info("session name: {}".format(test_patient))
            X = sessions[node]['data']
            LOG.info("session number: {}".format(len(X)))
            y_true = sessions[node]['label']

            if np.sum(y_true) == 0:
                continue

            X = np.expand_dims(X, 0)
            # y = np.expand_dims(y_true, -1)
            # y = np.expand_dims(y, 0)
            input_random = np.random.randn(X.shape[0], X.shape[1], 16)
            mmd_predicted = vae_mmd_model.predict([X, input_random])
            print("Predict shape: {}".format(mmd_predicted.shape))
            mmd_maximum = [np.argmax(mmd_predicted)]
            plot_mmd(mmd_predicted[0,:,0], mmd_maximum, y_true, node, subdirname)

            y_non_zero = np.where(y_true > 0, 1, 0)
            y_diff = np.diff(y_non_zero)
            start_points = np.where(y_diff > 0)[0]
            stop_points = np.where(y_diff < 0)[0]
            middle_points = (start_points + stop_points) // 2
            LOG.info("points: {}, {}".format(start_points, stop_points))

            t_diff = np.abs(middle_points - mmd_maximum[0])
            LOG.info("Time diff : {}".format(t_diff))
            middle_diff.append(np.min(t_diff))

    # with open("../output/z_16.pickle", "wb") as pickle_file:
    #     pickle.dump(z_dict, pickle_file)

    print(middle_diff)
    plt.figure()
    plt.hist(middle_diff)
    plt.savefig("{}/hist_diff.png".format(subdirname))
    plt.show()


if __name__ == "__main__":
    # tf.config.experimental.set_visible_devices([], 'GPU')
    main()
