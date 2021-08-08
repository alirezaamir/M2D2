import os
import sys
import logging
import numpy as np
from utils import vae_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics.pairwise import polynomial_kernel
import matplotlib.pyplot as plt
from utils.params import pat_list
from utils.data import get_epilepsiae_test, build_dataset_pickle
from utils.gru_visualization import get_within_between

sys.path.append("../")
LOG = logging.getLogger(os.path.basename(__file__))
ch = logging.StreamHandler()
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch.setFormatter(logging.Formatter(log_fmt))
LOG.addHandler(ch)
LOG.setLevel(logging.INFO)

SF = 256
SEG_LENGTH = 1024
EXCLUDED_SIZE = 15
interval_len = 6


def get_interval_mmd(K_mat):
    mmd = []
    N = 2 * interval_len
    input_length = K_mat.shape[0]
    for index in range(interval_len, input_length - interval_len):
        M = input_length - N
        start = index - interval_len
        end = index + interval_len
        Kxx = K_mat[start: end, start: end].sum()
        Kxy = K_mat[start:end, :start].sum() + K_mat[start:end, end:].sum()
        Kyy = K_mat[:start, :start].sum() + K_mat[end:, end:].sum() + 2 * K_mat[end:, :start].sum()
        mmd.append(np.sqrt(
            ((1 / float(N * N)) * Kxx) +
            ((1 / float(M * M)) * Kyy) -
            ((2 / float(N * M)) * Kxy)
        ))
    return np.array(mmd)


def get_simplified_mmd(K_mat):
    mmd = []
    N = 2 * interval_len
    input_length = K_mat.shape[0]
    for index in range(interval_len, input_length - interval_len):
        M = input_length - N
        start = index - interval_len
        end = index + interval_len
        Kxx = K_mat[start: end, start: end].sum()
        Kxy = K_mat[start:end, :].sum()
        mmd.append(
            ((1 / float(N * N)) * Kxx) -
            ((2 / float(N * M)) * Kxy))
    return np.array(mmd)


def find_original_index(original_index_list, new_index, K_len):
    if len(original_index_list) == 0 :
        return new_index

    total_array = np.arange(K_len, dtype=np.int)
    complementary_array = [i for i in total_array if i not in original_index_list]

    original_index = complementary_array[new_index]
    return original_index


def get_changing_points(K_mat, steps):
    removed_list = []
    changing_point_list = []
    initial_mmd = get_interval_mmd(K_mat)
    original_len = K_mat.shape[0]
    # initial_mmd_corr = get_mmd_corrected(initial_mmd)
    for step in range(steps):
        mmd = get_interval_mmd(K_mat)
        # mmd_corr = get_mmd_corrected(mmd)
        arg_max_mmd = np.argmax(mmd)
        original_index = find_original_index(removed_list, arg_max_mmd, original_len)
        changing_point_list.append(original_index)

        start = np.max((0, arg_max_mmd - EXCLUDED_SIZE))
        end = np.min((K_mat.shape[0], arg_max_mmd + EXCLUDED_SIZE))

        # remove rows and columns in K_mat from start to end
        changing_interval = np.arange(start, end, dtype=np.int)
        K_mat = np.delete(K_mat, changing_interval, 0)
        K_mat = np.delete(K_mat, changing_interval, 1)

        # add the removed interval to the list for finding the original point in the next step
        original_interval = changing_interval + (original_index - arg_max_mmd)
        removed_list = np.concatenate((removed_list, original_interval))

    return changing_point_list, initial_mmd


def plot_mmd(mmd, argmax_mmd, y_true, name, dir):

    y_non_zero = np.where(y_true > 0, 1, 0)
    y_diff = np.diff(y_non_zero)
    start_points = np.where(y_diff > 0)[0]
    stop_points = np.where(y_diff < 0)[0]

    x = np.linspace(0, y_true.shape[0]*4, y_true.shape[0])
    plt.figure()
    plt.plot(x, mmd, label="MMD")
    for seizure_start, seizure_stop in zip(start_points, stop_points):
        plt.axvspan(x[seizure_start], x[seizure_stop], color='r', alpha=0.5)
    for point_idx, max_point in enumerate(argmax_mmd):
        plt.plot(x[max_point], mmd[max_point], 'go', markersize=18)
        # plt.text(x[max_point-(len(mmd)*0.02), mmd[max_point]-(mmd[argmax_mmd[0]] * 0.025),
        #          str(point_idx+1), fontsize=18, fontweight='bold')
    plt.xlabel("Time (second)")
    plt.ylabel("MMD")
    plt.title("Seizure detection for patient {}".format(name))
    plt.savefig("{}/{}.png".format(dir, name))
    plt.close()


def main():
    dirname = "../temp/vae_mmd"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    source_arch = 'vae_unsup_chb'
    test_arch = 'Epilepsiae_un'
    beta = 1e-05
    latent_dim = 128
    lr = 0.0001
    decay = 0.02
    gamma = 0.0

    root = "../output/vae/{}/".format(source_arch)
    stub = "seg_n_{}/beta_{}/latent_dim_{}/lr_{}/decay_{}/gamma_{}/test_{}/saved_model/"
    build_model = vae_model.build_VAE_model
    build_model_args = {
        "input_shape": (SEG_LENGTH, 2,),
        "enc_dimension": latent_dim,
        "beta": beta,
        "gamma": 0,
        "optim": Adam(lr),
        "FS": SF
    }

    model, _ = build_model(**build_model_args)

    kernel = polynomial_kernel
    kernel_name = "polynomial_seizures"
    dirname = "../temp/vae_mmd/{}".format(kernel_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    subdirname = "{}/{}/{}".format(dirname, SEG_LENGTH, test_arch)
    if not os.path.exists(subdirname):
        os.makedirs(subdirname)

    # sessions2 = create_seizure_dataset(SEG_LENGTH, SF)
    middle_diff = []
    z_dict = {}
    # for test_patient in range(1,25):
    J_list = {}
    for pat_id in pat_list:#range(1,25):
        # test_patient = pat_list[pat_id]
        source_pat = -1
        sessions = get_epilepsiae_test(test_patient=pat_id)
        # sessions = (test_patient=test_patient)

        # Load the specific weights for the model
        dirname = root + stub.format(SEG_LENGTH, beta, latent_dim, lr, decay, gamma, source_pat)
        if not os.path.exists(dirname):
            print("Model does not exist in {}".format(dirname))
            exit()
        model.load_weights(dirname)
        intermediate_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[1].output)

        for node in sessions.keys():   # Loop2: nodes in the dataset
            # print("node: {}".format(node))
            # patient_num = int(node[3:5])
            # if test_patient != patient_num:
            #     continue

            LOG.info("session name: {}".format(pat_id))
            X = sessions[node]['data']
            LOG.info("session number: {}".format(len(X)))
            y_true = sessions[node]['label']

            if np.sum(y_true) == 0:
                continue

            LOG.info("Session {}\nmean:{}, std: {}".format(node, np.mean(X), np.std(X)))

            latent = intermediate_model.predict(X)[2]
            z_dict[node] = latent
            print("Z space {} : {}".format(node, latent.shape))
            print("y true: {}".format(y_true.shape))
            Sb, Sw, J = get_within_between(latent, y_true)
            J_list[node] = J
            continue

            K = kernel(latent)
            mmd_maximum, mmd = get_changing_points(K, 1)
            LOG.info("mmd maximum : {}".format(mmd_maximum))
            print("MMD shape: {}\ny shape:{}".format(mmd.shape, y_true.shape))
            y_true = y_true[interval_len: -interval_len]
            plot_mmd(mmd, mmd_maximum, y_true, node, subdirname)

            y_non_zero = np.where(y_true > 0, 1, 0)
            y_non_zero = np.concatenate((y_non_zero, [0]))
            # For sections which have seizure at the end or start of the section
            y_non_zero = np.concatenate(([0], y_non_zero,))

            y_diff = np.diff(y_non_zero)
            start_points = np.where(y_diff > 0)[0]
            stop_points = np.where(y_diff < 0)[0]
            LOG.info("points: {}, {}".format(start_points, stop_points))
            accepted_points = []
            for start, stop in zip(start_points, stop_points):
                accepted_points += range(start, stop)
            middle_points = (start_points + stop_points) // 2
            LOG.info("start: {}, stop: {}\npoints: {}".format(start_points, stop_points, accepted_points))

            t_diff = np.abs(accepted_points - mmd_maximum[0])

            LOG.info("Time diff : {}".format(t_diff))
            if len(t_diff) >0 :
                middle_diff.append(np.min(t_diff))

    # with open("../output/z_16.pickle", "wb") as pickle_file:
    #     pickle.dump(z_dict, pickle_file)

    print("J s: {}".format(J_list))
    print(middle_diff)
    plt.figure()
    plt.hist(middle_diff)
    plt.savefig("{}/hist_diff.png".format(subdirname))
    plt.show()


class PrintLogs(tf.keras.callbacks.Callback):
    def __init__(self, epochs):
        self.epochs = epochs

    def set_params(self, params):
        params['epochs'] = 0

    def on_epoch_begin(self, epoch, logs=None):
        print('Epoch %d/%d: ' % (epoch + 1, self.epochs), end='')


if __name__ == "__main__":
    tf.config.experimental.set_visible_devices([], 'GPU')
    main()
