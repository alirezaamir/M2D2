import os
import sys
import tables
import logging
import numpy as np
from scipy import signal, integrate
import vae_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.append("../")
LOG = logging.getLogger(os.path.basename(__file__))
ch = logging.StreamHandler()
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch.setFormatter(logging.Formatter(log_fmt))
LOG.addHandler(ch)
LOG.setLevel(logging.INFO)

SF = 256
SEG_LENGTH = 512
EXCLUDED_SIZE = 64


def get_PCA(x):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    return principalComponents


def get_TSNE(x):
    tsne = TSNE(n_components=2)
    x_embedded = tsne.fit_transform(x)
    return x_embedded


def main():
    dirname = "../temp/vae_mmd"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    arch = 'psd_non_single_seizure'
    beta = 0.001
    latent_dim = 16
    lr = 0.0001
    decay = 0.0
    gamma = 5000000.0

    root = "../output/vae/{}/".format(arch)
    stub = "seg_n_{}/beta_{}/latent_dim_{}/lr_{}/decay_{}/gamma_{}"
    dirname = root + stub.format(SEG_LENGTH, beta, latent_dim, lr, decay, gamma)
    build_model = vae_model.build_model
    build_model_args = {
        "input_shape": (SEG_LENGTH, 2,),
        "enc_dimension": latent_dim,
        "beta": beta,
        "gamma": 0,
        "optim": Adam(lr),
        "FS": SF
    }

    model, _ = build_model(**build_model_args)
    for layer in model.layers:
        print(layer.name)
        # print(layer.output.shape)

    if not os.path.exists(dirname):
        print("Model does not exist")
        exit()
    model.load_weights(dirname)
    intermediate_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[1].output)
    kernel = polynomial_kernel
    kernel_name = "polynomial_seizures"
    dirname = "../temp/vae_mmd/{}".format(kernel_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    subdirname = "{}/single_seizure".format(dirname)
    if not os.path.exists(subdirname):
        os.makedirs(subdirname)

    middle_diff = []
    interval_diff = []
    interval_diff_normal = []

    with tables.open_file("../input/eeg_data_temples2.h5") as h5_file:
        for node in h5_file.walk_nodes("/", "CArray"):
            # LOG.info("Processing: {}".format(node._v_name))
            if len(node.attrs.seizures) != 1:
                continue

            data = node.read()
            X, y = data[:, :-1], data[:, -1]
            if len(node.attrs.seizures) < 1:
                start = X.shape[0]//2
                stop = X.shape[0]//2
            else:
                start = np.min(np.where(y > 0)[0])
                stop = np.max(np.where(y > 0)[0])

            buff_mins = 20
            minv = max(0, start - (buff_mins * 60 * SF))
            maxv = min(X.shape[0], stop + (buff_mins * 60 * SF))
            X = X[minv:maxv, :]
            y = y[minv:maxv]

            sos = signal.butter(3, 50, fs=SF, btype="lowpass", output="sos")
            X = signal.sosfilt(sos, X, axis=1)
            Z = []
            q = []
            for ix in range(SEG_LENGTH, X.shape[0], SEG_LENGTH):
                Z.append(X[ix - SEG_LENGTH:ix, :])
                q.append(np.any(y[ix - SEG_LENGTH:ix]))
            Z = np.array(Z)
            y = np.array(q)

            y_non_zero = np.where(y > 0, 1, 0)
            y_diff = np.diff(y_non_zero)
            start_points = np.where(y_diff > 0)[0]
            stop_points = np.where(y_diff < 0)[0]
            middle_points = (start_points + stop_points) // 2
            LOG.info("points: {}, {}".format(start_points, stop_points))

            latent = intermediate_model.predict(Z)[2]
            sigma = intermediate_model.predict(Z)[1]
            mu = intermediate_model.predict(Z)[0]
            mean_sigma = np.mean(sigma, axis=1)
            mean_mu = np.mean(mu, axis=1)

            plt.figure()
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
            plt.subplot(211)
            plt.plot(mean_sigma)
            plt.title('Sigma')
            for seizure_start, seizure_stop in zip(start_points, stop_points):
                plt.axvspan(seizure_start, seizure_stop, color='r', alpha=0.5)
            plt.subplot(212)
            plt.plot(mean_mu)
            plt.title('Mu')
            for seizure_start, seizure_stop in zip(start_points, stop_points):
                plt.axvspan(seizure_start, seizure_stop, color='r', alpha=0.5)
            plt.savefig("{}/{}_sigma.png".format(subdirname, node._v_name))
            plt.close()

            components = get_PCA(latent)

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('Principal Component 1', fontsize=15)
            ax.set_ylabel('Principal Component 2', fontsize=15)
            ax.set_title('2 component PCA', fontsize=20)
            targets = [0, 1]
            colors = ['r', 'b']
            for target, color in zip(targets, colors):
                indicesToKeep = y == target
                ax.scatter(components[indicesToKeep, 0]
                           , components[indicesToKeep, 1]
                           , c=color)
            ax.legend(targets)
            ax.grid()
            fig.savefig("{}/{}_PCA.png".format(subdirname, node._v_name))
            plt.close(fig)

            K = kernel(latent)
            mmd = []
            for N in range(1, Z.shape[0]):
                M = Z.shape[0] - N
                Kxx = K[:N, :N].sum()
                Kxy = K[:N, N:].sum()
                Kyy = K[N:, N:].sum()
                mmd.append(np.sqrt(
                    ((1 / float(N * N)) * Kxx) +
                    ((1 / float(M * M)) * Kyy) -
                    ((2 / float(N * M)) * Kxy)
                ))

            ws = []
            mmd = np.array(mmd)
            mmd_corr = np.zeros(mmd.size)
            N = Z.shape[0] - 1
            for ix in range(1, mmd_corr.size):
                w = (N / float(ix * (N - ix)))
                ws.append(w)
                mmd_corr[ix] = mmd[ix] - w * mmd.max()

            arg_max_mmd = np.argmax(mmd_corr[EXCLUDED_SIZE:-EXCLUDED_SIZE]) + EXCLUDED_SIZE
            plt.figure()
            plt.plot(mmd_corr, label="MMD")
            for seizure_start, seizure_stop in zip(start_points, stop_points):
                plt.axvspan(seizure_start, seizure_stop, color='r', alpha=0.5)
            if len(node.attrs.seizures) > 0:
                plt.plot(arg_max_mmd, mmd_corr[arg_max_mmd], 'go', markersize=12)
            plt.savefig("{}/{}_mmd_corrected.png".format(subdirname, node._v_name))
            plt.close()

            plt.figure()
            plt.plot(mmd, label="MMD")
            for seizure_start, seizure_stop in zip(start_points, stop_points):
                plt.axvspan(seizure_start, seizure_stop, color='r', alpha=0.5)
            plt.savefig("{}/{}_mmd.png".format(subdirname, node._v_name))
            plt.close()

            t_diff= np.abs(middle_points - arg_max_mmd)
            LOG.info("Time diff : {}".format(t_diff))
            middle_diff.append(t_diff)

            if start_points[0] < arg_max_mmd < stop_points[0]:
                delta = 0
            else:
                delta = np.min([np.abs(start_points[0] - arg_max_mmd),
                               np.abs(arg_max_mmd - stop_points[0])])
            T = stop_points[0]-start_points[0]
            LOG.info("Interval diff : {}, {}T".format(delta, delta/T))
            interval_diff.append(delta)
            interval_diff_normal.append(delta/T)

    LOG.info("Metrics:\nDelta to middle:{}\nDiff: {}\nNormalized Diff: {}".format(np.mean(middle_diff),
                                                                                  np.mean(interval_diff),
                                                                                  np.mean(interval_diff_normal)))
    plt.figure()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.subplot(311)
    plt.hist(middle_diff)
    plt.title('to middle')
    plt.subplot(312)
    plt.hist(interval_diff)
    plt.title('to start/stop point')
    plt.subplot(313)
    plt.hist(interval_diff_normal)
    plt.title('normalized to start/stop point')
    plt.savefig("{}/hist_diff.png".format(subdirname))
    plt.close()


if __name__=="__main__":
    # tf.config.experimental.set_visible_devices([], 'GPU')
    main()