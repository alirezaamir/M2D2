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

    with tables.open_file("../input/eeg_data_temples2.h5") as h5_file:
        for node in h5_file.walk_nodes("/", "CArray"):
            LOG.info("Processing: {}".format(node._v_name))
            if len(node.attrs.seizures) != 1:
                continue

            data = node.read()
            X, y = data[:, :-1], data[:, -1]
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

            latent = intermediate_model.predict(Z)[2]
            sigma = intermediate_model.predict(Z)[1]
            sum_sigma = np.sum(sigma, axis=1)
            print('sigma: {}'.format(sum_sigma.shape))

            plt.figure()
            plt.plot(sum_sigma)
            plt.axvline(x=np.min(np.where(y > 0)[0]), linewidth=2, color="red")
            plt.axvline(x=np.max(np.where(y > 0)[0]), linewidth=2, color="red")
            plt.savefig("{}/{}_sigma.png".format(dirname, node._v_name))
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
            fig.savefig("{}/{}_tsne.png".format(dirname, node._v_name))
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
            for ix in range(1, mmd_corr.size):
                w = ((Z.shape[0] - 1) / float(ix * (N - ix)))
                ws.append(w)
                mmd_corr[ix] = mmd[ix] - w * mmd.max()

            plt.figure()
            plt.plot(mmd_corr, label="MMD")
            plt.axvline(x=np.min(np.where(y > 0)[0]), linewidth=2, color="red")
            plt.axvline(x=np.max(np.where(y > 0)[0]), linewidth=2, color="red")
            plt.savefig("{}/{}_mmd_corrected.png".format(dirname, node._v_name))
            plt.close()

            plt.figure()
            plt.plot(mmd, label="MMD")
            plt.axvline(x=np.min(np.where(y > 0)[0]), linewidth=2, color="red")
            plt.axvline(x=np.max(np.where(y > 0)[0]), linewidth=2, color="red")
            plt.savefig("{}/{}_mmd.png".format(dirname, node._v_name))
            plt.close()


if __name__=="__main__":
    # tf.config.experimental.set_visible_devices([], 'GPU')
    main()