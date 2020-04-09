import os
import sys
import json
import time
import pywt
import tables
import mmd_vae
import logging
import datetime
import matplotlib
matplotlib.use("Agg")

sys.path.append("../")

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

from scipy import signal
from params import SEG_N
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback, EarlyStopping, CSVLogger, LearningRateScheduler

np.random.seed(13298)

LOG = logging.getLogger(os.path.basename(__file__))
ch = logging.StreamHandler()
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch.setFormatter(logging.Formatter(log_fmt))
ch.setLevel(logging.INFO)
LOG.addHandler(ch)
LOG.setLevel(logging.INFO)

FS = 256.0

def main():
    arch = sys.argv[1]
    beta = float(sys.argv[2])
    latent_dim = int(sys.argv[3])
    lr = float(sys.argv[4])
    decay = float(sys.argv[5])

    param_str = """
    ==========================
        Arch:           {}
        Beta:           {}
        Decay:          {}
        Encoder Dim:    {}
        Learning Rate:  {}
    ==========================""".format(arch, beta, decay, latent_dim, lr)
    LOG.info("Training Model with parameters:{}".format(param_str))

    build_model = mmd_vae.build_model
    root = "../../output/vae/{}/".format(arch)
    stub = "/seg_n_{}/beta_{}/latent_dim_{}/lr_{}/decay_{}"
    dirname = root + stub.format(SEG_N, beta, latent_dim, lr, decay)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    # else:
    #     LOG.warning("Model already trained...")
    #     return

    beta = K.variable(0.)
    build_model_args = {
        "input_shape": (SEG_N, 2,),
        "enc_dimension": latent_dim,
        "beta": beta,
        "gamma": 1e-3,
        "optim": Adam(lr)
    }

    model, _ = build_model(**build_model_args)
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    LOG.info("Model Summary:")
    LOG.info("\n".join(model_summary))

    train_model(model, dirname, lr, decay, beta)


def train_model(model, dirname, lr_init, decay, beta):
    max_epochs = 200
    patience   = 30
    batch_size = 32
    beta_start_epoch = 10

    history        = CSVLogger(dirname + "/training.log")
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True)
    scheduler      = LearningRateScheduler(lambda x,y: lr_init*np.exp(-decay*x))
    beta_annealing = AnnealingCallback(beta, beta_start_epoch, max_epochs)

    train_data = build_dataset("train", batch_size)
    valid_data = build_dataset("valid", batch_size)

    model.fit(train_data, validation_data=valid_data, epochs=max_epochs,
              callbacks=[early_stopping, history, scheduler, beta_annealing])

    model.save(dirname + "/saved_model.h5")


def build_dataset(mode, batch_size):
    dirname = "../../temp/vae/{}".format(mode)
    filenames = ["{}/{}".format(dirname, x) for x in os.listdir(dirname)]
    dataset = tf.data.TFRecordDataset(
            filenames
        ).map(
            _parse_fn
        ).shuffle(4096).batch(batch_size)
    return dataset


def _parse_fn(proto):
    parse_dict = {"channels": tf.io.FixedLenFeature([], tf.string)}
    example = tf.io.parse_single_example(proto, parse_dict)
    X = tf.io.parse_tensor(example["channels"], out_type=tf.float32)
    X = tf.reshape(X, [SEG_N,2])  # Annoying hack needed for Keras
    return X


class AnnealingCallback(Callback):
    
    def __init__(self, beta, beta_start_epoch=3, max_epochs=200):
        self.beta = beta
        self.beta_start_epoch = beta_start_epoch
        self.max_epochs = max_epochs

    def on_epoch_end(self, epoch, logs=None):
         if epoch > self.beta_start_epoch:
            beta_new = min(
                K.get_value(self.beta) + (1 / float(self.max_epochs)), 1.0)
            K.set_value(self.beta, beta_new)


if __name__=="__main__":
    main()
