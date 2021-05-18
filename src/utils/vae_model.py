import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from utils.mmd_layer import MMDLayer
from utils.losses import log_normal_pdf
from tensorflow.keras import models, layers, losses
from tensorflow.keras.utils import plot_model


def build_VAE_model(input_shape=None,
                    enc_dimension=None,
                    beta=None,
                    gamma=None,
                    optim=None,
                    FS=None):
    ii = layers.Input(shape=(input_shape))
    x = ii

    num_conv_layers = 3
    for _ in range(num_conv_layers):
        x = layers.Conv1D(8, 3, padding="same", activation="relu")(x)
        x = layers.Conv1D(8, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(2)(x)

    shape = K.int_shape(x)
    x = layers.Flatten()(x)
    mu = layers.Dense(enc_dimension, activation="linear", name="mu")(x)
    sigma = layers.Dense(enc_dimension, activation="linear", name="sigma")(x)
    z = layers.Lambda(
        sampling, output_shape=(enc_dimension,), name="latents")([mu, sigma])
    encoder = models.Model(inputs=ii, outputs=[mu, sigma, z, ii], name="encoder")

    #
    y_true = layers.Input(shape=(1), name="true_label")

    cl_dense1 = layers.Dense(enc_dimension, activation="relu", name="classifier_dense1")(z)
    cl_dense2 = layers.Dense(1, activation='sigmoid', name="classifier_dense2")(cl_dense1)

    latent_inputs = layers.Input(shape=(enc_dimension,), name='z_sampling')
    q = layers.Dense(shape[1] * shape[2], activation="relu")(latent_inputs)
    q = layers.Reshape((shape[1], shape[2]))(q)

    for _ in range(num_conv_layers):
        q = layers.Conv1D(8, 3, padding="same", activation="relu")(q)
        q = layers.UpSampling1D(size=2)(q)

    dec_output = layers.Conv1D(
        2, 3, padding="same", activation="linear", name="reconstruction")(q)
    decoder = models.Model(
        inputs=latent_inputs, outputs=dec_output, name="decoder")
    x_hat = decoder(encoder(ii)[2])

    recons_cost = K.mean(losses.mse(ii, x_hat))

    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mu, sigma)
    divergence = K.mean(-logpz + logqz_x)

    classification_cost = K.mean(losses.binary_crossentropy(y_true=y_true, y_pred=cl_dense2))

    cost = recons_cost + beta * divergence + gamma * classification_cost

    model = models.Model(inputs=[ii, y_true], outputs=x_hat)
    model.add_loss(cost)
    model.compile(optimizer=optim)

    model.metrics.append(divergence)
    model.metrics_names.append("mmd_elbo")

    return model, encoder


def get_mmd_model(state_len=None,
                  latent_dim=None,
                  signal_len=None,
                  seq_len = None,
                  trainable_vae=True):
    input_signal = tf.keras.layers.Input(shape=(seq_len, signal_len, 2))
    x = input_signal
    num_conv_layers = 3
    for i in range(num_conv_layers):
        x = layers.TimeDistributed(layers.Conv1D(8, 3, padding="same", activation="relu", trainable=trainable_vae),
                                   name="conv1d_{}_1".format(i+1))(x)
        x = layers.TimeDistributed(layers.Conv1D(8, 3, padding="same", activation="relu", trainable=trainable_vae),
                                   name="conv1d_{}_2".format(i+1))(x)
        x = layers.TimeDistributed(layers.MaxPooling1D(2), name="pool_{}".format(i+1))(x)

    x = layers.TimeDistributed(layers.Flatten(), name='flatten')(x)
    mu = layers.TimeDistributed(layers.Dense(latent_dim, activation="linear", name="mu", trainable=trainable_vae),
                                name='mu')(x)
    sigma = layers.TimeDistributed(layers.Dense(latent_dim, activation="linear", name="sigma", trainable=trainable_vae),
                                   name='sigma')(x)
    z = layers.Lambda(
        new_sampling, output_shape=(latent_dim,), name="latents")([mu, sigma, latent_dim])
    mmd = tf.keras.layers.Bidirectional(MMDLayer(state_len, latent_dim, mask_th=16, go_backwards=False), name='MMD')(z)
    interval = tf.keras.layers.Conv1D(filters=9, kernel_size=17, padding='same', use_bias=False, name='conv_interval',
                                      trainable=False)(mmd)
    gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=25, return_sequences=True), name='GRU')(interval)
    dense1 = tf.keras.layers.Dense(100, activation='relu', name='dense1')(gru)
    final_dense = tf.keras.layers.Dense(1, activation='sigmoid', name='final_dense')(dense1)

    model = tf.keras.models.Model(inputs=input_signal, outputs=final_dense)
    model.add_loss(tf.abs(tf.reduce_mean(tf.abs(z)-1)) * 1e-2)
    return model


def get_conventional_model(state_len=None,
                  latent_dim=None,
                  signal_len=None,
                  seq_len = None,
                  trainable_vae=True):
    input_signal = tf.keras.layers.Input(shape=(seq_len, signal_len, 2))
    x = input_signal
    num_conv_layers = 3
    for i in range(num_conv_layers):
        x = layers.TimeDistributed(layers.Conv1D(8, 3, padding="same", activation="relu", trainable=trainable_vae),
                                   name="conv1d_{}_1".format(i + 1))(x)
        x = layers.TimeDistributed(layers.Conv1D(8, 3, padding="same", activation="relu", trainable=trainable_vae),
                                   name="conv1d_{}_2".format(i + 1))(x)
        x = layers.TimeDistributed(layers.MaxPooling1D(2), name="pool_{}".format(i + 1))(x)

    x = layers.TimeDistributed(layers.Flatten(), name='flatten')(x)
    mu = layers.TimeDistributed(layers.Dense(latent_dim, activation="linear", name="mu", trainable=trainable_vae,
                                             kernel_initializer='zeros'),
                                name='mu')(x)
    sigma = layers.TimeDistributed(layers.Dense(latent_dim, activation="linear", name="sigma", trainable=trainable_vae,
                                                kernel_initializer='zeros'),
                                    name='sigma')(x)
    z = layers.Lambda(
        MMD_free_sampling, output_shape=(latent_dim,), name="latents")([mu, sigma, latent_dim])

    dense1 = tf.keras.layers.Dense(4*latent_dim, activation='relu', name='dense1')(z)
    dense2 = tf.keras.layers.Dense(2*latent_dim, activation='relu', name='dense2')(dense1)
    final_dense = tf.keras.layers.Dense(1, activation='sigmoid', name='final_dense')(dense2)

    model = tf.keras.models.Model(inputs=input_signal, outputs=final_dense)
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mu, sigma)
    divergence = K.mean(-logpz + logqz_x)
    model.add_loss(divergence * 1e-4)
    return model


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def new_sampling(args):
    z_mean, z_log_var, dim = args
    batch = K.shape(z_mean)[0]
    epsilon = K.random_normal(shape=(batch, dim))
    z = z_mean + K.exp(0.5 * z_log_var) * epsilon
    z_expanded = K.expand_dims(z, axis=2)
    return z_expanded


def MMD_free_sampling(args):
    z_mean, z_log_var, dim = args
    batch = K.shape(z_mean)[0]
    seq = K.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch,seq, dim))
    z = z_mean + K.exp(0.5 * z_log_var) * epsilon
    return z

