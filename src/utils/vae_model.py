import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from utils.mmd_layer import MMDLayer
from utils.losses import log_normal_pdf
from tensorflow.keras import models, layers, losses
from tensorflow.keras.utils import plot_model


def build_model(input_shape=None,
                enc_dimension=None,
                beta=None,
                gamma=None,
                optim=None,
                FS=None):
    ii = layers.Input(shape=(input_shape))
    x = ii
    ii_permutation = K.permute_dimensions(ii, (0, 2, 1))
    input_fft = tf.signal.rfft(ii_permutation)
    fft_permutation = K.permute_dimensions(input_fft, (0, 2, 1))
    input_psd = 1 / (FS * input_shape[0]) * tf.math.square(fft_permutation)

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
    cl_input = layers.Input(shape=(enc_dimension,), name='z_classifier_in')
    cl_dense1 = layers.Dense(enc_dimension, activation="relu", name="classifier_dense1")(cl_input)
    cl_dense2 = layers.Dense(1, activation=None, name="classifier_dense2")(cl_dense1)
    classifier = models.Model(
        inputs=cl_input, outputs=cl_dense2, name="classifier")
    y_class = classifier(encoder(ii)[2])

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
    x_hat_permutation = K.permute_dimensions(x_hat, (0, 2, 1))
    x_hat_fft = tf.signal.rfft(x_hat_permutation)
    fft_permutation = K.permute_dimensions(x_hat_fft, (0, 2, 1))
    x_hat_psd = 1 / (FS * input_shape[0]) * tf.math.square(fft_permutation)

    recons_cost = K.mean(losses.mse(ii, x_hat))
    freq_cost = K.mean(losses.mse(input_psd[:, 1:, :], x_hat_psd[:, 1:, :]))
    freq_cost = K.abs(freq_cost)

    # z_actual = tf.random.normal(tf.stack([200, enc_dimension]))
    # divergence = mmd_loss(z, z_actual)

    # First method for KL divergence
    # z_actual = tf.random.normal([1, enc_dimension])
    # kl = tf.keras.losses.KLDivergence()
    # divergence = kl(z_actual, z)

    # Second method to calculate the KL divergence
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mu, sigma)
    divergence = K.mean(-logpz + logqz_x)

    # classification_cost = K.mean(losses.binary_crossentropy(y_true=y_true, y_pred=y_class))

    cost = recons_cost + beta * divergence  # + gamma*freq_cost + 0.1 * classification_cost

    model = models.Model(inputs=[ii, y_true], outputs=x_hat)
    model.add_loss(cost)
    model.compile(optimizer=optim)

    # model.metrics.append(recons_cost)
    # model.metrics_names.append("recons_cost")

    # model.metrics.append(freq_cost)
    # model.metrics_names.append("freq_cost")

    model.metrics.append(divergence)
    model.metrics_names.append("mmd_elbo")

    return model, encoder


def build_ae_model(input_shape=None,
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
    z = layers.Dense(enc_dimension, activation="linear", name="z")(x)

    #
    y_true = layers.Input(shape=(1), name="true_label")
    # cl_dense1 = layers.Dense(enc_dimension, activation="relu", name="classifier_dense1")(z)
    # cl_dense2 = layers.Dense(1, activation=None, name="classifier_dense2")(cl_dense1)

    q = layers.Dense(shape[1] * shape[2], activation="relu")(z)
    q = layers.Reshape((shape[1], shape[2]))(q)

    for _ in range(num_conv_layers):
        q = layers.Conv1D(8, 3, padding="same", activation="relu")(q)
        q = layers.UpSampling1D(size=2)(q)

    dec_output = layers.Conv1D(
        2, 3, padding="same", activation="linear", name="reconstruction")(q)
    x_hat = dec_output

    recons_cost = K.mean(losses.mse(ii, x_hat))

    # classification_cost = K.mean(losses.binary_crossentropy(y_true=y_true, y_pred=cl_dense2))

    cost = recons_cost  # + 0.1 * classification_cost  # + beta*divergence + gamma*freq_cost

    model = models.Model(inputs=[ii, y_true], outputs=x_hat)
    model.add_loss(cost)
    model.compile(optimizer=optim)

    encoder = models.Model(inputs=model.input, outputs=model.get_layer('z').output)

    return model, encoder


def get_mmd_model(state_len=None,
                  latent_dim=None,
                  signal_len=None,
                  seq_len = None,
                  trainable_vae=True):
    initial_state = np.zeros((state_len, latent_dim))
    input_signal = tf.keras.layers.Input(shape=(seq_len, signal_len, 2))
    input_random = tf.keras.layers.Input(shape=(seq_len, latent_dim))
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
        simple_sampling, output_shape=(latent_dim,), name="latents")([mu, sigma, input_random])
    bidirectional = tf.keras.layers.Bidirectional(MMDLayer(initial_state, mask_th=6, go_backwards=False))(z)
    gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=100, return_sequences=True))(bidirectional)
    dense1 = tf.keras.layers.Dense(latent_dim, activation='relu', name='dense1')(gru)
    final_dense = tf.keras.layers.Dense(1, activation=None, name='final_dense')(dense1)
    flatten = tf.keras.layers.Flatten()(final_dense)
    softmax = tf.keras.layers.Softmax()(flatten)

    model = tf.keras.models.Model(inputs=[input_signal, input_random], outputs=final_dense)
    return model


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def simple_sampling(args):
    z_mean, z_log_var, epsilon = args
    z =  epsilon * K.exp(0.5 * z_log_var) + z_mean
    z_expanded = K.expand_dims(z, axis=2)
    return z_expanded


if __name__ == '__main__':
    tf.config.experimental.set_visible_devices([], 'GPU')
    model = build_ae_model(input_shape=(1024, 2,), enc_dimension=16, beta=0, optim='adam', gamma=100, FS=256)
    plot_model(model[0], '../../output/AE_model.png', show_shapes=True, show_layer_names=False)
