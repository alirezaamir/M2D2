import utils
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K

from losses import mmd_loss
from tensorflow import keras
from tensorflow.keras import models, layers, losses


def build_model(input_shape=None, 
                enc_dimension=None, 
                beta=None,
                gamma=None,
                optim=None):

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
    encoder = models.Model(inputs=ii, outputs=[mu, sigma, z], name="encoder")

    latent_inputs = layers.Input(shape=(enc_dimension,), name='z_sampling')
    q = layers.Dense(shape[1]*shape[2], activation="relu")(latent_inputs)
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

    z_actual = tf.random.normal(tf.stack([200, enc_dimension]))
    divergence = mmd_loss(z, z_actual)

    cost = recons_cost + beta*divergence

    model = models.Model(inputs=ii, outputs=x_hat)
    model.add_loss(cost)
    model.compile(optim)
    
    model.metrics.append(recons_cost)
    model.metrics_names.append("recons_cost")
    
    model.metrics.append(divergence)
    model.metrics_names.append("mmd_elbo")

    return model, decoder


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon