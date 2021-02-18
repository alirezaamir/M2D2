import pprint
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class MMDLayer(tf.keras.layers.Layer):
    def __init__(self, initial_state, mask_th, go_backwards, return_sequences=True, return_state=False, **kwargs):
        self.initial_state = tf.constant(initial_state, dtype=tf.float32)
        self.state_len = initial_state.shape[0]
        self.latent_dim = initial_state.shape[1]
        self.go_backwards = go_backwards
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.mask_th = mask_th
        # self.go_backward = backward
        super(MMDLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MMDLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.state_len

    def call(self, x, mask=None):
        d_t1 = self.initial_state
        z_t = x
        _, kernel, d_t = tf.keras.backend.rnn(self.step, z_t, [d_t1], go_backwards=self.go_backwards)

        return kernel

    def get_masked_sum(self, input_array, th):
        time_steps = tf.range(start=0, limit=self.state_len, dtype=tf.float32)
        sigmoid = tf.sigmoid(time_steps - th)
        masked_inside = tf.multiply(input_array, 1 - sigmoid)
        inside_sum = tf.reduce_sum(masked_inside, axis=1, keepdims=True)
        return inside_sum

    def step(self, input, states):
        d_t1 = states
        z_t = input

        # Output
        tile_dim = tf.constant([1, self.state_len, 1], dtype=tf.int32)
        z_tile = tf.tile(z_t, tile_dim)
        diff = tf.subtract(z_tile, d_t1)
        diff2 = tf.pow(diff, 2)
        sum = tf.reduce_sum(diff2, axis=2)
        gamma = tf.multiply(-1/self.latent_dim, sum)
        coeff = tf.add(1.0, gamma)
        poly = tf.pow(coeff, 3)

        inside1 = self.get_masked_sum(poly, 1)
        inside2 = self.get_masked_sum(poly, 2)
        inside4 = self.get_masked_sum(poly, 4)
        inside8 = self.get_masked_sum(poly, 8)
        inside16 = self.get_masked_sum(poly, 16)
        inside1000 = self.get_masked_sum(poly, 1024)

        inout = tf.concat([inside1, inside2, inside4, inside8, inside16, inside1000], axis=1)
        # rbf = tf.exp(gamma)

        # Updating the state
        concatenated = tf.concat([z_t, d_t1], axis=1)
        sliced = tf.slice(concatenated, [0, 0, 0], [1, self.state_len, self.latent_dim])
        d_t = tf.squeeze(sliced, axis=0)

        return inout, d_t

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            dict(
                initial_state=self.initial_state,
                go_backwards= self.go_backwards,
                return_sequences = self.return_sequences,
                return_state = self.return_state,
                mask_th = self.mask_th
            )
        )
        return cfg
