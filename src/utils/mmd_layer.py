import pprint
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class MMDLayer(tf.keras.layers.Layer):
    def __init__(self, state_len, latent_dim, mask_th, go_backwards, return_sequences=True, return_state=False, **kwargs):
        self.state_len = state_len
        self.latent_dim = latent_dim
        self.go_backwards = go_backwards
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.mask_th = mask_th
        # self.init_state = tf.constant(np.zeros(shape=(state_len, latent_dim)), dtype=tf.float32)
        # self.go_backward = backward
        # w_init = tf.random_normal_initializer()
        # self.w = tf.Variable(
        #     initial_value=w_init(shape=(state_len, latent_dim), dtype="float32"),
        #     trainable=False,
        # )

        super(MMDLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MMDLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.state_len

    def call(self, x, mask=None):
        d_t1 = tf.random.normal(shape=(self.state_len, self.latent_dim), dtype=tf.float32)
        # d_t1 = self.w
        z_t = x
        # d_t1 = x[1]
        _, kernel, d_t = tf.keras.backend.rnn(self.step, z_t, [d_t1], go_backwards=self.go_backwards)

        return kernel

    def get_masked_sum(self, input_array, th):
        time_steps = tf.range(start=0, limit=self.state_len, dtype=tf.float32)
        sigmoid = tf.sigmoid(time_steps - th - 0.5)
        masked_inside = tf.multiply(input_array, 1 - sigmoid)
        inside_sum = tf.reduce_sum(masked_inside, axis=1, keepdims=True)
        # average = tf.divide(inside_sum, th)
        return inside_sum

    def step(self, input, states):
        d_t1 = states
        z_t = input

        # Output
        tile_dim = tf.constant([1, self.state_len, 1], dtype=tf.int32)
        z_tile = tf.tile(z_t, tile_dim)
        dot = tf.multiply(z_tile, d_t1)
        # diff2 = tf.pow(diff, 2)
        sum = tf.reduce_sum(dot, axis=2)
        gamma = tf.multiply(1/self.latent_dim, sum)
        coeff = tf.add(1.0, gamma)
        poly = tf.pow(coeff, 3)

        inside0 = self.get_masked_sum(poly, 0)
        inside1 = self.get_masked_sum(poly, 1)
        inside2 = self.get_masked_sum(poly, 2)
        inside3 = self.get_masked_sum(poly, 3)
        inside4 = self.get_masked_sum(poly, 4)
        inside5 = self.get_masked_sum(poly, 5)
        inside6 = self.get_masked_sum(poly, 6)
        inside7 = self.get_masked_sum(poly, 7)
        inside8 = self.get_masked_sum(poly, 8)
        inside9 = self.get_masked_sum(poly, 9)
        inside10 = self.get_masked_sum(poly, 10)
        inside11 = self.get_masked_sum(poly, 11)
        inside12 = self.get_masked_sum(poly, 12)
        inside1000 = self.get_masked_sum(poly, self.state_len)

        inout = tf.concat([inside0, inside1, inside2, inside3, inside4, inside5, inside6, inside7, inside8,
                           inside9, inside10, inside11, inside12, inside1000], axis=1)
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
                state_len = self.state_len,
                latent_dim = self.latent_dim,
                go_backwards= self.go_backwards,
                return_sequences = self.return_sequences,
                return_state = self.return_state,
                mask_th = self.mask_th
            )
        )
        return cfg
