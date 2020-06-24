import tensorflow as tf
import tensorflow.keras.backend as K

from losses import mmd_loss
from tensorflow.keras import models, layers, losses
from tensorflow.keras.utils import plot_model


def build_model(input_shape=None,
                label_shape = None,
                enc_dimension=None,
                optim=None):
    z1_input = layers.Input(shape=input_shape)
    label_input = layers.Input(shape=label_shape)

    concat_input = layers.Concatenate()([z1_input, label_input])
    # dense1 = layers.Dense(32, activation='relu')(concat_input)
    dense1 = layers.Dense(32, activation='relu')(z1_input)
    latent = layers.Dense(enc_dimension, activation='relu')(dense1)
    dense3 = layers.Dense(32, activation='relu')(latent)
    recons = layers.Dense(input_shape[0], activation=None)(dense3)

    # recons_cost = K.mean(losses.mse(concat_input, recons))

    model = models.Model(inputs=z1_input, outputs=recons, name="encoder")
    model.compile(optimizer=optim, loss = "mse")
    return model


if __name__ == '__main__':
    model = build_model(input_shape=(16,), enc_dimension=8, optim='adam', label_shape=(1,))
    plot_model(model, '../output/AE_model.png', show_shapes=True, show_layer_names=False)