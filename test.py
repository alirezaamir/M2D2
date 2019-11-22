import tcgrn

import numpy as np
import matplotlib.pyplot as plt

from keras import datasets

np.random.seed(13298)

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
imgs = (x_test.reshape(-1, 784).astype(np.float64) / 255.)[:10000,].tolist()
lbls = (y_test == 1).astype("int")[:10000].tolist()

out = tcgrn.tcgrn(imgs)