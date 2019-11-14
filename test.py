import tnml

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(13298)

# (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
# imgs = x_test.reshape(-1, 784)[:100,].tolist()
# lbls = (y_test == 1).astype("int")[:100].tolist()

x1 = np.random.rand(1000,1)
x2 = np.random.rand(1000,1)
x3 = np.random.rand(1000,1)
x4 = np.random.rand(1000,1)

A = np.concatenate((
    np.cos((np.pi/2.)*x1), np.sin((np.pi/2.)*x1),
    np.cos((np.pi/2.)*x2), np.sin((np.pi/2.)*x2),
    np.cos((np.pi/2.)*x3), np.sin((np.pi/2.)*x3),
    np.cos((np.pi/2.)*x4), np.sin((np.pi/2.)*x4)), axis=1)

print A[:3,:]

w = np.random.rand(8,1)
lbls = A.dot(w).ravel().tolist()

imgs = np.concatenate((x1, x2, x3, x4), axis=1).tolist()
pred = tnml.tnml(imgs, lbls)

plt.scatter(lbls, pred, color="darkgreen", alpha=0.5)
plt.show()