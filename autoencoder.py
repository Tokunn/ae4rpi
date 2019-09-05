
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[y_train==1]
x_test = x_test[(y_test==1) | (y_test==9)]

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(392, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(392, activation='relu'),
    tf.keras.layers.Dense(784, activation='sigmoid'),
    tf.keras.layers.Reshape((28,28))
])

model.compile(optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_squared_error'])


model.fit(x_train,
          x_train,
          epochs=100,
          batch_size=256)

model.evaluate(x_test, x_test)

n_test = 40
predictions = model.predict(x_test[:n_test])

plt.figure()
col = 3
row = 10
for index in range(row):
    instance = x_test[index]
    decoded_img = predictions[index]
    diff_img = instance - decoded_img
    diff = round(np.sum(np.abs(diff_img)))

    subplot = plt.subplot(row, col, col*index+1)
    plt.imshow(instance)
    subplot = plt.subplot(row, col, col*index+2)
    plt.imshow(decoded_img)
    subplot = plt.subplot(row, col, col*index+3)
    plt.imshow(diff_img)
    subplot.set_ylabel(str(diff))

plt.figure()

diff_list = x_test[:n_test] - predictions
diff_list = [np.sum(np.abs(x_test[i] - predictions[i])) for i in range(len(predictions))]
plt.hist(diff_list)
plt.show()
