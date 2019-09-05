
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys, os

from datetime import datetime
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn import metrics

tb_cb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
cbks = [tb_cb]


MNIST = False
CHANNEL1 = True


if MNIST:
    mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[y_train==1]
    x_test = x_test[(y_test==1) | (y_test==9)]
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
else:
    import mvtechad
    (x_train, y_train), (x_test, y_test) = mvtechad.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    if CHANNEL1:
        x_train = x_train[:,:,:,0]
        x_test = x_test[:,:,:,0]
        x_train = np.reshape(x_train, (len(x_train), 128, 128, 1))
        x_test = np.reshape(x_test, (len(x_test), 128, 128, 1))
        y_test = y_test // 255.0


# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dense(392, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dense(392, activation='relu'),
#     tf.keras.layers.Dense(784, activation='sigmoid'),
#     tf.keras.layers.Reshape((28,28))
# ])

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (4,4), strides=(2,2), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(32, (4,4), strides=(2,2), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (4,4), strides=(2,2), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (4,4), strides=(2,2), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), activation='relu', padding='same'),
    
    tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (4,4), strides=(1,1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (4,4), strides=(1,1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (4,4), strides=(1,1), activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (4,4), strides=(1,1), activation='relu', padding='same'),

    tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same') # CHANNEL
])

#model = tf.keras.models.Sequential([
#    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
#    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
#    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
#    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
#    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
#    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
#    
#    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
#    tf.keras.layers.UpSampling2D((2, 2)),
#    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
#    tf.keras.layers.UpSampling2D((2, 2)),
#    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
#    tf.keras.layers.UpSampling2D((2, 2)),
#    tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same') # CHANNEL
#])

model.compile(optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_squared_error'])

model.fit(x_train,
          x_train,
          epochs=50,
          batch_size=256,
          callbacks=cbks)

model.summary()

model.evaluate(x_test, x_test)

predictions = model.predict(x_test)
if MNIST:
    predictions = np.reshape(predictions, (len(predictions), 28, 28))
    x_test = np.reshape(x_test, (len(x_test), 28, 28))
elif CHANNEL1:
    predictions = np.reshape(predictions, (len(predictions), 128, 128))
    x_test = np.reshape(x_test, (len(x_test), 128, 128))
    #y_test = np.reshape(y_test, (len(y_test), 128, 128))

# ROC
plt.figure()
y_test_reshape = np.reshape(y_test, (-1))
predictions_reshape = np.reshape(predictions, (-1))

print(y_test_reshape, predictions_reshape)
print(y_test_reshape.shape, predictions_reshape.shape)
print(y_test_reshape.dtype, predictions_reshape.dtype)
print(np.histogram(y_test_reshape))
print(np.histogram(predictions_reshape))
fpr, tpr, threshoulds = metrics.roc_curve(y_test_reshape, predictions_reshape)
auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROS curve (area = %.2f)'%auc)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.savefig('roc.png')

plt.figure()
#col = 3
#row = 10
#for index in range(row):
for index in range(len(predictions)):
    instance = x_test[index]
    decoded_img = predictions[index]
    diff_img = instance - decoded_img
    diff = round(np.sum(np.abs(diff_img)))

    #subplot = plt.subplot(row, col, col*index+1)
    #plt.imshow(instance)
    #subplot = plt.subplot(row, col, col*index+2)
    #plt.imshow(decoded_img)
    #subplot = plt.subplot(row, col, col*index+3)
    #plt.imshow(diff_img)
    #subplot.set_ylabel(str(diff))

    plt.figure()
    subplot = plt.subplot(1, 3, 1)
    plt.imshow(instance)
    subplot = plt.subplot(1, 3, 2)
    plt.imshow(decoded_img)
    subplot = plt.subplot(1, 3, 3)
    plt.imshow(diff_img)
    subplot.set_xlabel(str(diff))
    os.makedirs('output', exist_ok=True)
    plt.savefig(os.path.join('output', str(index)+'.png'))

#plt.savefig('result.png')


plt.figure()

diff_list = x_test - predictions
diff_list = [np.sum(np.abs(x_test[i] - predictions[i])) for i in range(len(predictions))]
plt.hist(diff_list)
plt.savefig('hist.png')
