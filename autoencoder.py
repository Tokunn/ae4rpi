#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import math

from PIL import Image

from datetime import datetime
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn import metrics

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/debug/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)



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

class TensorBoardImage(tf.keras.callbacks.Callback):
    def __init__(self, model, tag):
        super().__init__() 
        self._model = model
        self._tag = tag

    def calc_roc(self, diff_img, epoch):
        plt.figure()
        # flatten
        print(y_test.shape, diff_img.shape)
        y_test_reshape = np.reshape(y_test, (-1))
        diff_img_reshape = np.reshape(diff_img, (-1))
        # calc ROC,AUC
        fpr, tpr, threshoulds = metrics.roc_curve(y_test_reshape, diff_img_reshape)
        auc = metrics.auc(fpr, tpr)
        tf.summary.scalar('auc', data=auc, step=epoch)
        # Save ROC Image
        plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
        plt.legend()
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        plt.savefig(os.path.join(logdir, 'temp.png'))
        # Load Image
        fig = np.asarray(Image.open(os.path.join(logdir, 'temp.png')))
        fig = np.reshape(fig, (-1, fig.shape[0], fig.shape[1], fig.shape[2]))
        tf.summary.image('ROC', fig, step=epoch)

    def on_epoch_end(self, epoch, logs={}):
        global y_test
        # Predict
        predictions = self._model.predict(x_test)
        # Diff
        diff_img = x_test - predictions
        diff_list = np.array([np.sum(np.abs(x_test[i] - predictions[i])) for i in range(len(predictions))])
        # Concatenate
        y_test = np.reshape(y_test, (-1, y_test.shape[1], y_test.shape[2], 1))
        results1 = np.concatenate((x_test, predictions), axis=2)
        results2 = np.concatenate((diff_img, y_test), axis=2)
        results = np.concatenate((results1, results2), axis=1)
        # Reshape
        results = np.reshape(results, (-1, results.shape[1], results.shape[2], 1))
        # Write Image
        tf.summary.image(self._tag, results, max_outputs=30, step=epoch)
        # ROC
        self.calc_roc(diff_img, epoch)
        # Write loss
        for key in logs.keys():
            tf.summary.scalar(key, data=logs[key], step=epoch)

        return

model.compile(optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_squared_error'])

# Callbacks
tbi_callback = TensorBoardImage(model, 'Prediction')
#cbks = [tensorboard_callback, tbi_callback]
cbks = [tbi_callback]
# Training
model.fit(x_train,
          x_train,
          epochs=200,
          batch_size=256,
          callbacks=cbks)

model.summary()
## Evaluate
#model.evaluate(x_test, x_test)
