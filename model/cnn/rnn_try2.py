from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
import csv
import re

import tensorflow as tf
from keras.preprocessing import sequence
from tensorflow.keras import layers

batch_size = 100 #bei 2000 54%...
# Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
# Each input sequence will be of size (28, 28) (height is treated like time).
input_dim = 7

units = 64
output_size = 2  # labels are from 0 to 9


#TODO: Zahlen runden? gegen overfitting (aber kein overfitting festgestellt...)



def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# Build the RNN model
def build_model(allow_cudnn_kernel=True):
  # CuDNN is only available at the layer level, and not at the cell level.
  # This means `LSTM(units)` will use the CuDNN kernel,
  # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
  if allow_cudnn_kernel:
    # The LSTM layer with default options uses CuDNN.


    lstm_layer = tf.keras.layers.LSTM(units, input_shape=(None, input_dim))
    lstm_layer = tf.keras.layers.LSTM(units, input_shape=(None, input_dim))
  else:
    # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
    lstm_layer = tf.keras.layers.RNN(
        tf.keras.layers.LSTMCell(units),
        input_shape=(None, input_dim))


    ###fuer bidirectional ###
#    lstm_layer = tf.keras.layers.Bidirectional(
#        tf.keras.layers.LSTMCell(units),
#        input_shape=(None, input_dim))

  model = tf.keras.models.Sequential([
      lstm_layer,
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(output_size, activation='softmax')]
  )
  return model



mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
sample, sample_label = x_train[0], y_train[0]

x_train2 = x_train
y_train2 = y_train

print(np.array(x_train).shape)
print(np.array(y_train).shape)

#x_train = [[[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4]], [[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4]]]
#x_train = [x_train[0], x_train[1], x_train[0], x_train[1], x_train[0], x_train[1]]*10000
#y_train = [1, 2, 3, 4, 5, 6]*10000
model = build_model(allow_cudnn_kernel=True)
#x_test = x_train
#y_test = y_train


#reading data and pre-processing
reader = csv.reader(open("outputTEST.csv"), delimiter=",")
sorted_list = sorted(reader, key=lambda row: natural_keys(row[1]), reverse=False)
#for row in sorted_list:
#    print(row)

positive_X = []
positive_Y = []
negative_X = []
negative_Y = []
new_x = []

start_new_x = True
currently_positive = False
previous_sentence_number = -1
previous_speaker_number = -1

def add_emotions_to_x(row, x):
    x.append([float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8])])
    return x


for row in sorted_list[:-1]:
    #print(row)
    sentence_id = row[1].split("/")[8].split("_")
    sentence_number = int(sentence_id[0])
    speaker_number = int(sentence_id[2][:-5])
    #print(sentence_number + "  " + speaker_number)

    if start_new_x:
        #new_x.append([row[2], row[3], row[4], row[5], row[6], row[7], row[8]])
        new_x = add_emotions_to_x(row, new_x)
        start_new_x = False
        previous_sentence_number = sentence_number
        previous_speaker_number = speaker_number
        if row[0] == "POSITIVE":
            positive_Y.append(1)
            currently_positive = True
        if row[0] == "NEGATIVE":
            negative_Y.append(0)
            currently_positive = False
    else:
        #next sentence in speech of the same talker
        if previous_sentence_number + 1 == sentence_number and previous_speaker_number == speaker_number:
            #new_x.append([row[2], row[3], row[4], row[5], row[6], row[7], row[8]])
            new_x = add_emotions_to_x(row, new_x)
        else:
            if currently_positive:
                positive_X.append(new_x)
            else:
                negative_X.append(new_x)
            new_x = []
            #new_x.append([row[2], row[3], row[4], row[5], row[6], row[7], row[8]])
            new_x = add_emotions_to_x(row, new_x)

            if row[0] == "POSITIVE":
                positive_Y.append(1)
                currently_positive = True
            if row[0] == "NEGATIVE":
                negative_Y.append(0)
                currently_positive = False
    if row == sorted_list[-2]:
        if currently_positive:
            positive_X.append(new_x)
        else:
            negative_X.append(new_x)
    previous_sentence_number = sentence_number
    previous_speaker_number = speaker_number

print(len(positive_X))
print(len(positive_X))
print(len(positive_Y))
print(len(negative_X))
print(len(negative_Y))
#print(X)

positive_X_train = positive_X[:5000]
positive_Y_train = positive_Y[:5000]
negative_X_train = negative_X[:5000]
negative_Y_train = negative_Y[:5000]

positive_X_test = positive_X[5001:]
positive_Y_test = positive_Y[5001:]
negative_X_test = negative_X[5001:]
negative_Y_test = negative_Y[5001:]

X_train = positive_X_train + negative_X_train
Y_train = positive_Y_train + negative_Y_train
X_test = positive_X_test + negative_X_test
Y_test = positive_Y_test + negative_Y_test

#print(X_train[0])
#print(x_train[0])
maxlen = 300
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          validation_data=(X_test, Y_test),
          batch_size=batch_size,
              epochs=50)

###[] aussenrum um alles, wenns heisst expected 3d

#start_training()