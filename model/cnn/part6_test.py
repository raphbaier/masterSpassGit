# Keras
import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.utils import np_utils, to_categorical
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation, Input, Dense)

# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Other
from tqdm import tqdm, tqdm_pandas
import scipy
from scipy.stats import skew
import librosa
import librosa.display
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import seaborn as sns
import glob
import os
import sys
import IPython.display as ipd  # To play sound in the notebook
import warnings

import arff

# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")



df = pd.read_csv("input/Data_path.csv")
df.head()
n = 30
X = np.empty(shape=(df.shape[0], n, 216, 1))

print(df.shape)
print(df.shape[0])

file_path = df.path[0]
print(file_path)

sampling_rate=44100
audio_duration=2.5


data, _ = librosa.load(file_path, sr=sampling_rate
                               , res_type="kaiser_fast"
                               , duration=2.5
                               , offset=0.5
                               )

offset = 0
input_length = len(data)


print(data)
data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")
print(data)

MFCC = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n)
print(MFCC)
print(len(MFCC))
print(len(MFCC[0]))


#get mfcc features from arff
arff_path = "/home/raphael/masterSpassGit/model/cnn/input_arff/male/sad/_KL_sa10.arff"
with open(arff_path) as arff_opened:
    arff_file = arff.load(arff_opened)
arff_opened.close()

print(arff_file['attributes'])
print(arff_file['data'])


counter = 0
derived = True
mfcc_list = []

last_counter = "0"
new_mfcc = []

print(len(arff_file['attributes']))
print(len(arff_file['data'][0]))
print(arff_file['data'][0])

for attribute in arff_file['attributes']:
    attribute_found = False
    if 'mfcc_sma' in attribute[0]:
        if not derived:
            if not "sma_de" in attribute[0]:
                print(attribute[0])
                attribute_found = True
        else:
            attribute_found = True

    if attribute_found:
        attributes = attribute[0].replace("]", "[").split("[")
        print(attributes)
        if attributes[1] != last_counter:

            mfcc_list.append(new_mfcc)
            last_counter = attributes[1]
            new_mfcc = []
            new_mfcc.append(arff_file['data'][0][counter])

        else:
            new_mfcc.append(arff_file['data'][0][counter])


    counter += 1

mfcc_list.append(new_mfcc)

#test it
print(len(mfcc_list))
for ele in mfcc_list:
    print(len(ele))