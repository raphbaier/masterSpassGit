### Apply to new data ###

# Importing required libraries
from keras.models import Sequential, Model, model_from_json
import matplotlib.pyplot as plt
import keras
import pickle
import wave  # !pip install wave
import os
import pandas as pd
import numpy as np
import sys
import warnings
import librosa
import librosa.display
import IPython.display as ipd  # To play sound in the notebook
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils, to_categorical
from tqdm import tqdm, tqdm_pandas

import os


# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

file_path = 'input/cremad/AudioWAV/1001_DFA_ANG_XX.wav'


def speedNpitch(data):
    """
    Speed and Pitch Tuning.
    """
    # you can change low and high here
    length_change = np.random.uniform(low=0.8, high=1)
    speed_fac = 1.2 / length_change  # try changing 1.0 to 2.0 ... =D
    tmp = np.interp(np.arange(0, len(data), speed_fac), np.arange(0, len(data)), data)
    minlen = min(data.shape[0], tmp.shape[0])
    data *= 0
    data[0:minlen] = tmp[0:minlen]
    return data

def prepare_data(df, n, aug, mfcc):

    print("OKOKOKOKOKOK")

    print(df)
    print(df.shape[0])
    X = np.empty(shape=(df.shape[0], n, 216, 1))

    #meins
    #X = np.empty(shape=(df.shape[0], 30, 21, 1))


    input_length = sampling_rate * audio_duration

    cnt = 0
    for fname in tqdm(df.path):
        file_path = fname
        data, _ = librosa.load(file_path, sr=sampling_rate
                               , res_type="kaiser_fast"
                               , duration=2.5
                               , offset=0.5
                               )

        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length + offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")

        # Augmentation?
        if aug == 1:
            data = speedNpitch(data)

        # which feature?
        if mfcc == 1: ## 1
            # MFCC extraction
            MFCC = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc)
            MFCC = np.expand_dims(MFCC, axis=-1)
            X[cnt,] = MFCC
        else:
            # Log-melspectogram
            melspec = librosa.feature.melspectrogram(data, n_mels=n_melspec)
            logspec = librosa.amplitude_to_db(melspec)
            logspec = np.expand_dims(logspec, axis=-1)
            X[cnt,] = logspec
        cnt += 1

    print(X)
    return X


sampling_rate=44100
audio_duration=2.5

data, _ = librosa.load(file_path, sr=sampling_rate
                               , res_type="kaiser_fast"
                               , duration=2.5
                               , offset=0.5
                               )

# loading json and model architecture
json_file = open('model_json_conv.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("saved_models/Emotion_Model_Conv.h5")
print("Loaded model from disk")

n = 30
n_mfcc = 30

"""
counter = 1
for filename in os.listdir("input/cremad/AudioWAV"):
    counter += 1
X = np.empty(shape=(counter, n, 216, 1))

input_length = sampling_rate * audio_duration
counter = 0
for filename in os.listdir("input/cremad/AudioWAV"):

    data, _ = librosa.load("input/cremad/AudioWAV/" + filename, sr=sampling_rate
                           , res_type="kaiser_fast"
                           , duration=2.5
                           , offset=0.5
                           )

    # Random offset / Padding
    if len(data) > input_length:
        max_offset = len(data) - input_length
        offset = np.random.randint(max_offset)
        data = data[offset:(input_length + offset)]
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")

    # MFCC extraction
    MFCC = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc)
    MFCC = np.expand_dims(MFCC, axis=-1)
    X[counter,] = MFCC
    counter += 1
"""


"""
#Normalization
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_test = (X - mean)/std
"""

ref = pd.read_csv("input/Data_path.csv", nrows=12000)
ref.head()

mfcc = prepare_data(ref, n = n_mfcc, aug = 0, mfcc = 1)


# Split between train and test
#X_train, X_test, y_train, y_test = train_test_split(mfcc
#                                                    , ref.labels
#                                                    , test_size=0.25
#                                                    , shuffle=True
#                                                    , random_state=42
#                                                   )

# one hot encode the target
#lb = LabelEncoder()
#y_train = np_utils.to_categorical(lb.fit_transform(y_train))
#y_test = np_utils.to_categorical(lb.fit_transform(y_test))

# Normalization as per the standard NN process
#mean = np.mean(X_train, axis=0)
#std = np.std(X_train, axis=0)

#X_train = (X_train - mean)/std
#X_test = (X_test - mean)/std

opt = optimizers.Adam(0.001)
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

X_test = mfcc
X_train = mfcc




newpred = loaded_model.predict(X_train)
print(newpred)
filename = 'labels'
infile = open(filename,'rb')
lb = pickle.load(infile)
infile.close()

# Get the final predicted label
final = newpred.argmax(axis=1)
final = final.astype(int).flatten()
final = (lb.inverse_transform((final)))
print(final) #emo(final) #gender(final)
for fin in final:
    print(fin)


print("HAE")
counter = 0

true_counter = 0

for x in X_test:
    z = np.empty(shape=(1, n, 216, 1))
    z[0] = x
    pred = loaded_model.predict(z)
    #print(pred)
    #print(pred.argmax(axis=1))

    filename = 'labels'
    infile = open(filename, 'rb')
    lb = pickle.load(infile)
    infile.close()

    # Get the final predicted label
    final = pred.argmax(axis=1)
    final = final.astype(int).flatten()
    final = (lb.inverse_transform((final)))
    print(ref.labels[counter])
    print(final)  # emo(final) #gender(final)

    if ref.labels[counter] == final[0]:
        true_counter += 1

    counter += 1

print(true_counter)

"""
opt = optimizers.Adam(0.001)
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
"""