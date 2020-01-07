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

'''
1. Data Augmentation method
'''


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


'''
2. Extracting the MFCC feature as an image (Matrix format).
'''

def get_mfcc_from_arff(file):
    # get mfcc features from arff
    try:
        with open(file) as arff_opened:
            arff_file = arff.load(arff_opened)
        arff_opened.close()
    except:
        with open(
                "/home/raphael/masterSpassGit/model/cnn/input_arff/female/disgust/_03-01-07-01-01-01-02.arff") as file2:
            arff_file = arff.load(file2)
        file2.close()

    counter = 0
    derived = True
    mfcc_list = []

    last_counter = "0"
    new_mfcc = []
    for attribute in arff_file['attributes']:
        attribute_found = False
        if 'mfcc_sma' in attribute[0]:
            if not derived:
                if not "sma_de" in attribute[0]:
                    attribute_found = True
            else:
                attribute_found = True

        if attribute_found:
            if len(arff_file['data']) > 0:
            #if counter < len(arff_file['data'][0]):
                new_mfcc.append(arff_file['data'][0][counter])
            else:
                return None
            """
            attributes = attribute[0].replace("]", "[").split("[")
            if attributes[1] != last_counter:

                mfcc_list.append(new_mfcc)
                last_counter = attributes[1]
                new_mfcc = []
                new_mfcc.append(arff_file['data'][0][counter])

            else:
                new_mfcc.append(arff_file['data'][0][counter])"""
        counter += 1
    new_mfcc = np.array(new_mfcc)
    new_mfcc = new_mfcc.reshape((30, 21))
    mfcc_list.append(new_mfcc)

    return mfcc_list



def prepare_data(df, n, aug, mfcc):

    print("OKOKOKOKOKOK")

    print(df)
    print(df.shape[0])
    X = np.empty(shape=(df.shape[0], n, 21, 1))

    #meins
    #X = np.empty(shape=(df.shape[0], 30, 21, 1))


    input_length = sampling_rate * audio_duration

    cnt = 0
    for fname in tqdm(df.path):
        file_path = fname
        # Augmentation?
        if aug == 1:
            data = speedNpitch(data)

        elif mfcc == 1:
            mfccc = get_mfcc_from_arff(fname)
            mfccc = np.expand_dims(mfccc, axis=-1)
            X[cnt,] = mfccc

        else:
            # Log-melspectogram
            melspec = librosa.feature.melspectrogram(data, n_mels=n_melspec)
            logspec = librosa.amplitude_to_db(melspec)
            logspec = np.expand_dims(logspec, axis=-1)
            X[cnt,] = logspec

        cnt += 1

    return X


'''
3. Confusion matrix plot
'''


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    '''Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    '''
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


'''
# 4. Create the 2D CNN model
'''


def get_2d_conv_model(n):
    ''' Create a standard deep 2D convolutional neural network'''
    #nclass = 14
    nclass = 2
    inp = Input(shape=(n, 21, 1))


    x = Convolution2D(32, (4, 10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(rate=0.2)(x)

    out = Dense(nclass, activation=softmax)(x)
    model = models.Model(inputs=inp, outputs=out)

    opt = optimizers.Adam(0.001) #0.001
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model


'''
# 5. Other functions
'''


class get_results:
    '''
    We're going to create a class (blueprint template) for generating the results based on the various model approaches.
    So instead of repeating the functions each time, we assign the results into on object with its associated variables
    depending on each combination:
        1) MFCC with no augmentation
        2) MFCC with augmentation
        3) Logmelspec with no augmentation
        4) Logmelspec with augmentation
    '''

    def __init__(self, model_history, model, X_test, y_test, labels):
        self.model_history = model_history
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.labels = labels

    def create_plot(self, model_history):
        '''Check the logloss of both train and validation, make sure they are close and have plateau'''
        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def create_results(self, model):
        '''predict on test set and get accuracy results'''
        opt = optimizers.Adam(0.001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        score = model.evaluate(X_test, y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    """
    def confusion_results(self, X_test, y_test, labels, model):
        '''plot confusion matrix results'''
        preds = model.predict(X_test,
                              batch_size=16,
                              verbose=2)
        preds = preds.argmax(axis=1)
        preds = preds.astype(int).flatten()
        preds = (lb.inverse_transform((preds)))

        actual = y_test.argmax(axis=1)
        actual = actual.astype(int).flatten()
        actual = (lb.inverse_transform((actual)))

        classes = labels
        classes.sort()

        c = confusion_matrix(actual, preds)
        print("DODA")
        print(actual)
        print(preds)

        fig = plt.figure()
        plt.matshow(c)
        #plt.title('positive courses vs negative courses')
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicated Label')
        plt.savefig('confusion_matrix' + '.pdf')


        print_confusion_matrix(c, class_names=classes)"""


    def confusion_results(self, X_test, y_test, labels, model):
        '''plot confusion matrix results'''
        preds = model.predict(X_test,
                              batch_size=16,
                              verbose=2)
        preds = preds.argmax(axis=1)
        preds = preds.astype(int).flatten()
        preds = (lb.inverse_transform((preds)))

        actual = y_test.argmax(axis=1)
        actual = actual.astype(int).flatten()
        actual = (lb.inverse_transform((actual)))

        classes = labels
        classes.sort()

        c = confusion_matrix(actual, preds)
        print_confusion_matrix(c, class_names=classes)

        fig = plt.figure()
        plt.matshow(c)
        # plt.title('positive courses vs negative courses')
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicated Label')
        plt.savefig('confusion_matrix' + '.pdf')



    def accuracy_results_gender(self, X_test, y_test, labels, model):
        '''Print out the accuracy score and confusion matrix heat map of the Gender classification results'''

        preds = model.predict(X_test,
                              batch_size=16,
                              verbose=2)
        preds = preds.argmax(axis=1)
        preds = preds.astype(int).flatten()
        preds = (lb.inverse_transform((preds)))

        actual = y_test.argmax(axis=1)
        actual = actual.astype(int).flatten()
        actual = (lb.inverse_transform((actual)))

        # print(accuracy_score(actual, preds))

        actual = pd.DataFrame(actual).replace({'female_angry': 'female'
                                                  , 'female_disgust': 'female'
                                                  , 'female_fear': 'female'
                                                  , 'female_happy': 'female'
                                                  , 'female_sad': 'female'
                                                  , 'female_surprise': 'female'
                                                  , 'female_neutral': 'female'
                                                  , 'male_angry': 'male'
                                                  , 'male_fear': 'male'
                                                  , 'male_happy': 'male'
                                                  , 'male_sad': 'male'
                                                  , 'male_surprise': 'male'
                                                  , 'male_neutral': 'male'
                                                  , 'male_disgust': 'male'
                                               })
        preds = pd.DataFrame(preds).replace({'female_angry': 'female'
                                                , 'female_disgust': 'female'
                                                , 'female_fear': 'female'
                                                , 'female_happy': 'female'
                                                , 'female_sad': 'female'
                                                , 'female_surprise': 'female'
                                                , 'female_neutral': 'female'
                                                , 'male_angry': 'male'
                                                , 'male_fear': 'male'
                                                , 'male_happy': 'male'
                                                , 'male_sad': 'male'
                                                , 'male_surprise': 'male'
                                                , 'male_neutral': 'male'
                                                , 'male_disgust': 'male'
                                             })

        classes = actual.loc[:, 0].unique()
        classes.sort()

        c = confusion_matrix(actual, preds)

        print(accuracy_score(actual, preds))
        print_confusion_matrix(c, class_names=classes)

#mit nrows schneller
ref = pd.read_csv("input/arff_data_path_large.csv", nrows=200000) #354895

ref.head()

print("OKOK")
for row in ref:
    print("WOS")
    print(row)

print(ref)


print("OKOK")
sampling_rate=44100
audio_duration=2.5
n_mfcc = 30


print("PREPARING...")
mfcc = prepare_data(ref, n = n_mfcc, aug = 0, mfcc = 1)
print("OKOKOKOKOKOKOKOK PREPARED")
print(mfcc)

# Split between train and test
#X_train, X_test, y_train, y_test = train_test_split(mfcc
#                                                    , ref.labels
#                                                    , test_size=0.25
#                                                    , shuffle=True
#                                                    , random_state=42
#                                                   )

X_train, X_test, y_train, y_test = train_test_split(mfcc
                                                    , ref.labels
                                                    , test_size=0.15
                                                    , shuffle=False
                                                   )


# one hot encode the target
lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

# Normalization as per the standard NN process

###have to normalize it later, too? better not!

#mean = np.mean(X_train, axis=0)
#std = np.std(X_train, axis=0)

#X_train = (X_train - mean)/std
#X_test = (X_test - mean)/std

# Build CNN model

#maybe smaller batchsize
model = get_2d_conv_model(n=n_mfcc)
model_history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    batch_size=16, verbose = 2, epochs=300)




results = get_results(model_history,model,X_test,y_test, ref.labels.unique())
results.confusion_results(X_test, y_test, ref.labels.unique(), model)
results.create_results(model)
results.create_plot(model_history)
print(results.create_results(model))
print("WIESO NA NET")
print(results.confusion_results(X_test, y_test, ref.labels.unique(), model))
results.accuracy_results_gender(X_test, y_test, ref.labels.unique(), model)

save = True
if save:
    # Save model and weights
    model_name = 'Emotion_Model_Conv_earnings.h5'
    save_dir = os.path.join(os.getcwd(), 'saved_models')

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Save model and weights at %s ' % model_path)

    # Save the model to disk
    model_json = model.to_json()
    with open("model_json_conv_earnings.json", "w") as json_file:
        json_file.write(model_json)