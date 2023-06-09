'''
Hey Ditto net train / test / inference pipeline.

author: Omar Barazanji
date: 2023
'''

import platform
import sys
import json
import sqlite3
import numpy as np
import os

from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from matplotlib import pyplot as plt

# import queue
import time
import sounddevice as sd

# supress tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

TRAIN = False
REINFORCE = False
TFLITE = True
MODEL_SELECT = 1  # 0 for HeyDittoNet-v2, 1 for HeyDittoNet-v1
MODEL = ['HeyDittoNet-v1', 'HeyDittoNet-v2'][MODEL_SELECT]
RATE = 16000
WINDOW = int(RATE/4)
STRIDE = int((RATE - WINDOW)/4)
SENSITIVITY = 0.99


class HeyDittoNet:
    '''
    HeyDittoNet is a model for recognizing "Hey Ditto" from machine's default mic.
    '''

    def __init__(self, train=False, model_type='HeyDittoNet-v1', tflite=True, path='', reinforce=REINFORCE):
        self.train = train
        self.model_type = model_type
        self.tflite = tflite
        self.activated = 0
        self.path = path
        self.reinforce = reinforce
        if train:
            self.load_data()
            model = self.create_model()
            self.train_model(model)
            plt.show()
        else:
            self.retries = 0  # if can't connect to device
            self.load_model()

    def load_data(self):
        try:
            if MODEL == 'HeyDittoNet-v1':
                self.x = np.load('data/x_data_ts.npy', allow_pickle=True)
                self.y = np.load('data/y_data_ts.npy', allow_pickle=True)
            else:
                self.x = np.load('data/x_data.npy', allow_pickle=True)
                self.y = np.load('data/y_data.npy', allow_pickle=True)
            print('Found cached x and y...')

        except BaseException as e:
            print(e)
            print('cached x and y .npy not found! Run create_data.py in data/ ...')
            self.x = []
            self.y = []

    def load_model(self):
        if not self.tflite:
            self.model = keras.models.load_model(
                f'{self.path}models/{self.model_type}')
        else:
            # Load TFLite model and allocate tensors.
            with open(f'{self.path}models/model.tflite', 'rb') as f:
                self.model = f.read()
            self.interpreter = tf.lite.Interpreter(model_content=self.model)
            self.interpreter.allocate_tensors()
            self.input_index = self.interpreter.get_input_details()[0]["index"]
            # self.interpreter.resize_tensor_input(self.input_index, [2, 124, 129, 1])
            self.interpreter.allocate_tensors()
            # self.input_shape = self.interpreter.get_input_details()[0]["shape"]
            # print('\nTFLite Input Shape: ', str(
            #     tuple(self.input_shape)))
            self.output_index = self.interpreter.get_output_details()[
                0]["index"]

    def create_model(self):
        print('Xshape: ', self.x.shape)
        print('Yshape:', self.y.shape)
        if self.model_type == 'HeyDittoNet-v2':
            self.early_stop_callback = tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=3, restore_best_weights=True)
            xshape = self.x.shape[1:]
            T = 4  # number of LSTM time units
            CNN_OUT = 64
            # LSTM_FEATURES = int(T*CNN_OUT)
            model = Sequential([
                layers.Input(shape=xshape),
                layers.Resizing(32, 32),
                layers.Normalization(),

                layers.Conv2D(32, (5, 5), strides=(2, 2),
                              padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(1, 2)),

                layers.Conv2D(64, (3, 3), strides=(2, 2),
                              padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(1, 2), padding='same'),

                layers.Conv2D(CNN_OUT, (3, 3), strides=(2, 2),
                              padding="same", activation="relu"),
                layers.BatchNormalization(),

                layers.Reshape((T, CNN_OUT)),

                layers.LSTM(
                    units=16,
                    input_shape=(None, T, CNN_OUT),
                    return_sequences=False,
                    activation='tanh'
                ),
                # layers.LSTM(16, return_sequences=False, activation='tanh'),

                layers.Dense(32, activation='relu'),

                layers.Dropout(0.2),

                layers.Dense(1, activation='sigmoid'),
            ])
            model.build((None, xshape[0], xshape[1], xshape[2]))
            model.summary()

            model.compile(loss='binary_crossentropy',
                          optimizer='adam', metrics='accuracy')
            return model
        elif self.model_type == 'HeyDittoNet-v1':
            self.early_stop_callback = tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=3, restore_best_weights=True)

            N = 32

            conv_model = Sequential()

            conv_model.add(layers.Resizing(30, 30))
            conv_model.add(layers.Normalization()),

            conv_model.add(layers.Conv2D(
                32, (5, 5), strides=(2, 2), padding="same", activation="relu"))
            conv_model.add(layers.BatchNormalization())
            conv_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

            conv_model.add(layers.Conv2D(
                64, (3, 3), strides=(2, 2), padding="same", activation="relu"))
            conv_model.add(layers.BatchNormalization())
            conv_model.add(layers.MaxPooling2D(
                pool_size=(2, 2), padding='same'))

            conv_model.add(layers.Conv2D(
                64, (3, 3), strides=(2, 2), padding="same", activation="relu"))
            conv_model.add(layers.BatchNormalization())

            conv_model.add(layers.Flatten())
            conv_model.add(layers.Dense(N, activation='relu'))
            conv_model.add(layers.Dropout(0.2))

            model = Sequential()

            model.add(layers.TimeDistributed(conv_model, input_shape=(
                self.x.shape[1], self.x.shape[2], self.x.shape[3], self.x.shape[4])))

            model.add(layers.LSTM(16, return_sequences=True)),
            model.add(layers.LSTM(16, return_sequences=False)),

            model.add(layers.Dense(N, activation='relu'))
            model.add(layers.Dropout(0.2))

            model.add(layers.Dense(1))
            model.add(layers.Activation('sigmoid'))

            conv_model.build(
                (None, self.x.shape[2], self.x.shape[3], self.x.shape[4]))
            conv_model.summary()
            model.build(
                (None, self.x.shape[1], self.x.shape[2], self.x.shape[3]))
            model.summary()

            model.compile(loss='binary_crossentropy',
                          optimizer='adam', metrics='accuracy')

            return model

    def train_model(self, model):
        if self.model_type == 'HeyDittoNet-v2':
            epochs = 100
            batch_size = 32
        else:
            epochs = 100
            batch_size = 64
        name = f'{self.model_type}'
        xtrain, xtest, ytrain, ytest = train_test_split(
            self.x, self.y, train_size=0.9)
        self.hist = model.fit(xtrain, ytrain, epochs=epochs, verbose=1,
                              batch_size=batch_size, callbacks=[self.early_stop_callback])
        self.plot_history(self.hist)
        # model.summary()
        ypreds = model.predict(xtest)
        self.ypreds = []
        for y in ypreds:
            if y >= 0.6:
                self.ypreds.append(1)
            else:
                self.ypreds.append(0)
        self.ypreds = np.array(self.ypreds)
        accuracy = accuracy_score(ytest, self.ypreds)
        print(f'\n\n[Accuracy: {accuracy}]\n\n')
        self.ytest = ytest
        model.save(f'models/{name}')
        self.model = model

    def plot_history(self, history):
        plt.figure()
        plt.plot(history.history['loss'])
        plt.title('Model Training Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training loss'], loc='upper right')

    def normalize_audio(self, signal, rms_level=-1):
        """
        Normalize the signal.
        ref: https://superkogito.github.io/blog/2020/04/30/rms_normalization.html
        """
        try:
            signal = np.array(signal).astype('float32')

            # linear rms level and scaling factor
            r = 10**(rms_level / 10.0)
            a = np.sqrt((len(signal) * r**2) / np.sum(signal**2))

            # normalize
            y = signal * a
        except BaseException as e:
            print(e)
            return signal

        return y

    def callback(self, indata, frames, time_, status):
        time_, status = None, None
        indata = np.array(indata).flatten()
        for vals in indata:
            self.buffer.append(vals)
        if len(self.buffer) >= RATE and self.frames == 0:
            self.frames += frames
            self.buffer = self.buffer[-RATE:]
            normalized = self.normalize_audio(self.buffer)
            if self.model_type == 'HeyDittoNet-v1':
                spect = self.get_spectrograms(normalized)
            else:
                spect = self.get_spectrogram(normalized)
            if self.tflite:
                if self.model_type == 'HeyDittoNet-v1':
                    self.interpreter.set_tensor(
                        self.input_index, np.expand_dims(spect, 0))
                    self.interpreter.invoke()
                    # self.interpreter.set_tensor(
                    #     self.input_index, np.expand_dims(spect[1], 0))
                    pred = self.interpreter.get_tensor(self.output_index)
                else:
                    self.interpreter.set_tensor(
                        self.input_index, np.expand_dims(spect, 0))
                    self.interpreter.invoke()
                    pred = self.interpreter.get_tensor(self.output_index)
            else:
                pred = self.model(np.expand_dims(spect, 0))
            K.clear_session()
            if pred[0][0] >= SENSITIVITY:
                # used for when this module is a thread that's always on (requires 2 mics in Ditto Assistant)
                if self.activation_time:

                    # if prev activation and current are 2 seconds apart (filter double triggers)
                    if int(time.time()) - self.activation_time > 2:
                        print(f'Activated with confidence: {pred[0][0]*100}%')
                        self.activation_requests.activated = 1
                        # log new activation time
                        self.activation_time = int(time.time())
                        if self.reinforce:
                            self.train_data_x.append(normalized)
                            self.train_data_y.append(0)
                    else:
                        pass  # do nothing on double triggers..

                else:  # first activation from boot, always allow
                    print(f'Activated with confidence: {pred[0][0]*100}%')
                    self.activation_requests.activated = 1
                    # log first activation time
                    self.activation_time = int(time.time())
                    if self.reinforce:
                        self.train_data_x.append(normalized)
                        self.train_data_y.append(0)
            else:
                time.sleep(0.001)
                # print(f'{pred[0][0]*100}%')
                pass

            # garbage
            pred = None
            spect = None

        if self.frames > 0:
            self.frames += frames
            if self.frames >= RATE/4:
                self.frames = 0

        # garbage
        self = None
        normalized = None
        indata = None

    def get_spectrogram(waveform: list) -> list:
        '''
        Function for converting 16K Hz waveform to spectrogram.
        ref: https://www.tensorflow.org/tutorials/audio/simple_audio
        '''

        # Zero-padding for an audio waveform with less than 16,000 samples.
        input_len = RATE
        waveform = waveform[:input_len]
        zero_padding = tf.zeros(
            [RATE] - tf.shape(waveform),
            dtype=tf.float32)
        # Cast the waveform tensors' dtype to float32.
        waveform = tf.cast(waveform, dtype=tf.float32)
        # Concatenate the waveform with `zero_padding`, which ensures all audio
        # clips are of the same length.
        equal_length = tf.concat([waveform, zero_padding], 0)
        # Convert the waveform to a spectrogram via a STFT.
        # spectrogram = tf.signal.stft(
        #     equal_length, frame_length=255, frame_step=128)
        # Obtain the magnitude of the STFT.
        # spectrogram = tf.abs(spectrogram)
        # Add a `channels` dimension, so that the spectrogram can be used
        # as image-like input data with convolution layers (which expect
        # shape (`batch_size`, `height`, `width`, `channels`).
        # spectrogram = spectrogram[..., tf.newaxis]
        fbank_feat = logfbank(equal_length, 16000, nfilt=32)
        spectrogram = fbank_feat[..., tf.newaxis]
        return spectrogram

    def get_spectrograms(self, waveform: list, stride=STRIDE) -> list:
        '''
        Function for converting 16K Hz waveform to two half-second spectrograms for TIME_SERIES model.
        ref: https://www.tensorflow.org/tutorials/audio/simple_audio
        '''

        # Zero-padding for an audio waveform with less than 16,000 samples.
        input_len = RATE
        waveform = waveform[:input_len]
        zero_padding = tf.zeros(
            [RATE] - tf.shape(waveform),
            dtype=tf.float32)
        waveform = tf.cast(waveform, dtype=tf.float32)
        waveform = tf.concat([waveform, zero_padding], 0)
        chunk_size = WINDOW
        spectrograms = []
        for i in range(0, len(waveform)-WINDOW, STRIDE):
            data = waveform[i:i+chunk_size]

            # Convert the waveform to a spectrogram via a STFT.
            spectrogram = tf.signal.stft(
                data, frame_length=255, frame_step=128)
            # Obtain the magnitude of the STFT.
            spectrogram = tf.abs(spectrogram)
            # Add a `channels` dimension, so that the spectrogram can be used
            # as image-like input data with convolution layers (which expect
            # shape (`batch_size`, `height`, `width`, `channels`).
            spectrogram = spectrogram[..., tf.newaxis]

            spectrograms.append(spectrogram)

        return spectrograms

    def save_reinforce_trigger(self):
        with open(f'{self.path}data/reinforce_background/conf.json', 'r') as f:
            conf = json.load(f)
            sesssion_number = conf['sessions_total']
        print('saving to cache...')
        np.save(
            f'{self.path}data/reinforce_background/{sesssion_number}_train_data_x.npy', self.train_data_x)
        np.save(
            f'{self.path}data/reinforce_background/{sesssion_number}_train_data_y.npy', self.train_data_y)
        with open(f'{self.path}data/reinforce_background/conf.json', 'w') as f:
            conf['sessions_total'] = sesssion_number+1
            json.dump(conf, f)

    def listen_for_name(self):
        # sampling rate
        fs = RATE

        # callback resources
        self.buffer = []
        self.train_data_x = []
        self.train_data_y = []
        self.frames = 0
        self.activation_time = None

        # import activation_requests
        if self.path == 'modules/ditto_activation/':
            # import Ditto Activation requests
            from modules.ditto_activation.activation_requests import ActivationRequests
        elif self.path == '':
            from activation_requests import ActivationRequests
        self.activation_requests = ActivationRequests()
        if self.activation_requests.mic_on:
            print('\nidle...\n')
        else:
            print('\nmic muted...\n')

        if 'linux' in platform.platform().lower():
            device_id = 1
        else:
            device_id = sd.default.device[0]

        try:
            with sd.InputStream(device=device_id,
                                samplerate=fs,
                                dtype='float32',
                                latency='low',
                                channels=1,
                                callback=self.callback,
                                blocksize=int(RATE/4)) as stream:
                while True:

                    self.activation_requests.check_for_gesture()
                    self.activation_requests.check_for_request()
                    if not self.activation_requests.mic_on:
                        self.buffer = []
                    activated = self.activation_requests.activated
                    if activated and self.reinforce:
                        self.save_reinforce_trigger()
                        stream.close()
                        return 1
                    if activated:
                        stream.close()
                        return 1

        except KeyboardInterrupt:
            stream.close()
            return -1
        except BaseException as e:
            print(e)
            stream.close()
            return -1

    def main_loop(self):
        if self.reinforce:
            while True:
                wake = self.listen_for_name()
                if wake == -1:
                    break

        elif not self.train:
            wake = self.listen_for_name()


if __name__ == "__main__":

    network = HeyDittoNet(
        train=TRAIN,
        model_type=MODEL,
        tflite=TFLITE,
        reinforce=REINFORCE
    )
    network.main_loop()
