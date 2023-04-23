'''
Hey Ditto net train / test / inference pipeline.

author: Omar Barazanji
date: 2023
'''

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
SENSITIVITY = 0.99

PATH = ''
if len(sys.argv) >= 2:
    PATH = sys.argv[1]


class HeyDittoNet:
    '''
    HeyDittoNet is a model for recognizing "Hey Ditto" from machine's default mic.
    '''

    def __init__(self, train=False, model_type='HeyDittoNet-v2', tflite=True, path=PATH):
        # self.q = queue.Queue()
        self.train = train
        self.model_type = model_type
        self.tflite = tflite
        self.activated = 0
        self.path = path
        if train:
            self.load_data()
            model = self.create_model()
            self.train_model(model)
            plt.show()
        else:
            self.load_model()

    def load_data(self):
        try:
            if MODEL == 'HeyDittoNet-v1':
                self.x = np.load('data/x_data_lstm.npy', allow_pickle=True)
                self.y = np.load('data/y_data_lstm.npy', allow_pickle=True)
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
        if self.model_type == 'HeyDittoNet-v2':
            self.early_stop_callback = tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=3, restore_best_weights=True)
            xshape = self.x.shape[1:]
            model = Sequential([
                layers.Input(shape=xshape),
                layers.Resizing(32, 32),
                layers.Normalization(),

                layers.Conv2D(32, (5, 5), strides=(2, 2),
                              padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2)),

                layers.Conv2D(64, (5, 5), strides=(4, 4),
                              padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

                layers.Conv2D(128, (3, 3), strides=(4, 4),
                              padding="same", activation="relu"),
                layers.BatchNormalization(),
                # layers.MaxPooling2D(pool_size=(2, 2)),
                # layers.Flatten(),
                # layers.Dense(32, activation='relu'),
                layers.Reshape((2, 64)),

                layers.LSTM(8, input_shape=(None, 2, 64), activation='relu'),
                # layers.Dropout(0.3),

                layers.Dense(32, activation='relu'),
                # layers.Dropout(0.5),

                layers.Dense(1),
                layers.Activation('sigmoid')
            ])

            model.compile(loss='binary_crossentropy',
                          optimizer='adam', metrics='accuracy')
            return model
        elif self.model_type == 'HeyDittoNet-v1':
            self.early_stop_callback = tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=10, restore_best_weights=True)

            N = 32

            conv_model = Sequential()
            conv_model.add(layers.Resizing(32, 32))
            conv_model.add(layers.Normalization()),
            conv_model.add(layers.Conv2D(
                32, (5, 5), strides=(2, 2), padding="same", activation="relu"))
            conv_model.add(layers.BatchNormalization())
            conv_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

            conv_model.add(layers.Conv2D(
                64, (5, 5), strides=(4, 4), padding="same", activation="relu"))
            conv_model.add(layers.BatchNormalization())
            conv_model.add(layers.MaxPooling2D(
                pool_size=(2, 2), padding='same'))

            conv_model.add(layers.Conv2D(
                128, (3, 3), strides=(4, 4), padding="same", activation="relu"))
            conv_model.add(layers.BatchNormalization())
            # conv_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

            # conv_model.add(layers.Conv2D(
            # 64, (3, 3), strides=(3, 3), padding="same", activation="relu"))
            # conv_model.add(layers.BatchNormalization())
            # conv_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            conv_model.add(layers.Flatten())
            conv_model.add(layers.Dense(N, activation='relu'))

            model = Sequential()

            model.add(layers.TimeDistributed(conv_model, input_shape=(
                self.x.shape[1], self.x.shape[2], self.x.shape[3], self.x.shape[4])))

            model.add(layers.LSTM(8)),

            model.add(layers.Dense(int(N/4), activation='relu'))
            # model.add(layers.Dropout(0.5))
            model.add(layers.Dense(1))
            model.add(layers.Activation('sigmoid'))
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
        model.summary()
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

    def callback(self, indata, frames, time_, status):
        # self.q.put(indata.copy())
        indata = np.array(indata).flatten()
        for vals in indata:
            self.buffer.append(vals)
        if len(self.buffer) >= RATE and self.frames == 0:
            self.frames += frames
            self.buffer = self.buffer[-RATE:]
            if self.model_type == 'HeyDittoNet-v1':
                spect = self.get_spectrograms(self.buffer)
            else:
                spect = self.get_spectrogram(self.buffer)
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
                if self.activation_time:

                    # if prev activation and current are 2 seconds apart (filter double triggers)
                    if int(time.time()) - self.activation_time > 2:
                        print(f'Activated with confidence: {pred[0][0]*100}%')
                        self.activated = 1
                        # log new activation time
                        self.activation_time = int(time.time())
                        self.send_ditto_wake()  # send wake to database to start wake sequence in Ditto Assistant
                        if self.reinforce:
                            self.train_data_x.append(spect)
                            self.train_data_y.append(0)
                    else:
                        pass  # do nothing on double triggers..

                else:  # first activation from boot, always allow
                    print(f'Activated with confidence: {pred[0][0]*100}%')
                    self.activated = 1
                    # log first activation time
                    self.activation_time = int(time.time())
                    self.send_ditto_wake()
                    if self.reinforce:
                        self.train_data_x.append(spect)
                        self.train_data_y.append(0)
            else:
                # print(f'{pred[0][0]*100}%')
                pass
        if self.frames > 0:
            self.frames += frames
            if self.frames >= RATE/4:
                self.frames = 0

    def get_spectrogram(self, waveform: list) -> list:
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
        spectrogram = tf.signal.stft(
            equal_length, frame_length=255, frame_step=128)
        # Obtain the magnitude of the STFT.
        spectrogram = tf.abs(spectrogram)
        # Add a `channels` dimension, so that the spectrogram can be used
        # as image-like input data with convolution layers (which expect
        # shape (`batch_size`, `height`, `width`, `channels`).
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    def get_spectrograms(self, waveform: list) -> list:
        '''
        Function for converting 16K Hz waveform to two half-second spectrograms for LSTM model.
        ref: https://www.tensorflow.org/tutorials/audio/simple_audio
        '''

        import tensorflow as tf

        # Zero-padding for an audio waveform with less than 16,000 samples.
        input_len = RATE
        waveform = waveform[:input_len]
        waveform1 = waveform[:8000]  # 1st half second
        waveform2 = waveform[8000:]  # 2nd half second

        zero_padding1 = tf.zeros(
            [RATE] - tf.shape(waveform1),
            dtype=tf.float32)
        zero_padding2 = tf.zeros(
            [RATE] - tf.shape(waveform2),
            dtype=tf.float32)
        # Cast the waveform tensors' dtype to float32.
        waveform1 = tf.cast(waveform1, dtype=tf.float32)
        waveform2 = tf.cast(waveform2, dtype=tf.float32)
        # Concatenate the waveform with `zero_padding`, which ensures all audio
        # clips are of the same length.
        equal_length1 = tf.concat([waveform1, zero_padding1], 0)
        equal_length2 = tf.concat([waveform2, zero_padding2], 0)

        # Convert the waveform to a spectrogram via a STFT.
        spectrogram1 = tf.signal.stft(
            equal_length1, frame_length=255, frame_step=128)
        # Obtain the magnitude of the STFT.
        spectrogram1 = tf.abs(spectrogram1)
        # Add a `channels` dimension, so that the spectrogram can be used
        # as image-like input data with convolution layers (which expect
        # shape (`batch_size`, `height`, `width`, `channels`).
        spectrogram1 = spectrogram1[..., tf.newaxis]

        spectrogram2 = tf.signal.stft(
            equal_length2, frame_length=255, frame_step=128)
        # Obtain the magnitude of the STFT.
        spectrogram2 = tf.abs(spectrogram2)
        # Add a `channels` dimension, so that the spectrogram can be used
        # as image-like input data with convolution layers (which expect
        # shape (`batch_size`, `height`, `width`, `channels`).
        spectrogram2 = spectrogram2[..., tf.newaxis]

        return spectrogram1, spectrogram2

    def listen_for_name(self, reinforce=False):
        self.activated = 0
        self.timeout = time.time() + 4  # 4 seconds (used for gesture recognition)
        self.running = True
        fs = RATE
        self.buffer = []
        self.train_data_x = []
        self.train_data_y = []
        self.reinforce = reinforce
        self.frames = 0
        self.activation_time = None
        self.start_time = time.time()
        with sd.InputStream(device=sd.default.device[0],
                            samplerate=fs,
                            dtype='float32',
                            latency=None,
                            channels=1,
                            callback=self.callback,
                            blocksize=1024) as stream:
            try:
                while True:
                    time.sleep(0.001)
                    if self.activated and reinforce:
                        with open(f'{self.path}data/reinforced_data/conf.json', 'r') as f:
                            conf = json.load(f)
                            sesssion_number = conf['sessions_total']
                        print('saving to cache...')
                        np.save(
                            f'{self.path}data/reinforced_data/{sesssion_number}_train_data_x.npy', self.train_data_x)
                        np.save(
                            f'{self.path}data/reinforced_data/{sesssion_number}_train_data_y.npy', self.train_data_y)
                        with open(f'{self.path}data/reinforced_data/conf.json', 'w') as f:
                            conf['sessions_total'] = sesssion_number+1
                            json.dump(conf, f)
                        return 1
            except KeyboardInterrupt:
                stream.close()
                exit()

    def send_ditto_wake(self):
        SQL = sqlite3.connect(f'ditto.db')
        cur = SQL.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS ditto_requests(request VARCHAR, action VARCHAR)")
        SQL.commit()
        cur.execute(
            "INSERT INTO ditto_requests VALUES('activation', 'activate')")
        SQL.commit()
        SQL.close()


if __name__ == "__main__":

    network = HeyDittoNet(
        train=TRAIN,
        model_type=MODEL,
        tflite=TFLITE
    )
    if REINFORCE:
        while True:
            wake = network.listen_for_name(REINFORCE)

    elif not TRAIN:
        network.listen_for_name()
