'''
Hey Ditto net train / test / inference pipeline. 

author: Omar Barazanji
date: 2023
'''

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
MODEL_SELECT = 0  # 0 for CNN, 1 for CNN-LSTM
MODEL = ['CNN', 'CNN-LSTM'][MODEL_SELECT]
RATE = 16000
SENSITIVITY = 0.99


class HeyDittoNet:
    '''
    HeyDittoNet is a model for recognizing "Hey Ditto" from machine's default mic. 
    '''

    def __init__(self, train=False, model_type='CNN', tflite=False, path=''):
        # self.q = queue.Queue()
        self.train = train
        self.model_type = model_type
        self.tflite = tflite
        self.activated = 0
        self.path = path
        if train:
            self.load_data()
            # if model_type == 'CNN-LSTM': self.create_time_series()
            model = self.create_model()
            self.train_model(model)
            plt.show()
        else:
            self.load_model()

    def load_data(self):
        try:
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
                f'{self.path}models/HeyDittoNet_{self.model_type}')
        else:
            # Load TFLite model and allocate tensors.
            with open(f'{self.path}models/model.tflite', 'rb') as f:
                self.model = f.read()
            self.interpreter = tf.lite.Interpreter(model_content=self.model)
            self.interpreter.allocate_tensors()
            self.input_index = self.interpreter.get_input_details()[0]["index"]
            self.output_index = self.interpreter.get_output_details()[
                0]["index"]

    def create_model(self):
        if self.model_type == 'CNN':
            self.early_stop_callback = tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=5)
            xshape = self.x.shape[1:]
            model = Sequential([
                layers.Input(shape=xshape),
                layers.Resizing(32, 32),
                layers.Normalization(),

                layers.Conv2D(32, (5, 5), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2)),

                # layers.Conv2D(64, (5, 5), padding="same", activation="relu"),
                # layers.BatchNormalization(),
                # layers.MaxPooling2D(pool_size=(2, 2)),

                # layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                # layers.BatchNormalization(),
                # layers.MaxPooling2D(pool_size=(2,2)),
                layers.Flatten(),

                layers.Dense(16, activation='relu'),
                # layers.Dropout(0.3),

                # layers.Dense(512, activation='relu'),
                # layers.Dropout(0.5),

                layers.Dense(1),
                layers.Activation('sigmoid')
            ])

            model.compile(loss='binary_crossentropy',
                          optimizer='adam', metrics='accuracy')
            return model
        elif self.model_type == 'CNN-LSTM':
            self.early_stop_callback = tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=5)
            model = Sequential([
                layers.Resizing(32, 32),

                # layers.Conv2D(32, (5, 5), padding="same", activation="relu"),
                # layers.BatchNormalization(),
                # layers.MaxPooling2D(pool_size=(2, 2)),

                # layers.Conv2D(64, (5, 5), padding="same", activation="relu"),
                # layers.BatchNormalization(),
                # layers.MaxPooling2D(pool_size=(2, 2)),

                layers.Conv2D(32, (5, 5), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.TimeDistributed(layers.Flatten()),
                layers.LSTM(8),
                layers.Dense(16, activation='relu'),
                # layers.Dropout(0.5),
                layers.Dense(1),
                layers.Activation('sigmoid')
            ])
            model.compile(loss='binary_crossentropy',
                          optimizer='adam', metrics='accuracy')
            return model

    def train_model(self, model):
        if self.model_type == 'CNN':
            epochs = 30
            batch_size = 32
        else:
            epochs = 30
            batch_size = 32
        name = f'HeyDittoNet_{self.model_type}'
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

    def callback(self, indata, frames, time, status):
        # self.q.put(indata.copy())
        indata = np.array(indata).flatten()
        for vals in indata:
            self.buffer.append(vals)
        if len(self.buffer) >= RATE and self.frames == 0:
            self.frames += frames
            self.buffer = self.buffer[-RATE:]
            spect = self.get_spectrogram(self.buffer)
            if self.tflite:
                self.interpreter.set_tensor(
                    self.input_index, np.expand_dims(spect, 0))
                self.interpreter.invoke()
                pred = self.interpreter.get_tensor(self.output_index)
            else:
                pred = self.model(np.expand_dims(spect, 0))
            K.clear_session()
            if pred[0][0] >= SENSITIVITY:
                print(f'Activated with confidence: {pred[0][0]*100}%')
                self.activated = 1
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

    def listen_for_name(self, reinforce=False):
        self.activated = 0
        self.timeout = time.time() + 4  # 4 seconds (used for gesture recognition)
        # set to false to interrupt elsewhere (might remove this)
        self.running = True
        self.prompt = ""  # used for GUI skip wake and skip STT (inject prompt)
        # set to true in check_for_request function to skip STT module
        self.inject_prompt = False
        self.gesture = ""  # grabbed from gesture_recognition module
        # set to true in check_for_gesture function to skip wake using gesture
        self.gesture_activation = False
        self.reset_conversation = False  # set to true in check_for_request
        self.palm_count = 0  # used to filter false positives
        self.like_count = 0
        self.dislike_count = 0
        fs = RATE
        self.buffer = []
        self.train_data_x = []
        self.train_data_y = []
        self.reinforce = reinforce
        self.frames = 0
        # print(sd.query_devices())
        print('\nidle...\n')
        self.start_time = time.time()
        with sd.InputStream(device=sd.default.device[0], samplerate=fs, dtype='float32', latency=None, channels=1, callback=self.callback) as stream:
            while True:
                self.check_for_request()
                self.check_for_gesture()
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

                if self.activated or self.running == False:
                    break

        return self.activated

    def check_for_gesture(self):
        '''
        Checks for gesture to skip wake.
        '''
        def reset_counts():
            self.like_count = 0
            self.dislike_count = 0
            self.palm_count = 0

        if time.time() > self.timeout:
            # print('gesture check timeout')
            self.timeout = time.time() + 4
            # reset gesture counters
            reset_counts()
        try:
            SQL = sqlite3.connect(f'ditto.db')
            cur = SQL.cursor()
            req = cur.execute("select * from gestures")
            req = req.fetchall()
            like_gest = False
            dislike_gest = False
            palm_gest = False
            for i in req:
                if 'like' in i:
                    like_gest = True
                    print('like')
                if 'dislike' in i:
                    dislike_gest = True
                    print('dislike')
                if 'palm' in i:
                    print('palm')
                    palm_gest = True
            if like_gest or dislike_gest or palm_gest:
                if like_gest:
                    self.like_count += 1
                if dislike_gest:
                    self.dislike_count += 1
                if palm_gest:
                    self.palm_count += 1

                if self.like_count == 2:
                    reset_counts()
                    print("\n[Activated from Like Gesture]\n")
                    self.running = False
                    self.gesture_activation = True
                    self.gesture = 'like'

                if self.dislike_count == 2:
                    reset_counts()
                    print("\n[Activated from Dislike Gesture]\n")
                    self.running = False
                    self.gesture_activation = True
                    self.gesture = 'dislike'

                if self.palm_count == 2:
                    reset_counts()
                    print("\n[Activated from Palm Gesture]\n")
                    self.running = False
                    self.gesture_activation = True
                    self.gesture = 'palm'
            cur.execute("DELETE FROM gestures")
            SQL.commit()
            SQL.close()
        except BaseException as e:
            pass
            # print(e)
        if self.gesture_activation:
            self.activated = 1

    def check_for_request(self):
        ''' 
        Checks if the user sent a prompt from the client GUI.
        '''
        try:

            SQL = sqlite3.connect(f'ditto.db')
            cur = SQL.cursor()
            req = cur.execute("select * from ditto_requests")
            req = req.fetchone()
            if req[0] == "prompt":
                self.prompt = req[1]
                print("\n[GUI prompt received]\n")
                cur.execute("DROP TABLE ditto_requests")
                SQL.close()
                self.running = False
                self.inject_prompt = True
                self.activated = 1
            if req[0] == "resetConversation":
                print("\n[Reset conversation request received]\n")
                cur.execute("DROP TABLE ditto_requests")
                SQL.close()
                self.running = False
                self.reset_conversation = True
                self.activated = 1

        except BaseException as e:
            pass
            # print(e)


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
        wake = network.listen_for_name()
        if wake:
            print('name spoken!')
