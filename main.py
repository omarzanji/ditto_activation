import json
import numpy as np
import os

from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot as plt

import queue
import time as tm 
import sounddevice as sd

# supress tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

TRAIN = False
REINFORCE = False
MODEL_SELECT = 1 # 0 for CNN, 1 for CNN-LSTM
MODEL = ['CNN', 'CNN-LSTM'][MODEL_SELECT]
RATE = 16000

q = queue.Queue()

class HeyDittoNet:
    '''
    HeyDittoNet is a model for recognizing "Hey Ditto" from machine's default mic. 
    '''
    def __init__(self, train=False, model_type='CNN'):
        self.train = train
        self.model_type = model_type
        self.activated = 0
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
        self.model = keras.models.load_model(f'models/HeyDittoNet_{self.model_type}')

    def create_model(self):
        if self.model_type == 'CNN':
            xshape = self.x.shape[1:]
            model = Sequential([
                layers.Input(shape=xshape),
                layers.Resizing(32, 32),
                layers.Normalization(),

                layers.Conv2D(32, (7,7), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2,2)),

                layers.Conv2D(64, (5,5), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2,2)),

                layers.Conv2D(128, (3,3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                # layers.MaxPooling2D(pool_size=(2,2)),
                layers.Flatten(),

                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),

                # layers.Dense(512, activation='relu'),
                # layers.Dropout(0.5),

                layers.Dense(1),
                layers.Activation('sigmoid')
            ])
            
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
            return model
        elif self.model_type == 'CNN-LSTM':
            model = Sequential([
                layers.Conv2D(32, (7,7), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(2,2)),
                layers.Conv2D(64, (5,5), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(2,2)),
                layers.Conv2D(128, (3,3), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(2,2)),
                layers.TimeDistributed(layers.Flatten()),
                layers.LSTM(32),
                layers.Dense(1),
                layers.Activation('sigmoid')
            ])
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
            return model

    def train_model(self, model):
        if self.model_type == 'CNN': 
            epochs = 25
            batch_size = 32
        else: 
            epochs = 8
            batch_size = 64
        name = f'HeyDittoNet_{self.model_type}'
        xtrain, xtest, ytrain, ytest = train_test_split(self.x, self.y, train_size=0.9)
        self.hist = model.fit(xtrain, ytrain, epochs=epochs, verbose=1, batch_size=batch_size)
        self.plot_history(self.hist)
        model.summary()
        ypreds = model.predict(xtest)
        self.ypreds = []
        for y in ypreds:
            if y>=0.6: self.ypreds.append(1)
            else: self.ypreds.append(0)
        self.ypreds = np.array(self.ypreds)
        accuracy = accuracy_score(ytest, self.ypreds)
        print(f'\n\n[Accuracy: {accuracy}]\n\n')
        self.ytest = ytest
        model.save(f'models/{name}')

    def plot_history(self, history):
        plt.figure()
        plt.plot(history.history['loss'])
        plt.title('Model Training Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training loss'], loc='upper right')

    def callback(self, indata, frames, time, status):
        q.put(indata.copy())
        indata = np.array(indata).flatten()
        for vals in indata:
            self.buffer.append(vals)
        if len(self.buffer) >= RATE and self.frames==0:
            self.frames+=frames
            self.buffer = self.buffer[-RATE:]
            spect = self.get_spectrogram(self.buffer)
            pred = self.model.predict(np.expand_dims(spect, 0), verbose=0)
            if pred[0][0] >= 0.6: 
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
                self.frames=0
                    
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
        fs = RATE
        self.buffer = []
        self.train_data_x = []
        self.train_data_y = []
        self.reinforce = reinforce
        self.frames = 0
        # print(sd.query_devices())
        print('\nidle...\n')
        self.start_time = tm.time()
        with sd.InputStream(device=0, samplerate=fs, dtype='float32', latency=None, channels=1, callback=self.callback) as stream:
            while True:
                q.get()
                if self.activated: break
        if reinforce:
            with open('data/reinforced_data/conf.json', 'r') as f:
                conf = json.load(f)
                sesssion_number = conf['sessions_total']
            print('saving to cache...')
            np.save(f'data/reinforced_data/{sesssion_number}_train_data_x.npy', self.train_data_x)
            np.save(f'data/reinforced_data/{sesssion_number}_train_data_y.npy', self.train_data_y)
            with open('data/reinforced_data/conf.json', 'w') as f:
                conf['sessions_total'] = sesssion_number+1
                json.dump(conf, f)
        return self.activated

if __name__ == "__main__":

    network = HeyDittoNet(
        train=TRAIN,
        model_type='CNN-LSTM'
    )
    
    wake = network.listen_for_name()
    if wake: print('name spoken!')