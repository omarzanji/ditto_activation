import json
import numpy as np

from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot as plt
    
import time as tm 
import sounddevice as sd

RATE = 16000

class HeyDittoNet:

    def __init__(self):
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
        self.model = keras.models.load_model('models/HeyDittoNet')

    def create_model(self):
        xshape = self.x.shape[1:]
        model = Sequential([
            layers.Input(shape=xshape),
            layers.Resizing(32, 32),
            layers.Normalization(),
            layers.Conv2D(32, 3, activation='relu'),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            # layers.Dense(64, activation='relu'),
            # layers.Dropout(0.2),
            layers.Dense(1),
            layers.Activation('sigmoid')
        ])
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
        return model

    def train_model(self, model):
        name = 'HeyDittoNet'
        xtrain, xtest, ytrain, ytest = train_test_split(self.x, self.y, train_size=0.9)
        self.hist = model.fit(xtrain, ytrain, epochs=100, verbose=1)
        self.plot_history(self.hist)
        model.summary()
        ypreds = model.predict(xtest)
        self.ypreds = []
        for y in ypreds:
            if y>=0.9: self.ypreds.append(1)
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
        
        indata = np.array(indata).flatten()
        for vals in indata:
            self.buffer.append(vals)
        if len(self.buffer) >= RATE and self.frames==0:
            self.frames+=frames
            self.buffer = self.buffer[-RATE:]
            spect = self.get_spectrogram(self.buffer)
            pred = self.model.predict(np.expand_dims(spect, 0))
            if pred[0][0] >= 0.9: 
                print(f'Activated with confidence: {pred[0][0]*100}%')
                if self.reinforce:
                    self.train_data_x.append(spect)
                    self.train_data_y.append(0)
            else: print(pred[0][0])
        if self.frames > 0:
            self.frames += frames
            if self.frames >= RATE/4:
                self.frames=0
        self.time = tm.time() - self.start_time
        
        
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

    def test_model(self, reinforce=False):

        fs = RATE
        self.buffer = []
        self.train_data_x = []
        self.train_data_y = []
        self.reinforce = reinforce
        self.frames = 0
        print(sd.query_devices())

        try:
            self.start_time = tm.time()
            with sd.InputStream(device=0, samplerate=fs, dtype='float32', latency=None, channels=1, callback=self.callback):
                input()
        except KeyboardInterrupt:
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

if __name__ == "__main__":
    network = HeyDittoNet()
    train = False
    if train:
        model = network.create_model()
        network.train_model(model)
        plt.show()
    else:
        network.load_model()
        network.test_model(reinforce=False)
