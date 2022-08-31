from email.mime import audio
import json
import os
import librosa
import numpy as np
import tensorflow as tf

RATE = 16000

def generate_data() -> tuple:
    '''
    Generates x and y training data for "Hey Ditto" speech recognition.
    '''

    files = os.listdir('raw_data')
    activation_set = []
    background_set = []
    for file_name in files:
        if 'heyditto' in file_name: activation_set.append('raw_data/'+file_name)
        if 'background' in file_name: background_set.append('raw_data/'+file_name)

    print(f'\n\nloaded {len(activation_set)} activation sets and {len(background_set)} background sets\n\n')

    x = []
    y = []

    for activation_phrase in activation_set:
        audio = librosa.load(activation_phrase, sr=16000)
        spect = get_spectrogram(audio[0])
        x.append(spect)
        y.append(1) # activate 

    for background_noise in background_set:
        audio = librosa.load(background_noise, sr=16000)
        spect = get_spectrogram(audio[0])
        x.append(spect)
        y.append(0) # do nothing

    try:
        with open('reinforced_data/conf.json', 'r') as f:
            conf = json.load(f)
            sessions_total = conf['sessions_total']
        for session in range(sessions_total):
            print(f'\n\nprocessing reinforcement session {session}')
            additional_x = np.load(f'reinforced_data/{session}_train_data_x.npy')
            additional_y = np.load(f'reinforced_data/{session}_train_data_y.npy')
            for ndx,spect in enumerate(additional_x):
                x.append(spect)
                y.append(additional_y[ndx])
    except:
        pass

    print('\n\nsaving x and y as .npy cache\n\n')

    np.save('x_data.npy', x)
    np.save('y_data.npy', y)
    return x, y

def get_spectrogram(waveform: list) -> list:
    '''
    Function for converting 16K Hz waveform to spectrogram.
    ref: https://www.tensorflow.org/tutorials/audio/simple_audio
    '''

    import tensorflow as tf

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

if __name__ == "__main__":
    x, y = generate_data()
