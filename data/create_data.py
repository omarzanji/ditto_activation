import json
import random
import os
import time
import librosa
import numpy as np
import tensorflow as tf
import sounddevice
from pydub import AudioSegment
from pydub import effects
from pydub.playback import play

# supress tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

RATE = 16000
LSTM = False


def white_noise(sample, amount=0.005):
    # Add white noise to sample with scaling
    wn = np.random.randn(len(sample))
    data_wn = sample + amount*wn
    return data_wn


def stretch(sample, rate=1):
    max_len = 16000
    data = librosa.effects.time_stretch(sample, rate=rate)
    if len(data) > max_len:  # if gets too big, snip the end
        data = data[:max_len]
    elif len(data) < 16000:  # if gets too small, pad with zeros
        data = np.pad(data, (0, max(0, max_len - len(data))))
    return data


def lower_volume(sample, db=10):
    raw_audio = AudioSegment.from_wav(sample)
    audio = effects.normalize(raw_audio)
    audio = audio - db
    samples = audio.get_array_of_samples()
    audio._spawn(samples)  # write to samples
    return np.array(samples).astype(np.float32, order='C') / 32768.0


def normalize_audio(sample):
    audio = AudioSegment.from_wav(sample)
    audio_norm = effects.normalize(audio)
    samples = audio_norm.get_array_of_samples()
    max_len = 16000
    if len(samples) > max_len:  # if gets too big, snip the end
        # print(sample)
        # print('sample too long (snipping)')
        # print(len(samples))
        samples = samples[:max_len]
    elif len(samples) < 16000:  # if gets too small, pad with zeros
        # print('sample too short (padding)')
        # print(len(samples))
        samples = np.pad(samples, (0, max(0, max_len - len(samples))))
    audio_norm._spawn(samples)  # write to samples
    return np.array(samples).astype(np.float32, order='C') / 32768.0


def combine_with(activation, background):
    a_audio = AudioSegment.from_wav(activation)
    a_audio_norm = effects.normalize(a_audio)
    b_audio = AudioSegment.from_wav(background)
    b_audio_norm = effects.normalize(b_audio) - np.random.randint(8, 10)
    audio = a_audio_norm.overlay(b_audio_norm)
    samples = audio.get_array_of_samples()  # write to samples
    return np.array(samples).astype(np.float32, order='C') / 32768.0


def generate_data() -> tuple:
    '''
    Generates x and y training data for "Hey Ditto" speech recognition.
    '''
    files = os.listdir('raw_data')
    activation_set = []
    background_set = []
    for file_name in files:
        if 'heyditto' in file_name.replace(' ', '').lower():
            activation_set.append('raw_data/'+file_name)
        if 'background' in file_name:
            background_set.append('raw_data/'+file_name)

    print(
        f'\n\nloaded {len(activation_set)} activation sets and {len(background_set)} background sets\n\n')
    print('processing...\n')
    x = []
    y = []

    t_cnt, f_cnt = 0, 0
    random.shuffle(activation_set)
    # random.shuffle(background_set) # probably shouldn't shuffle for RNN to understand people talking in order
    for activation_phrase, background_noise in zip(activation_set, background_set):
        # audio = librosa.load(activation_phrase, sr=16000)
        audio = [normalize_audio(activation_phrase)]
        # data augmentation
        audio_quiet = lower_volume(
            activation_phrase, db=np.random.randint(10, 15))
        # audio_really_quiet = lower_volume(activation_phrase, db=20)
        # audio_very_quiet = lower_volume(activation_phrase, db=30)
        audio_noise = white_noise(audio[0], amount=random.uniform(0.002, 0.02))
        audio_stretch_low = stretch(audio[0], rate=random.uniform(0.88, 0.99))
        audio_stretch_high = stretch(audio[0], rate=random.uniform(1.1, 1.3))
        combined_audio = combine_with(activation_phrase, background_noise)
        # print(audio[0])
        # print(audio_quiet)
        # print(get_spectrogram(audio[0]))
        # print(get_spectrogram(audio_normalized))
        # sounddevice.play(audio[0], samplerate=16000)
        # time.sleep(1)
        # sounddevice.play(audio_noise, samplerate=16000)
        # time.sleep(1)
        # sounddevice.play(audio_stretch_low, samplerate=16000)
        # time.sleep(1)
        # sounddevice.play(audio_stretch_high, samplerate=16000)
        # time.sleep(1)
        # sounddevice.play(audio_quiet, samplerate=16000)
        # time.sleep(1)
        # sounddevice.play(audio_really_quiet, samplerate=16000)
        # time.sleep(2)
        # sounddevice.play(combined_audio, samplerate=16000)
        # time.sleep(1)
        # exit()
        if LSTM:
            spect = get_spectrograms(audio[0])
            x.append(spect)
            return x
            y.append(1)  # activate
            x.append(get_spectrograms(audio_quiet))
            y.append(1)
            # x.append(get_spectrogram(audio_really_quiet))
            # y.append(1)
            # x.append(get_spectrogram(audio_very_quiet))
            # y.append(1)
            x.append(get_spectrograms(audio_noise))
            y.append(1)
            x.append(get_spectrograms(audio_stretch_low))
            y.append(1)
            x.append(get_spectrograms(audio_stretch_high))
            y.append(1)
            x.append(get_spectrograms(combined_audio))
            y.append(1)

        else:
            spect = get_spectrogram(audio[0])
            x.append(spect)
            y.append(1)  # activate
            x.append(get_spectrogram(audio_quiet))
            y.append(1)
            # x.append(get_spectrogram(audio_really_quiet))
            # y.append(1)
            # x.append(get_spectrogram(audio_very_quiet))
            # y.append(1)
            x.append(get_spectrogram(audio_noise))
            y.append(1)
            x.append(get_spectrogram(audio_stretch_low))
            y.append(1)
            x.append(get_spectrogram(audio_stretch_high))
            y.append(1)
            x.append(get_spectrogram(combined_audio))
            y.append(1)

        t_cnt += 6  # true class count

    for ndx, background_noise in enumerate(background_set):
        count = ndx + 1
        # audio = librosa.load(background_noise, sr=16000)
        audio = [normalize_audio(background_noise)]
        if LSTM:
            spect = get_spectrograms(audio[0])
        else:
            spect = get_spectrogram(audio[0])

        x.append(spect)
        y.append(0)
        N = 4000
        if count < N or\
                'Neural' in background_noise or 'Wavenet' in background_noise or\
                'Standard' in background_noise or 'News' in background_noise:  # apply augmentations to N false samples
            audio_noise = white_noise(
                audio[0], amount=random.uniform(0.002, 0.2))

            audio_loud = lower_volume(
                background_noise, db=np.random.randint(-30, -10)+np.random.rand())  # Increases volume

            audio_quiet = lower_volume(background_noise, db=np.random.randint(
                10, 20)+np.random.rand())  # Decreases volume

            if LSTM:
                x.append(get_spectrograms(audio_noise))
                y.append(0)
                x.append(get_spectrograms(audio_loud))
                y.append(0)
                x.append(get_spectrograms(audio_quiet))
                y.append(0)
            else:
                x.append(get_spectrogram(audio_noise))
                y.append(0)
                x.append(get_spectrogram(audio_loud))
                y.append(0)
                x.append(get_spectrogram(audio_quiet))
                y.append(0)

            f_cnt += 4  # false class count

            # sounddevice.play(audio[0], samplerate=16000)
            # time.sleep(1)

            # sounddevice.play(audio_noise, samplerate=16000)
            # time.sleep(1)

            # sounddevice.play(audio_loud, samplerate=16000)
            # time.sleep(1)

            # sounddevice.play(audio_quiet, samplerate=16000)
            # time.sleep(1)
        else:
            f_cnt += 1  # false class count
    if not LSTM:
        try:
            print('loading reinforcement sessions...\n')
            with open('reinforced_data/conf.json', 'r') as f:
                conf = json.load(f)
                sessions_total = conf['sessions_total']
            for session in range(sessions_total):
                print(f'\nprocessing reinforcement session {session}')
                additional_x = np.load(
                    f'reinforced_data/{session}_train_data_x.npy')
                additional_y = np.load(
                    f'reinforced_data/{session}_train_data_y.npy')
                for ndx, spect in enumerate(additional_x):
                    x.append(spect)
                    y.append(additional_y[ndx])
                    f_cnt += 1
        except:
            pass

    print(
        f'\n\ncreated {t_cnt} activation sets and {f_cnt} background sets\n\n')

    print('\n\nsaving x and y as .npy cache\n\n')
    if LSTM:
        np.save('x_data_lstm.npy', x)
        np.save('y_data_lstm.npy', y)
    else:
        np.save('x_data.npy', x)
        np.save('y_data.npy', y)
    return x, y


def get_spectrograms(waveform: list) -> list:
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
    x = generate_data()
