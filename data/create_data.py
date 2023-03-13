import json
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

def white_noise(sample):
    # Add white noise to sample with scaling 
    wn = np.random.randn(len(sample))
    data_wn = sample + 0.005*wn
    return data_wn

def stretch(sample, rate=1):
    max_len = 16000
    data = librosa.effects.time_stretch(sample, rate=rate)
    if len(data)>max_len: # if gets too big, snip the end
        data = data[:max_len]
    elif len(data)<16000: # if gets too small, pad with zeros
        data = np.pad(data, (0, max(0, max_len - len(data))))
    return data

def lower_volume(sample, db=10):
    raw_audio = AudioSegment.from_wav(sample)
    audio = effects.normalize(raw_audio)
    audio = audio - db
    samples = audio.get_array_of_samples()
    audio._spawn(samples) # write to samples
    return np.array(samples).astype(np.float32, order='C') / 32768.0

def normalize_audio(sample):
    audio = AudioSegment.from_wav(sample)
    audio_norm = effects.normalize(audio)
    samples = audio_norm.get_array_of_samples()
    max_len = 16000
    if len(samples)>max_len: # if gets too big, snip the end
        # print(sample)
        # print('sample too long (snipping)')
        # print(len(samples))
        samples = samples[:max_len]
    elif len(samples)<16000: # if gets too small, pad with zeros
        # print('sample too short (padding)')
        # print(len(samples))
        samples = np.pad(samples, (0, max(0, max_len - len(samples))))
    audio_norm._spawn(samples) # write to samples
    return np.array(samples).astype(np.float32, order='C') / 32768.0

def combine_with(activation, background):
    a_audio = AudioSegment.from_wav(activation)
    a_audio_norm = effects.normalize(a_audio)
    b_audio = AudioSegment.from_wav(background)
    b_audio_norm = effects.normalize(b_audio) - 8
    audio = a_audio_norm.overlay(b_audio_norm)
    samples = audio.get_array_of_samples() # write to samples
    return np.array(samples).astype(np.float32, order='C') / 32768.0


def generate_data() -> tuple:
    '''
    Generates x and y training data for "Hey Ditto" speech recognition.
    '''
    files = os.listdir('raw_data')
    activation_set = []
    background_set = []
    for file_name in files:
        if 'heyditto' in file_name.replace(' ', '').lower(): activation_set.append('raw_data/'+file_name)
        if 'background' in file_name: background_set.append('raw_data/'+file_name)

    print(f'\n\nloaded {len(activation_set)} activation sets and {len(background_set)} background sets\n\n')
    print('processing...\n')
    x = []
    y = []

    t_cnt, f_cnt = 0,0
    for activation_phrase, background_noise in zip(activation_set, background_set): #NOTE: zip through background sets too and overlay normalized audio with lowered background!!
        # audio = librosa.load(activation_phrase, sr=16000)
        audio = [normalize_audio(activation_phrase)]
        

        # data augmentation
        # audio_normalized = normalize_audio(activation_phrase)
        audio_quiet = lower_volume(activation_phrase, db=15)
        # audio_really_quiet = lower_volume(activation_phrase, db=20)
        # audio_very_quiet = lower_volume(activation_phrase, db=30)
        audio_noise = white_noise(audio[0])
        audio_stretch_low = stretch(audio[0], rate=0.9)
        audio_stretch_high = stretch(audio[0], rate=1.2)
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

        spect = get_spectrogram(audio[0])
        x.append(spect)
        y.append(1) # activate 
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

        t_cnt+=6 # true class count

    for background_noise in background_set:
        audio = librosa.load(background_noise, sr=16000)
        spect = get_spectrogram(audio[0])
        x.append(spect)
        y.append(0) # do nothing
        f_cnt+=1 # false class count

    try:
        print('loading reinforcement sessions...\n')
        with open('reinforced_data/conf.json', 'r') as f:
            conf = json.load(f)
            sessions_total = conf['sessions_total']
        for session in range(sessions_total):
            print(f'\nprocessing reinforcement session {session}')
            additional_x = np.load(f'reinforced_data/{session}_train_data_x.npy')
            additional_y = np.load(f'reinforced_data/{session}_train_data_y.npy')
            for ndx,spect in enumerate(additional_x):
                x.append(spect)
                y.append(additional_y[ndx])
                f_cnt+=1
    except:
        pass

    print(f'\n\ncreated {t_cnt} activation sets and {f_cnt} background sets\n\n')

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
