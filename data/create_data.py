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
from python_speech_features import logfbank


# supress tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

RATE = 16000

TIME_SERIES = False
WINDOW = int(RATE/4)
STRIDE = int((RATE - WINDOW)/4)

ACTIVATION_AUGMENTATION_PERCENT = 0.6
BACKGROUND_AUGMENTATION_PERCENT = 0.5
REINFORCEMENT_AUGMENTATION_PERCENT = 0.5


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


def downsample_audio(sample):
    random_downscale_sr = np.random.randint(4000, 8000)
    downsampled = librosa.resample(
        y=sample,
        orig_sr=16000,
        target_sr=random_downscale_sr)
    downsample_audio_upscaled = librosa.resample(
        y=downsampled,
        orig_sr=random_downscale_sr,
        target_sr=16000)
    return downsample_audio_upscaled


def lower_volume(sample, db=10):
    raw_audio = AudioSegment.from_wav(sample)
    audio = effects.normalize(raw_audio)
    audio = audio - db
    samples = audio.get_array_of_samples()
    audio._spawn(samples)  # write to samples
    return np.array(samples).astype(np.float32, order='C') / 32768.0


def rand_pitch(sample):
    # print(sample)
    sound = AudioSegment.from_wav(sample)
    rand = random.random()
    octaves = random.uniform(-0.4,
                             -0.2) if rand >= 0.5 else random.uniform(0.2, 0.4)
    new_sample_rate = int(sound.frame_rate * (2.0 ** octaves))
    hipitch_sound = sound._spawn(sound.raw_data, overrides={
                                 'frame_rate': new_sample_rate})
    audio = hipitch_sound.set_frame_rate(16000)
    audio = effects.normalize(audio)
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
    # amount to decrease background sample by
    decrease_amount = np.random.uniform(2 if 'music' in background else 4, 6)
    a_audio = AudioSegment.from_wav(activation)
    a_audio_norm = effects.normalize(a_audio)
    b_audio = AudioSegment.from_wav(background)
    b_audio_norm = effects.normalize(b_audio) - decrease_amount
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
    x = []
    y = []

    t_cnt, f_cnt = 0, 0
    random.shuffle(activation_set)
    random.shuffle(background_set)
    # count = 0

    print('processing activation sets...\n')

    activation_ndx = 0
    for activation_phrase, background_noise in zip(activation_set, background_set):


        # print percent if divisible by 100
        if activation_ndx % 100 == 0:
            # print progress in terms of percent
            print(f'activation set progress: {round(100*activation_ndx/len(activation_set), 2)}%')

        if 'Neural2' in activation_phrase:
            continue  # contains a lot of dirty samples...
        audio = [normalize_audio(activation_phrase)]

        # probability for data augmentation to keep mostly clean samples
        if random.random() > ACTIVATION_AUGMENTATION_PERCENT: # only augment data ACTIVATION_AUGMENTATION_PERCENT of the time 
            if TIME_SERIES:
                x.append(get_spectrograms(audio[0]))
                y.append(1)  # activate
            else:
                spect = get_spectrogram(audio[0])
                x.append(spect)
                y.append(1)  # activate
            t_cnt+=1
            activation_ndx += 1
            continue

        # data augmentation
        audio_quiet = lower_volume(
            activation_phrase, db=random.uniform(3.0, 9.0))
        audio_rand_pitch = rand_pitch(activation_phrase)
        # audio_really_quiet = lower_volume(activation_phrase, db=20)
        # audio_very_quiet = lower_volume(activation_phrase, db=30)
        audio_noise1 = white_noise(audio[0], amount=random.uniform(0.002, 0.2))
        audio_noise2 = white_noise(audio[0], amount=random.uniform(0.002, 0.1))
        # audio_stretch_low = stretch(audio[0], rate=random.uniform(0.88, 0.99))
        # audio_stretch_high = stretch(audio[0], rate=random.uniform(1.1, 1.3))
        audio_downsampled = downsample_audio(audio[0])
        combined_audio1 = combine_with(activation_phrase, background_noise)
        combined_audio2 = combine_with(activation_set[np.random.randint(len(activation_set))], background_set[np.random.randint(len(background_set))])

        # sounddevice.play(audio[0], samplerate=16000)
        # time.sleep(1)
        # sounddevice.play(audio_downsampled, samplerate=16000)
        # time.sleep(1)
        # sounddevice.play(audio_noise1, samplerate=16000)
        # time.sleep(1)
        # sounddevice.play(audio_noise2, samplerate=16000)
        # time.sleep(1)
        # sounddevice.play(audio_rand_pitch, samplerate=16000)
        # time.sleep(1)
        # sounddevice.play(audio_stretch_high, samplerate=16000)
        # time.sleep(1)
        # sounddevice.play(audio_quiet, samplerate=16000)
        # time.sleep(1)
        # sounddevice.play(audio_really_quiet, samplerate=16000)
        # time.sleep(2)
        # if 'music' in background_noise:
        # print(activation_phrase)
        #     sounddevice.play(combined_audio1, samplerate=16000)
        #     time.sleep(1)
        # sounddevice.play(combined_audio1, samplerate=16000)
        # time.sleep(1)
        # sounddevice.play(combined_audio2, samplerate=16000)
        # time.sleep(1)

        # exit()
        if TIME_SERIES:
            x.append(get_spectrograms(audio[0]))
            y.append(1)  # activate
            x.append(get_spectrograms(audio_quiet))
            y.append(1)
            x.append(get_spectrogram(audio_rand_pitch))
            y.append(1)
            x.append(get_spectrograms(audio_noise1))
            y.append(1)
            x.append(get_spectrograms(audio_noise2))
            y.append(1)
            x.append(get_spectrograms(audio_downsampled))
            y.append(1)
            x.append(get_spectrograms(combined_audio1))
            y.append(1)
            x.append(get_spectrograms(combined_audio2))
            y.append(1)

        else:
            spect = get_spectrogram(audio[0])
            x.append(spect)
            y.append(1)  # activate
            x.append(get_spectrogram(audio_quiet))
            y.append(1)
            x.append(get_spectrogram(audio_rand_pitch))
            y.append(1)
            x.append(get_spectrogram(audio_noise1))
            y.append(1)
            x.append(get_spectrogram(audio_noise2))
            y.append(1)
            x.append(get_spectrogram(audio_downsampled))
            y.append(1)
            x.append(get_spectrogram(combined_audio1))
            y.append(1)
            x.append(get_spectrogram(combined_audio2))
            y.append(1)

        t_cnt += 8  # true class count
        activation_ndx += 1

    print('processing background sets...\n')

    for ndx, background_noise in enumerate(background_set):

        # print percent if divisible by 100
        if ndx % 100 == 0:
            # print progress in terms of percent
            print(f'background set progress: {round(100*ndx/len(background_set), 2)}%')
            

        audio = [normalize_audio(background_noise)]
        if TIME_SERIES:
            spect = get_spectrograms(audio[0])
        else:
            spect = get_spectrogram(audio[0])

        x.append(spect)
        y.append(0)
        aug_prob = random.random()
        if aug_prob > BACKGROUND_AUGMENTATION_PERCENT:  # apply augmentations to BACKGROUND_AUGMENTATION_PERCENT of background samples
            audio_noise = white_noise(
                audio[0], amount=random.uniform(0.002, 0.3))

            audio_quiet = lower_volume(background_noise, db=np.random.randint(
                10, 20)+np.random.rand())  # Decreases volume

            combined_audio1 = combine_with(
                background_set[ndx-1], background_noise)
            if TIME_SERIES:
                x.append(get_spectrograms(audio_noise))
                y.append(0)
                x.append(get_spectrograms(audio_quiet))
                y.append(0)
                x.append(get_spectrograms(combined_audio1))
                y.append(0)
            else:
                x.append(get_spectrogram(audio_noise))
                y.append(0)
                x.append(get_spectrogram(audio_quiet))
                y.append(0)
                x.append(get_spectrogram(combined_audio1))
                y.append(0)

            f_cnt += 4  # false class count

            # sounddevice.play(audio[0], samplerate=16000)
            # time.sleep(1)

            # sounddevice.play(audio_noise, samplerate=16000)
            # time.sleep(1)

            # sounddevice.play(audio_quiet, samplerate=16000)
            # time.sleep(1)

            # sounddevice.play(combined_audio, samplerate=16000)
            # time.sleep(1)
        else:
            f_cnt += 1  # false class count

    # if not TIME_SERIES:
    print('loading reinforcement sessions...\n')

    reinforce_dir = 'reinforce_background/'
    with open(f'{reinforce_dir}conf.json', 'r') as f:
        conf = json.load(f)
        sessions_total = conf['sessions_total']
    for session in range(sessions_total):
        print(f'\nprocessing reinforcement session {session}')
        additional_x = np.load(
            f'{reinforce_dir}{session}_train_data_x.npy')
        additional_y = np.load(
            f'{reinforce_dir}{session}_train_data_y.npy')
        for ndx, sample in enumerate(additional_x):
            aug_prob = random.random()
            if aug_prob > REINFORCEMENT_AUGMENTATION_PERCENT:  # apply augmentations to REINFORCEMENT_AUGMENTATION_PERCENT of background samples
                if TIME_SERIES:
                    spect = get_spectrograms(sample)
                    spect_noise = get_spectrograms(
                        white_noise(sample, random.uniform(0.002, 0.3)))
                    spect_downsampled = get_spectrograms(downsample_audio(sample))
                else:
                    spect = get_spectrogram(sample)
                    spect_noise = get_spectrogram(
                        white_noise(sample, random.uniform(0.002, 0.3)))
                    spect_downsampled = get_spectrogram(downsample_audio(sample))

                x.append(spect)
                y.append(additional_y[ndx])  # always zero
                x.append(spect_noise)
                y.append(additional_y[ndx])  # always zero
                x.append(spect_downsampled)
                y.append(additional_y[ndx])  # always zero
                f_cnt += 3
            else:
                if TIME_SERIES:
                    spect = get_spectrograms(sample)
                else:
                    spect = get_spectrogram(sample)
                x.append(spect)
                y.append(additional_y[ndx])  # alwaus zero
                f_cnt += 1


    print(
        f'\n\ncreated {t_cnt} activation sets and {f_cnt} background sets\n\n')

    print('\n\nsaving x and y as .npy cache\n\n')

    if TIME_SERIES:
        print('Xshape', (len(x), len(x[0]), len(
            x[0][0]), len(x[0][0][0]), len(x[0][0][0][0])))
    else:
        print('Xshape', (len(x), len(x[0]), len(
            x[0][0]), len(x[0][0][0])))
    print('Yshape', len(y))
    print('\n')

    if TIME_SERIES:
        np.save('x_data_ts.npy', x)
        np.save('y_data_ts.npy', y)
    else:
        np.save('x_data.npy', x)
        np.save('y_data.npy', y)

    print('Done.')
    return x, y


def get_spectrograms(waveform: list, stride=STRIDE) -> list:
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
    fbank_feat = logfbank(equal_length, 16000, nfilt=26)
    spectrogram = fbank_feat[..., tf.newaxis]
    return spectrogram


if __name__ == "__main__":
    x = generate_data()
