import json
from main import HeyDittoNet
import os
import random
import librosa
import numpy as np

ditto = HeyDittoNet(tflite=False)
reinforce_conf = json.load(open('data/reinforced_data/conf.json', 'r'))

# dir = 'data/audio_books/'
dir = 'data/yt_audio/'
files = os.listdir(dir)
random.shuffle(files)
for ndx, file in enumerate(files):
    if '.ini' in file:
        continue
    X = []
    Y = []
    print(f'Testing on {file}')
    y, s = librosa.load(
        f'{dir}/{file}', sr=16000, mono=True)
    seconds = y.size / 16000
    # get size of 1 second chunk by dividing total size by sample rate
    chunk_size = int(y.size/seconds)
    for i in range(0, int(y.size), chunk_size):  # iterate through each second
        chunk_second = y[i:i+chunk_size+1]
        chunk_second = ditto.normalize_audio(chunk_second)
        spect = ditto.get_spectrogram(chunk_second)
        pred = ditto.model(np.expand_dims(spect, 0))
        if pred[0][0] >= 0.90:
            print(f'Activated with {pred[0][0]}')
            X.append(spect)
            Y.append(0)

    if len(X) >= 1:
        count = reinforce_conf['sessions_total']
        print(f'Saving {len(X)} samples!')
        np.save(
            f'data/reinforced_data/{count}_train_data_x.npy', np.array(X).astype('float32'))
        np.save(
            f'data/reinforced_data/{count}_train_data_y.npy', np.array(Y))

        reinforce_conf['sessions_total'] += 1

        json.dump(reinforce_conf, open('data/reinforced_data/conf.json', 'w'))
