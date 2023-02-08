import os
import librosa
import soundfile as sf
import time

# files = os.listdir('41000_data')
# for file in files:
#     # if 'heyditto' in file:
#     os.system(f'ffmpeg -y -i "41000_data/{file}" -ar 16000 "raw_data/{file}"')

# files = os.listdir('48000_data')
# for file in files:
#     # if 'heyditto3' in file:
#     os.system(f'ffmpeg -y -i "48000_data/{file}" -ar 16000 "raw_data/{file}"')

# files = os.listdir('synthetic_data/')
# for file in files:
#     # if 'heyditto3' in file:
#     os.system(f'ffmpeg -y -i "synthetic_data/{file}" -ar 16000 "raw_data/{file}.wav"')

# files = os.listdir('mp3_data/')
# for ndx,file in enumerate(files):
#     stamp = int(time.time())+ndx+1
#     os.system(f'ffmpeg -y -i "mp3_data/{file}" -ar 16000 "raw_data/heyditto-{stamp}-{file}.wav"')

# files = os.listdir('common_voice_dataset/data/')
# for ndx,file in enumerate(files):
#     stamp = int(time.time())+ndx+1
#     os.system(f'ffmpeg -y -i "common_voice_dataset/data/{file}" -ar 16000 "common_voice_dataset/wav_data/{file}.wav"')


# files = os.listdir('Hospital noise original/')
# size = len(files)
# print('converting hospital ambient noise to background dataset')
# for ndx,file in enumerate(files):
#     y, s = librosa.load(f'Hospital noise original/{file}',sr=16000, mono=True)
#     chunk_size = int(y.size/5) # each sample is 5 seconds, so to get 1 second chunks, divide by 5
#     for i in range(0, int(y.size), chunk_size): # iterate through each second
#         sf.write(f'raw_data/background_{file}_{i}.wav', y[i:i+chunk_size], 16000)

# y, s = librosa.load(f'background_data/background-horns.wav',sr=16000, mono=True)
# chunk_size = int(y.size/(y.size/16000)) # each sample is 1 second, so to get 1 second chunks, divide by RATE
# for i in range(0, int(y.size), chunk_size): # iterate through each second
#     sf.write(f'raw_data/background_horns_{i}.wav', y[i:i+chunk_size], 16000)


# files = os.listdir('common_voice_dataset/wav_data/')
# size = len(files)
# print('converting common voice dataset to background dataset')
# for ndx,file in enumerate(files):
#     y, s = librosa.load(f'common_voice_dataset/wav_data/{file}',sr=16000, mono=True)
#     seconds = y.size / 16000
#     chunk_size = int(y.size/seconds) # each sample is 5 seconds, so to get 1 second chunks, divide by 5
#     for i in range(0, int(y.size), chunk_size): # iterate through each second
#         sf.write(f'raw_data/background_{file}_{i}.wav', y[i:i+chunk_size+1], 16000)
#     if ndx==500: exit()