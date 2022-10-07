import os

# files = os.listdir('41000_data')

# for file in files:
#     os.system(f'ffmpeg -i "41000_data/{file}" -ar 16000 "raw_data/{file}"')

files = os.listdir('48000_data')

for file in files:
    if 'heyditto3' in file:
        os.system(f'ffmpeg -i "48000_data/{file}" -ar 16000 "raw_data/{file}"')

