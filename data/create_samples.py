import google.cloud.texttospeech as tts
import os
import random
import numpy as np
from elevenlabslib import *
import json
import soundfile as sf
import soundfile
import io


def save_bytes_to_path(filepath: str, audioData: bytes) -> None:
    """
    This function saves the audio data to the specified location.
    soundfile is used for the conversion, so it supports any format it does.
    :param filepath: The path where the data will be saved to.
    :param audioData: The audio data.
    """
    fp = open(filepath, "wb")
    tempSoundFile = soundfile.SoundFile(io.BytesIO(audioData))
    sf.write(fp, tempSoundFile.read(), tempSoundFile.samplerate)


def gen_eleven_labs_sample(text, fname='heyditto'):
    key = ''
    with open('api_key.json', 'r') as f:
        key = json.load(f)['key']
    if key == '':
        return 'Needs API Key'

    user = ElevenLabsUser(key)
    voices = user.get_all_voices()
    random.shuffle(voices)
    for voice in voices:
        if 'new' in voice.initialName:
            for i in range(5):
                print(
                    f'generating {voice.initialName}-{voice.voiceID} iteration {i+1}')
                s = np.random.rand()
                sb = np.random.rand()
                data = voice.generate_audio_bytes(
                    prompt=text,
                    stability=s,
                    similarity_boost=sb
                )
                save_bytes_to_path(
                    f"elvenlabs_samples/session8/{voice.voiceID}-{fname}-{i}-{s}-{sb}.wav", data)


def list_voices(language_code=None):
    client = tts.TextToSpeechClient()
    response = client.list_voices(language_code=language_code)
    voices = sorted(response.voices, key=lambda voice: voice.name)

    print(f" Voices: {len(voices)} ".center(60, "-"))
    return voices


def text_to_wav(voice_name: str, voice_gender: str, text: str, folder: str, pitch: float = 0):
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code, name=voice_name,
        ssml_gender=voice_gender
    )
    audio_config = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.LINEAR16, pitch=pitch)

    client = tts.TextToSpeechClient()
    response = client.synthesize_speech(
        input=text_input, voice=voice_params, audio_config=audio_config
    )
    if not os.path.exists(folder):
        os.mkdir(folder)
    # filename = f"{folder}/heyditto-{language_code}-{voice_name}-{voice_gender}-{pitch}.wav"
    filename = f"{folder}/background-{text.replace(' ','')}-{language_code}-{voice_name}-{voice_gender}-{pitch}.wav"
    with open(filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Generated speech saved to "{filename}"')


def generate_background_samples():
    voices = list_voices()
    # random.shuffle(voices)
    for ndx, voice in enumerate(voices):
        name = voice.name
        if 'US-Neural' in name:
            gender = voice.ssml_gender
            words = ["had it", "had to", "headed", "how dye do", "head to toe",
                     "Hideo", "hadith", "had to", "hated", "hooded", "Hey Dad", "heeded"]
            for word in words:
                # TODO: Run session13 background
                text_to_wav(name, gender, word,
                            'gtts_session13_background', pitch=0)


def generate_heyditto_samples():
    voices = list_voices()
    for voice in voices:
        name = voice.name
        if 'US-Neural' in name:
            gender = voice.ssml_gender
            pitches = list(range(-20, 21))
            random.shuffle(pitches)
            for ndx, pitch in enumerate(pitches):
                if ndx+1 == 20:
                    break
                print('generating voice with pitch %d' % pitch)
                text_to_wav(name, gender, "Hey Ditto",
                            'gtts_session4', pitch=pitch)


# generate_heyditto_samples()
# generate_background_samples()
# sentences = ["The quick brown fox jumps over the lazy dog.",
#              "The cat sat on the mat.",
#              "I love you.",
#              "I am a large language model.",
#              "The sky is blue.",
#              "The grass is green.",
#              "The sun is shining.",
#              "The birds are singing.",
#              "It is a beautiful day.",
#              "I am happy."]
sentences = ['Hey Ditto?',
             'HEY DITTO!!!',
             'hey ditto?',
             'Hey! Ditto!']
for sentence in sentences:
    gen_eleven_labs_sample(
        text=sentence,
        fname='heyditto'
    )
# gen_eleven_labs_sample(
#     text='Hey Ditto!',
#     fname='heyditto-exclamation'
# )
# gen_eleven_labs_sample(
#     text='Hey Ditto...',
#     fname='heyditto-dotdotdot'
# )

# words = ["had it", "had to", "headed", "how dye do", "head to toe", "Hideo", "hadith", "had to", "hated", "hooded", "Hey Dad", "heeded"]
# words = ['hey Data', 'data', 'need data', 'dada', 'hey dude']
# for word in words:
#     gen_eleven_labs_sample(
#         text=word,
#         fname='background'
#     )


# gen_eleven_labs_sample(
#     text="Alright, thank you for granting me the ability. Let's see...Dude was a young man who had just moved to the city in search of adventure. He was determined to make something out of his life and decided that he would take any job or opportunity that came his way. After several months, he eventually found himself working as a waiter at a small cafe on the outskirts of town. Despite not having much money, Dude made sure to enjoy every moment and savor all the experiences he encountered along the way.",
#     fname='background'
# )
