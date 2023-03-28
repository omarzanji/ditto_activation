import google.cloud.texttospeech as tts
import os
import random

import numpy as np

def list_voices(language_code=None):
    client = tts.TextToSpeechClient()
    response = client.list_voices(language_code=language_code)
    voices = sorted(response.voices, key=lambda voice: voice.name)

    print(f" Voices: {len(voices)} ".center(60, "-"))
    return voices


def text_to_wav(voice_name: str, voice_gender: str, text: str, folder: str, pitch: float=0):
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code, name=voice_name,
        ssml_gender=voice_gender
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16, pitch=pitch)

    client = tts.TextToSpeechClient()
    response = client.synthesize_speech(
        input=text_input, voice=voice_params, audio_config=audio_config
    )
    if not os.path.exists(folder): os.mkdir(folder)
    filename = f"{folder}/heyditto-{language_code}-{voice_name}-{voice_gender}-{pitch}.wav"
    # filename = f"{folder}/background-{language_code}-{voice_name}-{voice_gender}-{pitch}.wav"
    with open(filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Generated speech saved to "{filename}"')

def generate_background_samples():
    voices = list_voices()
    random.shuffle(voices)
    for ndx,voice in enumerate(voices):
        if ndx+1 == 40: break
        name = voice.name
        gender = voice.ssml_gender
        text_to_wav(name, gender, "Hey man", 'gtts_session13_background', pitch=0) # TODO: Ran session13 background

def generate_heyditto_samples():
    voices = list_voices()
    for voice in voices:
        name = voice.name
        if 'US-Neural' in name:
            gender = voice.ssml_gender
            pitches = list(range(-20, 21))
            random.shuffle(pitches)
            for ndx,pitch in enumerate(pitches):
                if ndx+1 == 20: break
                print('generating voice with pitch %d' % pitch)
                text_to_wav(name, gender, "Hey Ditto", 'gtts_session3', pitch=pitch)

generate_heyditto_samples()