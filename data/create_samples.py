import google.cloud.texttospeech as tts
import os
import random
import numpy as np
from elevenlabslib import *
import json
import soundfile as sf
import soundfile
import io

def save_bytes_to_path(filepath:str, audioData:bytes) -> None:
    """
    This function saves the audio data to the specified location.
    soundfile is used for the conversion, so it supports any format it does.
    :param filepath: The path where the data will be saved to.
    :param audioData: The audio data.
    """
    fp = open(filepath, "wb")
    tempSoundFile = soundfile.SoundFile(io.BytesIO(audioData))
    sf.write(fp,tempSoundFile.read(), tempSoundFile.samplerate)

def gen_eleven_labs_sample(text, fname='heyditto'):
    key = ''
    with open('api_key.json','r') as f:
        key = json.load(f)['key']
    if key == '': return 'Needs API Key'

    user = ElevenLabsUser(key)
    voices = user.get_all_voices()
    random.shuffle(voices)
    for voice in voices:
    # voice = user.get_voices_by_name("Rachel")[0]  # This is a list because multiple voices can have the same name

    # voice.play_preview(playInBackground=False)

    # voice.generate_and_play_audio(text, playInBackground=False)
        for i in range(20):
            print(f'generating {voice.voiceID} iteration {i+1}')
            s = np.random.rand()
            sb = np.random.rand()
            data = voice.generate_audio_bytes(
                prompt=text,
                stability=s,
                similarity_boost=sb
            )
            save_bytes_to_path(f"elvenlabs_samples/session5/{voice.voiceID}-{fname}-{i}-{s}-{sb}.wav", data)

    # for historyItem in user.get_history_items():
    #     if historyItem.text == text:
    #         # The first items are the newest, so we can stop as soon as we find one.
    #         historyItem.delete()
    #         break

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
    # filename = f"{folder}/heyditto-{language_code}-{voice_name}-{voice_gender}-{pitch}.wav"
    filename = f"{folder}/background-{text.replace(' ','')}-{language_code}-{voice_name}-{voice_gender}-{pitch}.wav"
    with open(filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Generated speech saved to "{filename}"')

def generate_background_samples():
    voices = list_voices()
    # random.shuffle(voices)
    for ndx,voice in enumerate(voices):
        name = voice.name
        if 'US-Neural' in name:
            gender = voice.ssml_gender
            words = ["had it", "had to", "headed", "how dye do", "head to toe", "Hideo", "hadith", "had to", "hated", "hooded", "Hey Dad", "heeded"]
            for word in words:
                text_to_wav(name, gender, word, 'gtts_session13_background', pitch=0) # TODO: Run session13 background

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
                text_to_wav(name, gender, "Hey Ditto", 'gtts_session4', pitch=pitch)

# generate_heyditto_samples()
# generate_background_samples()
# gen_eleven_labs_sample(
#     text='Hey Ditto.', 
#     fname='heyditto-period'
# )
gen_eleven_labs_sample(
    text='Hey Ditto!', 
    fname='heyditto-exclamation'
)
gen_eleven_labs_sample(
    text='Hey Ditto...', 
    fname='heyditto-dotdotdot'
)
