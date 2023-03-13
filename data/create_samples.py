import google.cloud.texttospeech as tts
import os
import random

def list_voices(language_code=None):
    client = tts.TextToSpeechClient()
    response = client.list_voices(language_code=language_code)
    voices = sorted(response.voices, key=lambda voice: voice.name)

    print(f" Voices: {len(voices)} ".center(60, "-"))
    return voices


def text_to_wav(voice_name: str, voice_gender: str, text: str, folder: str):
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code, name=voice_name,
        ssml_gender=voice_gender
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

    client = tts.TextToSpeechClient()
    response = client.synthesize_speech(
        input=text_input, voice=voice_params, audio_config=audio_config
    )
    if not os.path.exists(folder): os.mkdir(folder)
    filename = f"{folder}/background-{language_code}-{voice_name}-{voice_gender}.wav"
    with open(filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Generated speech saved to "{filename}"')

voices = list_voices()
random.shuffle(voices)
for ndx,voice in enumerate(voices):
    if ndx+1 == 40: break
    name = voice.name
    gender = voice.ssml_gender
    text_to_wav(name, gender, "Hey man", 'gtts_session13_background') # TODO: Ran session13 background
