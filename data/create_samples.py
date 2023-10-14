import google.cloud.texttospeech as tts
import os
import random
import numpy as np
from elevenlabslib import *
import json
import soundfile as sf
import soundfile
import io
import openai

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


def gen_eleven_labs_sample(fname:str, session_num:int, sentences:list):

    key = ''
    with open('api_key.json', 'r') as f:
        key = json.load(f)['key']
    if key == '':
        return 'Needs API Key'
    
    if 'heyditto' in fname:
        if not os.path.exists(f'elvenlabs_samples/session{session_num}'): 
            os.mkdir(f'elvenlabs_samples/session{session_num}')
    else:
        if not os.path.exists(f'elvenlabs_samples/session{session_num}-background'): 
            os.mkdir(f'elvenlabs_samples/session{session_num}-background')

    user = ElevenLabsUser(key)
    voices = user.get_all_voices()
    random.shuffle(voices)
    for voice in voices:
        # if 'new' in voice.initialName:
        for i in range(5):
            print(
                f'generating {voice.initialName}-{voice.voiceID} iteration {i+1}')
            s = np.random.rand()
            sb = np.random.rand()

            random.shuffle(sentences)
            text = sentences[0]
            
            data = voice.generate_audio_v2(
                prompt=text,
                generationOptions=GenerationOptions(stability=s, similarity_boost=sb)
            )
            if 'background' in fname:
                save_bytes_to_path(
                    f"elvenlabs_samples/session{session_num}-background/{voice.voiceID}-{fname}-{i}-{s}-{sb}.wav", data[0])
            else:
                save_bytes_to_path(
                    f"elvenlabs_samples/session{session_num}/{voice.voiceID}-{fname}-{i}-{s}-{sb}.wav", data[0])


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
    sentences = []
    for i in range(20):
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt="Prompt: generate a random sentence that uses uncommon words and phrases.\n Random sentence:",
            temperature=0.7,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6,
            stop=["Prompt:"]
        )
        sentences.append(response['choices'][0]['text'].strip())
    
    gen_eleven_labs_sample(
        fname='background',
        session_num=9,
        sentences=sentences
    )


def generate_heyditto_samples():
    sentences = [
        'Hey Ditto',
        'HEY DITTO!!!',
        'Hey? Ditto?',
        'Hey Ditto...',
        'HEY! DITTO!',
        'Hey Ditto?'
    ]
    gen_eleven_labs_sample(
        fname='heyditto',
        session_num=12,
        sentences=sentences
    )


# generate_background_samples()
generate_heyditto_samples()
