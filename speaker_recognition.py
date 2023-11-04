from hey_ditto_net_embeddings import HeyDittoNetEmbeddings
from pydub import AudioSegment
from pydub import effects
import numpy as np
from sounddevice import InputStream, wait

class SpeakerRecognition:

    def __init__(self):
        embeddings = HeyDittoNetEmbeddings()
        self.embeddings_model = embeddings.embeddings_model
        self.hey_ditto_net = embeddings.hey_ditto_net

    def get_audio_array_from_audio_path(self, audio_path):
        # make sure and convert to 16 bit wav
        raw_audio = AudioSegment.from_file(audio_path, format='mp3')
        audio = effects.normalize(raw_audio)
        samples = audio.get_array_of_samples()
        audio._spawn(samples)  # write to samples
        return np.array(samples).astype(np.float32, order='C') / 32768.0
    
    def get_embeddings(self, audio_path=None, audio=None):
        if audio_path:
            audio = self.get_audio_array_from_audio_path(audio_path)
        spect = self.hey_ditto_net.get_spectrogram(audio)
        embeddings = self.embeddings_model.predict(np.expand_dims(spect, axis=0))
        return embeddings.flatten()
    
    def get_similarity(self, audio1, audio2):
        embeddings_1 = self.get_embeddings(audio=audio1)
        embeddings_2 = self.get_embeddings(audio=audio2)
        # calculate distance between embeddings
        distance = np.linalg.norm(embeddings_1 - embeddings_2)
        # calculate similarity
        similarity = 1.0 / (1.0 + distance)
        return similarity
    
    def get_similarity_from_audio_paths(self, audio_path1, audio_path2):
        audio1 = self.get_audio_array_from_audio_path(audio_path1)
        audio2 = self.get_audio_array_from_audio_path(audio_path2)
        return self.get_similarity(audio1, audio2)
    
    def callback(self, indata, frames, time, status):
        for val in indata:
            self.buffer.append(val)
        if len(self.buffer) >= 16000:
            #ensure 16000 samples
            self.buffer = self.buffer[-16000:]
            # pad to audio 16000 samples if less than 16000 samples
            similarity = self.get_similarity(self.audio, np.array(self.buffer))
            print(similarity)

    def open_audio_stream_get_similarity_over_audio_path(self, audio_path):
        self.audio = self.get_audio_array_from_audio_path(audio_path)
        # open sounddevice audio input stream and every 0.2 seconds, get similarity 
        # from the last 1 second of audio vs the input audio path (audio_path)
        self.buffer = []        
        with InputStream(callback=self.callback, blocksize=int(16000/4), samplerate=16000):
            input()

        
    

if __name__ == "__main__":
    
    speaker_recognition = SpeakerRecognition()

    omar_heyditto = 'data/sample_audio/omar_heyditto.wav'
    speaker_recognition.open_audio_stream_get_similarity_over_audio_path(omar_heyditto)

    # speaker1_heyditto = 'data/sample_audio/dan_heyditto.mp3'
    # speaker1_talking = 'data/sample_audio/dan_talking.mp3'
    # speaker2_heyditto = 'data/sample_audio/mark_heyditto.mp3'
    # speaker2_talking = 'data/sample_audio/mark_talking.mp3'

    # print('speaker1_heyditto vs speaker1_talking')
    # print(speaker_recognition.get_similarity_from_audio_paths(speaker1_heyditto, speaker1_talking))

    # print('speaker2_heyditto vs speaker2_talking')
    # print(speaker_recognition.get_similarity_from_audio_paths(speaker1_heyditto, speaker2_heyditto))

    # print('speaker1_heyditto vs speaker2_talking')
    # print(speaker_recognition.get_similarity_from_audio_paths(speaker1_heyditto, speaker2_talking))

    # print('speaker2_heyditto vs speaker1_talking')
    # print(speaker_recognition.get_similarity_from_audio_paths(speaker2_heyditto, speaker1_talking))

    