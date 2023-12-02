# from hey_ditto_net_embeddings import HeyDittoNetEmbeddings
from pydub import AudioSegment
from pydub import effects
import pydub
import numpy as np
from sounddevice import InputStream, wait
import io

# class SpeakerRecognition:

#     def __init__(self):
#         embeddings = HeyDittoNetEmbeddings()
#         self.embeddings_model = embeddings.embeddings_model
#         self.hey_ditto_net = embeddings.hey_ditto_net

#     def get_audio_array_from_audio_path(self, audio_path):
#         # make sure and convert to 16 bit wav
#         raw_audio = AudioSegment.from_file(audio_path, format='mp3')
#         audio = effects.normalize(raw_audio)
#         samples = audio.get_array_of_samples()
#         audio._spawn(samples)  # write to samples
#         return np.array(samples).astype(np.float32, order='C') / 32768.0
    
#     def get_embeddings(self, audio_path=None, audio=None):
#         if audio_path:
#             audio = self.get_audio_array_from_audio_path(audio_path)
#         spect = self.hey_ditto_net.get_spectrogram(audio)
#         embeddings = self.embeddings_model.predict(np.expand_dims(spect, axis=0))
#         return embeddings.flatten()
    
#     def get_similarity(self, audio1, audio2):
#         embeddings_1 = self.get_embeddings(audio=audio1)
#         embeddings_2 = self.get_embeddings(audio=audio2)
#         # calculate distance between embeddings
#         distance = np.linalg.norm(embeddings_1 - embeddings_2)
#         # calculate similarity
#         similarity = 1.0 / (1.0 + distance)
#         return similarity
    
#     def get_similarity_from_audio_paths(self, audio_path1, audio_path2):
#         audio1 = self.get_audio_array_from_audio_path(audio_path1)
#         audio2 = self.get_audio_array_from_audio_path(audio_path2)
#         return self.get_similarity(audio1, audio2)
    
#     def callback(self, indata, frames, time, status):
#         for val in indata:
#             self.buffer.append(val)
#         if len(self.buffer) >= 16000:
#             #ensure 16000 samples
#             self.buffer = self.buffer[-16000:]
#             # pad to audio 16000 samples if less than 16000 samples
#             similarity = self.get_similarity(self.audio, np.array(self.buffer))
#             print(similarity)

#     def open_audio_stream_get_similarity_over_audio_path(self, audio_path):
#         self.audio = self.get_audio_array_from_audio_path(audio_path)
#         # open sounddevice audio input stream and every 0.2 seconds, get similarity 
#         # from the last 1 second of audio vs the input audio path (audio_path)
#         self.buffer = []        
#         with InputStream(callback=self.callback, blocksize=int(16000/4), samplerate=16000):
#             input()

from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch

class SpeakerRecognition:

    def __init__(self) -> None:
        self.load_models()

    def load_models(self):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            'microsoft/wavlm-base-sv',
            sampling_rate=16000
        )
        self.model = WavLMForXVector.from_pretrained(
            'microsoft/wavlm-base-sv',
        )

    def get_audio_array_from_audio_path(self, audio_path):
        # make sure and convert to 16 bit wav and specify sample rate
        raw_audio = AudioSegment.from_wav(audio_path)
        # convert to mono 
        raw_audio = raw_audio.set_channels(1)
        # convert to 16000 sample rate
        raw_audio = raw_audio.set_frame_rate(16000)
        audio = effects.normalize(raw_audio)
        samples = audio.get_array_of_samples()
        audio._spawn(samples)  # write to samples
        audio_array = np.array(samples).astype(np.float32, order='C') / 32768.0
        # make sure 16000 samples
        if len(audio_array) < 16000:
            audio_array = np.pad(audio_array, (0, 16000-len(audio_array)), constant_values=0)
        elif len(audio_array) > 16000:
            audio_array = audio_array[:16000]
        return audio_array


    def get_embeddings(self, audio_path=None, audio=None):
        # audio files are decoded on the fly
        # inputs = feature_extractor(dataset[:2]["audio"]["array"], return_tensors="pt")
        if audio_path:
            audio = self.get_audio_array_from_audio_path(audio_path)
        else:
            audio = self.get_audio_segment_from_numpy_array(np.array(audio), framerate=16000)
            audio = effects.normalize(audio)
            samples = audio.get_array_of_samples()
            audio._spawn(samples)  # write to samples
            audio_array = np.array(samples).astype(np.float32, order='C') / 32768.0
            audio = audio_array
        inputs = self.feature_extractor(audio, return_tensors="pt", sampling_rate=16000)
        embeddings = self.model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
        return embeddings

    def get_audio_segment_from_numpy_array(self, nparr, framerate):
        """
        Returns an AudioSegment created from the given numpy array.

        The numpy array must have shape = (num_samples, num_channels).

        :param nparr: The numpy array to create an AudioSegment from.
        :param framerate: The sample rate (Hz) of the segment to generate.
        :returns: An AudioSegment created from the given array.
        """
        # Check args
        if nparr.dtype.itemsize not in (1, 2, 4):
            raise ValueError("Numpy Array must contain 8, 16, or 32 bit values.")

        # Determine nchannels
        if len(nparr.shape) == 1:
            nchannels = 1
        elif len(nparr.shape) == 2:
            nchannels = nparr.shape[1]
        else:
            raise ValueError("Numpy Array must be one or two dimensional. Shape must be: (num_samples, num_channels), but is {}.".format(nparr.shape))

        # Fix shape if single dimensional
        nparr = np.reshape(nparr, (-1, nchannels))

        # Create an array of mono audio segments
        monos = []
        for i in range(nchannels):
            m = nparr[:, i]
            dubseg = pydub.AudioSegment(m.tobytes(), frame_rate=framerate, sample_width=nparr.dtype.itemsize, channels=1)
            monos.append(dubseg)

        return pydub.AudioSegment.from_mono_audiosegments(*monos)

    def get_similarity(self, embedding1, embedding2):
        # print()
        # the resulting embeddings can be used for cosine similarity-based retrieval
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        similarity = cosine_sim(embedding1, embedding2)

        threshold = 0.80
        
        if similarity < threshold:
            # print("Speakers are not the same!")
            pass
            
        else:    
            print()
            print("Speakers are the same!")
            print(f'Similarity: {similarity}')
            print()
        # print()
        return similarity
    
    def callback(self, indata, frames, time, status):
        for val in indata:
            self.buffer.append(val)
        if len(self.buffer) >= 16000:
            #ensure 16000 samples
            self.buffer = self.buffer[-16000:]
            # pad to audio 16000 samples if less than 16000 samples
            embeddings = self.get_embeddings(audio=self.buffer)
            similarity = self.get_similarity(self.audio_embeddings, embeddings)
            # print(similarity)

    def get_similarity_from_stream_given_audio(self, audio_path):
        self.audio = self.get_audio_array_from_audio_path(audio_path)
        self.audio_embeddings = self.get_embeddings(audio_path=audio_path)
        # open a sounddevice input stream from the default mic and get similarity
        # from the last 1 second of audio vs the input audio path (audio_path)
        # be sure to use sample rate = 16000
        self.buffer = []
        with InputStream(callback=self.callback, blocksize=int(16000/4), samplerate=16000):
            input()


if __name__ == "__main__":
    
    speaker_recognition = SpeakerRecognition()

    omar_heyditto = 'data/sample_audio/omar_heyditto.wav'

    speaker_recognition.get_similarity_from_stream_given_audio(omar_heyditto)

    # speaker1_heyditto = 'data/sample_audio/dan_heyditto.mp3'
    # speaker1_talking = 'data/sample_audio/dan_talking.mp3'
    # speaker2_heyditto = 'data/sample_audio/mark_heyditto.mp3'
    # speaker2_talking = 'data/sample_audio/mark_talking.mp3'

    # print('speaker1_heyditto vs speaker1_talking')
    # embedddings1 = speaker_recognition.get_embeddings(audio_path=speaker1_heyditto)
    # embedddings2 = speaker_recognition.get_embeddings(audio_path=speaker1_talking)
    # speaker_recognition.get_similarity(embedddings1, embedddings2)

    # print('speaker1_heyditto vs speaker2_heyditto')
    # embedddings1 = speaker_recognition.get_embeddings(audio_path=speaker1_heyditto)
    # embedddings2 = speaker_recognition.get_embeddings(audio_path=speaker2_heyditto)
    # speaker_recognition.get_similarity(embedddings1, embedddings2)

    # print('speaker2_heyditto vs speaker2_talking')
    # embedddings1 = speaker_recognition.get_embeddings(audio_path=speaker2_heyditto)
    # embedddings2 = speaker_recognition.get_embeddings(audio_path=speaker2_talking)
    # speaker_recognition.get_similarity(embedddings1, embedddings2)

    # print('speaker2_heyditto vs speaker1_talking')
    # embedddings1 = speaker_recognition.get_embeddings(audio_path=speaker2_heyditto)
    # embedddings2 = speaker_recognition.get_embeddings(audio_path=speaker1_talking)
    # speaker_recognition.get_similarity(embedddings1, embedddings2)  