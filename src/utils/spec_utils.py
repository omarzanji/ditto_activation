"""
Spectrogram utilities for Hey Ditto wake word detection.

Provides functions for converting audio waveforms to spectrograms
suitable for the HeyDittoNet model. Uses log-filterbank features
which are well-suited for speech recognition tasks.
"""

import numpy as np
import tensorflow as tf
from python_speech_features import logfbank
from typing import List

# Constants
SAMPLE_RATE = 16000
WINDOW = int(SAMPLE_RATE / 4)
STRIDE = int((SAMPLE_RATE - WINDOW) / 4)


def get_spectrogram(waveform: np.ndarray, sr: int = SAMPLE_RATE, duration: float = 1.5) -> np.ndarray:
    """
    Convert a 16kHz waveform to a log-filterbank spectrogram.

    Uses 32 mel-frequency filterbanks which capture the important
    frequency information for speech recognition.

    Args:
        waveform: Audio samples as numpy array
        sr: Sample rate (default: 16000 Hz)
        duration: Audio duration in seconds (default: 1.5s)

    Returns:
        Spectrogram as float32 numpy array with shape (frames, 32, 1)
        where frames depends on the waveform length (~149 for 1.5 second audio)
    """
    input_len = int(sr * duration)
    waveform = waveform[:input_len]

    waveform = tf.cast(waveform, dtype=tf.float32)

    target_len = input_len
    zero_padding = tf.zeros(target_len - tf.shape(waveform)[0], dtype=tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)

    fbank_feat = logfbank(equal_length.numpy(), sr, nfilt=32)

    spectrogram = fbank_feat[..., np.newaxis]

    return np.array(spectrogram).astype('float32')


def get_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
    fmax: int = 8000
) -> np.ndarray:
    """
    Alternative spectrogram using librosa's mel-spectrogram.

    Args:
        waveform: Audio samples as numpy array
        sr: Sample rate (default: 16000 Hz)
        n_mels: Number of mel bands (default: 64)
        n_fft: FFT window size (default: 1024)
        hop_length: Hop length between frames (default: 256)
        fmax: Maximum frequency (default: 8000 Hz)

    Returns:
        Log-mel spectrogram as float32 numpy array with shape (n_mels, frames, 1)
    """
    import librosa

    S = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=fmax
    )

    S_dB = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_dB + 80) / 80
    spectrogram = S_norm.T[..., np.newaxis]

    return spectrogram.astype('float32')


def visualize_spectrogram(
    spectrogram: np.ndarray,
    title: str = "Spectrogram",
    save_path: str = None
) -> None:
    """Visualize a spectrogram using matplotlib."""
    import matplotlib.pyplot as plt

    if spectrogram.ndim == 3:
        spectrogram = spectrogram[:, :, 0]

    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
