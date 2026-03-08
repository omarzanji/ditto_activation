"""
Audio utilities for Hey Ditto wake word detection.

Provides functions for loading, normalizing, and saving audio files.
All audio is processed at 16kHz sample rate with 1.5-second duration.
"""

import os
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment
from pydub import effects
from typing import Tuple, Optional

# Constants
SAMPLE_RATE = 16000
AUDIO_DURATION = 1.5  # seconds (enough for "Hey Ditto" at various speech rates)
TARGET_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)  # 24000 samples for 1.5s


def load_audio(
    file_path: str,
    sr: int = SAMPLE_RATE,
    duration: float = AUDIO_DURATION,
    mono: bool = True,
    pad_to_duration: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, resample to target rate, and ensure fixed duration.

    Args:
        file_path: Path to audio file (WAV, MP3, etc.)
        sr: Target sample rate (default: 16000 Hz)
        duration: Target duration in seconds (default: 1.5)
        mono: Convert to mono if True (default: True)
        pad_to_duration: If True, pad short audio to target duration (default: True)

    Returns:
        Tuple of (audio samples as float32 numpy array, sample rate)
    """
    y, orig_sr = librosa.load(file_path, sr=sr, mono=mono, duration=None)

    target_samples = int(sr * duration)

    if len(y) > target_samples:
        y = y[:target_samples]
    elif len(y) < target_samples and pad_to_duration:
        y = np.pad(y, (0, target_samples - len(y)), mode='constant')

    return y.astype(np.float32), sr


def normalize_audio(signal: np.ndarray, rms_level: float = -2.0) -> np.ndarray:
    """
    Normalize the audio signal using RMS normalization.

    Args:
        signal: Audio samples as numpy array
        rms_level: Target RMS level in dB (default: -2 dB)

    Returns:
        Normalized audio samples as float32 numpy array
    """
    try:
        signal = np.array(signal).astype('float32')

        if np.sum(signal**2) == 0:
            return signal

        r = 10 ** (rms_level / 10.0)
        a = np.sqrt((len(signal) * r**2) / np.sum(signal**2))

        y = signal * a

        return y.astype(np.float32)

    except Exception as e:
        print(f"Normalization error: {e}")
        return signal.astype(np.float32)


def normalize_audio_file(file_path: str, sr: int = SAMPLE_RATE, duration: float = AUDIO_DURATION) -> np.ndarray:
    """
    Load and normalize an audio file using pydub's normalize function.

    Args:
        file_path: Path to audio file
        sr: Target sample rate (default: 16000 Hz)
        duration: Target duration in seconds (default: 1.5)

    Returns:
        Normalized audio samples as float32 numpy array, padded to target duration
    """
    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    if ext in ('wav', 'wave'):
        audio = AudioSegment.from_wav(file_path)
    elif ext == 'mp3':
        audio = AudioSegment.from_mp3(file_path)
    else:
        audio = AudioSegment.from_file(file_path)

    audio_norm = effects.normalize(audio)

    samples = audio_norm.get_array_of_samples()

    target_samples = int(sr * duration)

    if len(samples) > target_samples:
        samples = samples[:target_samples]
    elif len(samples) < target_samples:
        samples = np.pad(samples, (0, target_samples - len(samples)), mode='constant')

    return np.array(samples).astype(np.float32) / 32768.0


def save_audio(
    samples: np.ndarray,
    file_path: str,
    sr: int = SAMPLE_RATE
) -> None:
    """
    Save audio samples to a WAV file.

    Args:
        samples: Audio samples as numpy array
        file_path: Output file path
        sr: Sample rate (default: 16000 Hz)
    """
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)

    samples = np.clip(samples, -1.0, 1.0)

    sf.write(file_path, samples, sr)


def get_audio_duration(file_path: str) -> float:
    """Get the duration of an audio file in seconds."""
    return librosa.get_duration(path=file_path)


def resample_audio(
    samples: np.ndarray,
    orig_sr: int,
    target_sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Resample audio to a different sample rate."""
    if orig_sr == target_sr:
        return samples

    return librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr)


def ensure_mono(samples: np.ndarray) -> np.ndarray:
    """Convert stereo audio to mono by averaging channels."""
    if samples.ndim > 1:
        return np.mean(samples, axis=1)
    return samples
