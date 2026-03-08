"""
Data augmentation utilities for Hey Ditto wake word detection.

Provides functions for augmenting audio samples to increase dataset
diversity and improve model robustness.
"""

import os
import random
import numpy as np
import librosa
from pydub import AudioSegment
from pydub import effects
from typing import Optional, List

# Constants
SAMPLE_RATE = 16000
AUDIO_DURATION = 1.5  # seconds
MAX_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)  # 24000 samples for 1.5 seconds at 16kHz


def white_noise(sample: np.ndarray, amount: float = 0.005) -> np.ndarray:
    """Add white (gaussian) noise to an audio sample."""
    wn = np.random.randn(len(sample))
    data_wn = sample + amount * wn
    return data_wn.astype(np.float32)


def stretch(sample: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """Time-stretch an audio sample without changing pitch."""
    data = librosa.effects.time_stretch(sample, rate=rate)

    if len(data) > MAX_SAMPLES:
        data = data[:MAX_SAMPLES]
    elif len(data) < MAX_SAMPLES:
        data = np.pad(data, (0, max(0, MAX_SAMPLES - len(data))))

    return data.astype(np.float32)


def downsample_audio(sample: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simulate low-quality audio by downsampling and upsampling."""
    random_downscale_sr = np.random.randint(4000, 8000)

    downsampled = librosa.resample(
        y=sample,
        orig_sr=sr,
        target_sr=random_downscale_sr
    )

    upsampled = librosa.resample(
        y=downsampled,
        orig_sr=random_downscale_sr,
        target_sr=sr
    )

    if len(upsampled) > MAX_SAMPLES:
        upsampled = upsampled[:MAX_SAMPLES]
    elif len(upsampled) < MAX_SAMPLES:
        upsampled = np.pad(upsampled, (0, max(0, MAX_SAMPLES - len(upsampled))))

    return upsampled.astype(np.float32)


def lower_volume(file_path: str, db: float = 10.0) -> np.ndarray:
    """Reduce the volume of an audio file."""
    raw_audio = AudioSegment.from_wav(file_path)
    audio = effects.normalize(raw_audio)
    audio = audio - db
    samples = audio.get_array_of_samples()
    audio._spawn(samples)
    return np.array(samples).astype(np.float32) / 32768.0


def lower_volume_array(sample: np.ndarray, db: float = 10.0) -> np.ndarray:
    """Reduce the volume of an audio sample array."""
    scale = 10 ** (-db / 20.0)
    return (sample * scale).astype(np.float32)


def rand_pitch(file_path: str) -> np.ndarray:
    """Apply random pitch shifting to an audio file."""
    sound = AudioSegment.from_wav(file_path)

    rand = random.random()
    octaves = random.uniform(-0.4, -0.2) if rand >= 0.5 else random.uniform(0.2, 0.4)

    new_sample_rate = int(sound.frame_rate * (2.0 ** octaves))
    hipitch_sound = sound._spawn(
        sound.raw_data,
        overrides={'frame_rate': new_sample_rate}
    )

    audio = hipitch_sound.set_frame_rate(SAMPLE_RATE)
    audio = effects.normalize(audio)
    samples = audio.get_array_of_samples()
    audio._spawn(samples)

    return np.array(samples).astype(np.float32) / 32768.0


def rand_pitch_array(sample: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply random pitch shifting to an audio sample array."""
    n_steps = random.uniform(-4, 4)

    shifted = librosa.effects.pitch_shift(
        y=sample,
        sr=sr,
        n_steps=n_steps
    )

    if len(shifted) > MAX_SAMPLES:
        shifted = shifted[:MAX_SAMPLES]
    elif len(shifted) < MAX_SAMPLES:
        shifted = np.pad(shifted, (0, max(0, MAX_SAMPLES - len(shifted))))

    return shifted.astype(np.float32)


def combine_with(activation_path: str, background_path: str) -> np.ndarray:
    """Overlay an activation sample with background noise."""
    decrease_amount = np.random.uniform(
        2 if 'music' in background_path else 4,
        6
    )

    a_audio = AudioSegment.from_wav(activation_path)
    a_audio_norm = effects.normalize(a_audio)

    b_audio = AudioSegment.from_wav(background_path)
    b_audio_norm = effects.normalize(b_audio) - decrease_amount

    combined = a_audio_norm.overlay(b_audio_norm)
    samples = combined.get_array_of_samples()

    return np.array(samples).astype(np.float32) / 32768.0


def combine_arrays(
    activation: np.ndarray,
    background: np.ndarray,
    snr_db: float = 10.0
) -> np.ndarray:
    """Overlay an activation sample with background noise using arrays."""
    min_len = min(len(activation), len(background))
    activation = activation[:min_len]
    background = background[:min_len]

    activation_rms = np.sqrt(np.mean(activation ** 2))
    background_rms = np.sqrt(np.mean(background ** 2))

    if background_rms > 0:
        target_background_rms = activation_rms / (10 ** (snr_db / 20.0))
        scale = target_background_rms / background_rms
        background = background * scale

    combined = activation + background

    max_val = np.max(np.abs(combined))
    if max_val > 1.0:
        combined = combined / max_val

    return combined.astype(np.float32)


def apply_random_augmentation(
    sample: np.ndarray,
    noise_paths: Optional[List[str]] = None,
    p: float = 0.5
) -> np.ndarray:
    """Apply a random augmentation to an audio sample."""
    if random.random() > p:
        return sample

    augmentations = [
        lambda s: white_noise(s, amount=random.uniform(0.002, 0.1)),
        lambda s: stretch(s, rate=random.uniform(0.9, 1.1)),
        lambda s: downsample_audio(s),
        lambda s: lower_volume_array(s, db=random.uniform(3, 15)),
        lambda s: rand_pitch_array(s),
    ]

    if noise_paths and len(noise_paths) > 0:
        def mix_with_background(s):
            noise_path = random.choice(noise_paths)
            try:
                noise, _ = librosa.load(noise_path, sr=SAMPLE_RATE, duration=AUDIO_DURATION)
                if len(noise) < len(s):
                    noise = np.pad(noise, (0, len(s) - len(noise)))
                return combine_arrays(s, noise[:len(s)], snr_db=random.uniform(5, 15))
            except Exception:
                return s
        augmentations.append(mix_with_background)

    aug_func = random.choice(augmentations)
    return aug_func(sample)


def apply_multiple_augmentations(
    sample: np.ndarray,
    noise_paths: Optional[List[str]] = None,
    num_augmentations: int = 3
) -> List[np.ndarray]:
    """Apply multiple different augmentations to create several variants."""
    augmented = []

    for _ in range(num_augmentations):
        aug_sample = apply_random_augmentation(sample, noise_paths, p=1.0)
        augmented.append(aug_sample)

    return augmented
