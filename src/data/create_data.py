"""
Dataset creation for Hey Ditto wake word detection.

This script processes audio samples from data/1/ (positive) and data/0/ (negative)
folders, applies data augmentation, generates spectrograms, and saves the final
dataset as x_data.npy and y_data.npy files.

Usage:
    python create_data.py
    python create_data.py --augment-ratio 0.6
    python create_data.py --no-augment
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import Tuple, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from utils.audio_utils import normalize_audio_file, load_audio, normalize_audio
from utils.spec_utils import get_spectrogram
from utils.augmentation import (
    white_noise, stretch, downsample_audio, lower_volume,
    rand_pitch, combine_with, lower_volume_array, rand_pitch_array
)

SAMPLE_RATE = 16000
DATA_DIR = Path(__file__).parent.parent.parent / "data"
POSITIVE_DIR = DATA_DIR / "1"
NEGATIVE_DIR = DATA_DIR / "0"
BACKGROUND_DIR = DATA_DIR / "background_noise"

ACTIVATION_AUGMENTATION_PERCENT = 0.6
BACKGROUND_AUGMENTATION_PERCENT = 0.5


def load_audio_files(directory: Path, recursive: bool = False) -> List[Path]:
    """Load all audio file paths from a directory."""
    if not directory.exists():
        print(f"Warning: Directory {directory} does not exist")
        return []

    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    files = []

    if recursive:
        for ext in audio_extensions:
            files.extend(directory.rglob(f"*{ext}"))
    else:
        for file in directory.iterdir():
            if file.suffix.lower() in audio_extensions:
                files.append(file)

    return files


def mix_with_background_snr(
    audio: np.ndarray,
    background: np.ndarray,
    snr_db: float
) -> np.ndarray:
    """Mix audio with background at a specific SNR level."""
    if len(background) < len(audio):
        background = np.pad(background, (0, len(audio) - len(background)))
    else:
        if len(background) > len(audio):
            start = random.randint(0, len(background) - len(audio))
            background = background[start:start + len(audio)]

    audio_rms = np.sqrt(np.mean(audio ** 2)) + 1e-10
    bg_rms = np.sqrt(np.mean(background ** 2)) + 1e-10

    target_bg_rms = audio_rms / (10 ** (snr_db / 20.0))
    background = background * (target_bg_rms / bg_rms)

    mixed = audio + background

    max_val = np.max(np.abs(mixed))
    if max_val > 0.95:
        mixed = mixed * (0.95 / max_val)

    return mixed.astype(np.float32)


def process_positive_samples(
    positive_files: List[Path],
    background_files: List[Path],
    augment_ratio: float = ACTIVATION_AUGMENTATION_PERCENT
) -> Tuple[List[np.ndarray], List[int]]:
    """Process positive samples with augmentation."""
    x_data = []
    y_data = []

    print(f"\nProcessing {len(positive_files)} positive samples...")

    for file_path in tqdm(positive_files, desc="Positive samples"):
        try:
            audio = normalize_audio_file(str(file_path))

            spect = get_spectrogram(audio)
            x_data.append(spect)
            y_data.append(1)

            if random.random() > augment_ratio:
                continue

            augmented_samples = []

            # 1. Lower volume
            audio_quiet = lower_volume_array(audio, db=random.uniform(3.0, 9.0))
            augmented_samples.append(audio_quiet)

            # 2. Random pitch shift
            audio_pitch = rand_pitch_array(audio)
            augmented_samples.append(audio_pitch)

            # 3. White noise (low)
            audio_noise1 = white_noise(audio, amount=random.uniform(0.002, 0.1))
            augmented_samples.append(audio_noise1)

            # 4. White noise (higher)
            audio_noise2 = white_noise(audio, amount=random.uniform(0.05, 0.2))
            augmented_samples.append(audio_noise2)

            # 5. Downsampled
            audio_downsampled = downsample_audio(audio)
            augmented_samples.append(audio_downsampled)

            # 6. Background mixing at various SNR levels
            if background_files:
                snr_levels = [20, 15, 10, 5, 3]
                num_bg_samples = min(3, len(background_files))
                for snr in random.sample(snr_levels, min(num_bg_samples, len(snr_levels))):
                    bg_file = random.choice(background_files)
                    try:
                        bg_audio, _ = load_audio(str(bg_file))
                        combined = mix_with_background_snr(audio, bg_audio, snr)
                        augmented_samples.append(combined)
                    except Exception:
                        pass

            for aug_audio in augmented_samples:
                spect = get_spectrogram(aug_audio)
                x_data.append(spect)
                y_data.append(1)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    return x_data, y_data


def process_negative_samples(
    negative_files: List[Path],
    augment_ratio: float = BACKGROUND_AUGMENTATION_PERCENT
) -> Tuple[List[np.ndarray], List[int]]:
    """Process negative samples with augmentation."""
    x_data = []
    y_data = []

    print(f"\nProcessing {len(negative_files)} negative samples...")

    for file_path in tqdm(negative_files, desc="Negative samples"):
        try:
            audio = normalize_audio_file(str(file_path))

            spect = get_spectrogram(audio)
            x_data.append(spect)
            y_data.append(0)

            if random.random() > augment_ratio:
                continue

            augmented_samples = []

            # 1. White noise
            audio_noise = white_noise(audio, amount=random.uniform(0.002, 0.3))
            augmented_samples.append(audio_noise)

            # 2. Lower volume
            audio_quiet = lower_volume_array(audio, db=random.uniform(10, 20))
            augmented_samples.append(audio_quiet)

            # 3. Downsampled
            audio_downsampled = downsample_audio(audio)
            augmented_samples.append(audio_downsampled)

            for aug_audio in augmented_samples:
                spect = get_spectrogram(aug_audio)
                x_data.append(spect)
                y_data.append(0)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    return x_data, y_data


def create_dataset(
    augment_ratio_positive: float = ACTIVATION_AUGMENTATION_PERCENT,
    augment_ratio_negative: float = BACKGROUND_AUGMENTATION_PERCENT,
    shuffle: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Create the complete dataset from data/1/ and data/0/ folders."""
    positive_files = load_audio_files(POSITIVE_DIR)
    negative_files = load_audio_files(NEGATIVE_DIR)
    background_files = load_audio_files(BACKGROUND_DIR, recursive=True)

    print(f"\n{'='*50}")
    print("Dataset Creation Summary")
    print(f"{'='*50}")
    print(f"Positive samples: {len(positive_files)}")
    print(f"Negative samples: {len(negative_files)}")
    print(f"Background noise files: {len(background_files)}")
    print(f"Positive augmentation ratio: {augment_ratio_positive}")
    print(f"Negative augmentation ratio: {augment_ratio_negative}")
    print(f"{'='*50}\n")

    if len(positive_files) == 0 and len(negative_files) == 0:
        raise ValueError(
            f"No audio files found in {POSITIVE_DIR} or {NEGATIVE_DIR}. "
            "Please add audio samples or run generate_samples.py first."
        )

    x_pos, y_pos = process_positive_samples(
        positive_files, background_files, augment_ratio_positive
    )
    x_neg, y_neg = process_negative_samples(
        negative_files, augment_ratio_negative
    )

    x_data = x_pos + x_neg
    y_data = y_pos + y_neg

    x_data = np.array(x_data).astype('float32')
    y_data = np.array(y_data).astype('int32')

    if shuffle:
        indices = np.random.permutation(len(y_data))
        x_data = x_data[indices]
        y_data = y_data[indices]

    pos_count = np.sum(y_data == 1)
    neg_count = np.sum(y_data == 0)

    print(f"\n{'='*50}")
    print("Final Dataset")
    print(f"{'='*50}")
    print(f"X shape: {x_data.shape}")
    print(f"Y shape: {y_data.shape}")
    print(f"Positive samples: {pos_count} ({100*pos_count/len(y_data):.1f}%)")
    print(f"Negative samples: {neg_count} ({100*neg_count/len(y_data):.1f}%)")
    print(f"{'='*50}\n")

    return x_data, y_data


def save_dataset(
    x_data: np.ndarray,
    y_data: np.ndarray,
    output_dir: Path = DATA_DIR
) -> None:
    """Save the dataset to numpy files."""
    x_path = output_dir / "x_data.npy"
    y_path = output_dir / "y_data.npy"

    np.save(str(x_path), x_data)
    np.save(str(y_path), y_data)

    print(f"Saved x_data.npy: {x_path} ({x_data.nbytes / 1024 / 1024:.2f} MB)")
    print(f"Saved y_data.npy: {y_path} ({y_data.nbytes / 1024:.2f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Create training dataset for Hey Ditto wake word detection"
    )
    parser.add_argument(
        "--augment-ratio", type=float, default=0.6,
        help="Ratio of samples to augment (0.0-1.0, default: 0.6)"
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Skip augmentation (faster, for testing)"
    )
    parser.add_argument(
        "--no-shuffle", action="store_true",
        help="Don't shuffle the dataset"
    )

    args = parser.parse_args()

    if args.no_augment:
        augment_ratio_pos = 0.0
        augment_ratio_neg = 0.0
    else:
        augment_ratio_pos = args.augment_ratio
        augment_ratio_neg = args.augment_ratio * 0.8

    print("\n" + "="*60)
    print("Hey Ditto - Dataset Creator")
    print("="*60)

    x_data, y_data = create_dataset(
        augment_ratio_positive=augment_ratio_pos,
        augment_ratio_negative=augment_ratio_neg,
        shuffle=not args.no_shuffle
    )

    save_dataset(x_data, y_data)

    print("\nDone! Dataset is ready for training.")
    print("Run 'python src/train.py' to train the model.")


if __name__ == "__main__":
    main()
