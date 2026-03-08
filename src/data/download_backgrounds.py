"""
Download free background audio datasets for Hey Ditto training.

Downloads and processes free environmental sound datasets to create
diverse background/negative samples for wake word detection.

Datasets:
- ESC-50: Environmental Sound Classification (2,000 clips, 50 categories)

Usage:
    python download_backgrounds.py
    python download_backgrounds.py --dataset esc50
    python download_backgrounds.py --force
"""

import os
import sys
import argparse
import zipfile
import shutil
import random
from pathlib import Path
import urllib.request
import ssl

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
import librosa

SAMPLE_RATE = 16000
CLIP_DURATION = 1.5  # 1.5 seconds to match model input
DATA_DIR = Path(__file__).parent.parent.parent / "data"
BACKGROUND_DIR = DATA_DIR / "background_noise"
NEGATIVE_DIR = DATA_DIR / "0"

DATASETS = {
    "esc50": {
        "url": "https://github.com/karoldvl/ESC-50/archive/master.zip",
        "description": "Environmental Sound Classification - 50 categories, 2000 clips",
        "audio_subdir": "ESC-50-master/audio",
    },
}


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> bool:
    """Download a file with progress indication."""
    try:
        print(f"{desc}: {url}")
        print(f"Saving to: {dest_path}")

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        def reporthook(count, block_size, total_size):
            percent = min(100, int(count * block_size * 100 / total_size)) if total_size > 0 else 0
            print(f"\r  Progress: {percent}%", end="", flush=True)

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(dest_path), reporthook)
        print()

        return True
    except Exception as e:
        print(f"\nError downloading: {e}")
        return False


def extract_archive(archive_path: Path, extract_dir: Path) -> bool:
    """Extract a zip archive."""
    try:
        print(f"Extracting: {archive_path}")
        extract_dir.mkdir(parents=True, exist_ok=True)

        if str(archive_path).endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(extract_dir)
        else:
            print(f"Unknown archive format: {archive_path}")
            return False

        print("Extraction complete!")
        return True
    except Exception as e:
        print(f"Error extracting: {e}")
        return False


def process_audio_to_clips(
    input_dir: Path,
    output_dir: Path,
    clip_duration: float = CLIP_DURATION,
    max_clips_per_file: int = 5,
    max_total_clips: int = 1000
) -> int:
    """Process audio files into clips matching model input duration."""
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_extensions = {'.wav', '.mp3', '.ogg', '.flac'}
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(input_dir.rglob(f"*{ext}"))

    print(f"Found {len(audio_files)} audio files")

    clips_created = 0
    target_samples = int(clip_duration * SAMPLE_RATE)

    random.shuffle(audio_files)

    for audio_file in audio_files:
        if clips_created >= max_total_clips:
            break

        try:
            y, sr = librosa.load(str(audio_file), sr=SAMPLE_RATE, mono=True)

            if len(y) < target_samples:
                y = np.pad(y, (0, target_samples - len(y)))
                clips_from_file = 1
            else:
                clips_from_file = min(
                    max_clips_per_file,
                    len(y) // target_samples
                )

            for i in range(clips_from_file):
                if clips_created >= max_total_clips:
                    break

                start = i * target_samples
                if start + target_samples > len(y):
                    start = random.randint(0, len(y) - target_samples)

                clip = y[start:start + target_samples]

                if np.max(np.abs(clip)) > 0:
                    clip = clip / np.max(np.abs(clip)) * 0.9

                clip_name = f"bg_{audio_file.stem}_{i}_{clips_created}.wav"
                sf.write(str(output_dir / clip_name), clip, SAMPLE_RATE)
                clips_created += 1

        except Exception:
            continue

    print(f"Created {clips_created} background clips")
    return clips_created


def download_esc50(force: bool = False) -> int:
    """Download and process ESC-50 dataset."""
    print("\n" + "="*60)
    print("Downloading ESC-50 Dataset")
    print("="*60)

    esc50_dir = BACKGROUND_DIR / "esc50"
    existing_files = list(esc50_dir.glob("*.wav")) if esc50_dir.exists() else []

    if len(existing_files) >= 400 and not force:
        print(f"ESC-50 already downloaded ({len(existing_files)} files). Use --force to re-download.")
        return len(existing_files)

    dataset = DATASETS["esc50"]
    temp_dir = DATA_DIR / "temp_esc50"
    archive_path = temp_dir / "esc50.zip"

    if not download_file(dataset["url"], archive_path, "Downloading ESC-50"):
        return 0

    if not extract_archive(archive_path, temp_dir):
        return 0

    audio_dir = temp_dir / dataset["audio_subdir"]
    clips = process_audio_to_clips(
        audio_dir,
        BACKGROUND_DIR / "esc50",
        max_clips_per_file=2,
        max_total_clips=500
    )

    process_audio_to_clips(
        audio_dir,
        NEGATIVE_DIR,
        max_clips_per_file=1,
        max_total_clips=200
    )

    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir, ignore_errors=True)

    return clips


def generate_synthetic_noise(output_dir: Path, count: int = 200, force: bool = False) -> int:
    """Generate synthetic noise samples (white, pink, brown, near-silence)."""
    print("\n" + "="*60)
    print("Generating Synthetic Noise")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(output_dir.glob("synthetic_*.wav"))
    if len(existing_files) >= count and not force:
        print(f"Synthetic noise already exists ({len(existing_files)} files). Use --force to regenerate.")
        return len(existing_files)

    target_samples = int(CLIP_DURATION * SAMPLE_RATE)
    created = 0

    for i in range(count):
        noise_type = random.choice(["white", "pink", "brown", "silence"])

        if noise_type == "white":
            noise = np.random.randn(target_samples) * random.uniform(0.1, 0.5)
        elif noise_type == "pink":
            white = np.random.randn(target_samples)
            pink = np.cumsum(white)
            pink = pink - np.mean(pink)
            pink = pink / np.max(np.abs(pink)) * random.uniform(0.1, 0.5)
            noise = pink
        elif noise_type == "brown":
            noise = np.cumsum(np.random.randn(target_samples) * 0.01)
            noise = noise - np.mean(noise)
            noise = noise / np.max(np.abs(noise)) * random.uniform(0.1, 0.3)
        else:
            noise = np.random.randn(target_samples) * random.uniform(0.001, 0.01)

        noise = np.clip(noise, -1.0, 1.0).astype(np.float32)

        filename = f"synthetic_{noise_type}_{i}.wav"
        sf.write(str(output_dir / filename), noise, SAMPLE_RATE)
        created += 1

    print(f"Created {created} synthetic noise samples")
    return created


def create_mixed_backgrounds(
    background_dir: Path,
    output_dir: Path,
    count: int = 200,
    force: bool = False,
    start_index: int = 0
) -> int:
    """Create mixed background samples by combining multiple sources."""
    print("\n" + "="*60)
    print("Creating Mixed Backgrounds")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)

    existing_mixed = list(output_dir.glob("mixed_bg_*.wav"))
    if len(existing_mixed) >= count and not force and start_index == 0:
        print(f"Mixed backgrounds already exist ({len(existing_mixed)} files). Use --force to regenerate.")
        return len(existing_mixed)

    if start_index > 0:
        print(f"Adding {count} more mixed backgrounds starting at index {start_index}")

    bg_files = list(background_dir.rglob("*.wav"))
    if len(bg_files) < 2:
        print("Not enough background files to mix")
        return 0

    target_samples = int(CLIP_DURATION * SAMPLE_RATE)
    created = 0

    for i in range(count):
        try:
            num_sources = random.randint(2, min(3, len(bg_files)))
            sources = random.sample(bg_files, num_sources)

            mixed = np.zeros(target_samples)

            for src_path in sources:
                y, _ = librosa.load(str(src_path), sr=SAMPLE_RATE, mono=True)
                if len(y) < target_samples:
                    y = np.pad(y, (0, target_samples - len(y)))
                else:
                    y = y[:target_samples]

                gain = random.uniform(0.3, 0.7)
                mixed += y * gain

            if np.max(np.abs(mixed)) > 0:
                mixed = mixed / np.max(np.abs(mixed)) * 0.9

            file_index = start_index + i
            filename = f"mixed_bg_{file_index}.wav"
            sf.write(str(output_dir / filename), mixed.astype(np.float32), SAMPLE_RATE)
            created += 1

        except Exception:
            continue

    print(f"Created {created} mixed background samples")
    return created


def main():
    parser = argparse.ArgumentParser(
        description="Download free background audio datasets"
    )
    parser.add_argument(
        "--dataset", choices=["esc50", "all"], default="all",
        help="Which dataset to download (default: all)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: data/background_noise)"
    )
    parser.add_argument(
        "--synthetic-count", type=int, default=200,
        help="Number of synthetic noise samples to generate"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download/regenerate even if files already exist"
    )
    parser.add_argument(
        "--more-mixed", type=int, default=0,
        help="Generate additional mixed background samples"
    )
    parser.add_argument(
        "--production", action="store_true",
        help="Production mode: generate 1000+ mixed background samples"
    )

    args = parser.parse_args()

    if args.output:
        global BACKGROUND_DIR
        BACKGROUND_DIR = Path(args.output)

    print("\n" + "="*60)
    print("Hey Ditto - Background Audio Downloader")
    print("="*60)
    print(f"Output directory: {BACKGROUND_DIR}")
    print(f"Negative class directory: {NEGATIVE_DIR}")
    if args.force:
        print("FORCE MODE: Will re-download/regenerate all files")

    BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)
    NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)

    total_clips = 0

    if args.dataset in ["esc50", "all"]:
        total_clips += download_esc50(force=args.force)

    total_clips += generate_synthetic_noise(
        BACKGROUND_DIR / "synthetic",
        args.synthetic_count,
        force=args.force
    )

    mixed_count = 100
    if args.production:
        mixed_count = 1000

    if total_clips > 10:
        total_clips += create_mixed_backgrounds(
            BACKGROUND_DIR,
            NEGATIVE_DIR,
            count=mixed_count,
            force=args.force
        )

    if args.more_mixed > 0:
        existing_mixed = list(NEGATIVE_DIR.glob("mixed_bg_*.wav"))
        start_index = len(existing_mixed)
        print(f"\nGenerating {args.more_mixed} additional mixed backgrounds...")
        total_clips += create_mixed_backgrounds(
            BACKGROUND_DIR,
            NEGATIVE_DIR,
            count=args.more_mixed,
            force=True,
            start_index=start_index
        )

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total background clips: {total_clips}")
    print(f"Background noise directory: {BACKGROUND_DIR}")
    print(f"Negative samples directory: {NEGATIVE_DIR}")

    neg_files = list(NEGATIVE_DIR.glob("*.wav"))
    print(f"Total negative samples in data/0/: {len(neg_files)}")
    print("\nDone!")


if __name__ == "__main__":
    main()
