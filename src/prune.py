"""
Post-training dataset pruning tool for Hey Ditto wake word detection.

Analyzes each audio sample against the trained model to find:
- False positives: negatives scored above threshold
- Weak positives: positives scored below threshold
- Borderline samples: anything within ±margin of threshold

Flagged samples can be reviewed, played, and quarantined (moved to data/quarantine/).

Usage:
    python src/prune.py                     # Analyze and print report
    python src/prune.py --threshold 0.5     # Custom threshold
    python src/prune.py --listen            # Play audio for each flagged sample
    python src/prune.py --delete            # Move flagged to data/quarantine/
    python src/prune.py --export report.csv # Export full analysis
"""

import os
import sys
import csv
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils.audio_utils import load_audio, normalize_audio
from utils.spec_utils import get_spectrogram

DATA_DIR = Path(__file__).parent.parent / "data"
POSITIVE_DIR = DATA_DIR / "1"
NEGATIVE_DIR = DATA_DIR / "0"
QUARANTINE_DIR = DATA_DIR / "quarantine"
MODELS_DIR = Path(__file__).parent.parent / "models"


def load_model(model_path: str = None):
    """Load trained model."""
    if model_path is None:
        model_path = str(MODELS_DIR / "HeyDittoNet-v3.keras")

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run training first: python src/train.py"
        )

    model = keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")
    return model


def predict_file(model, file_path: str) -> float:
    """Run prediction on a single audio file."""
    audio, sr = load_audio(file_path)
    audio = normalize_audio(audio)
    spectrogram = get_spectrogram(audio)
    pred = model.predict(np.expand_dims(spectrogram, 0), verbose=0)
    return float(pred[0][0])


def analyze_samples(
    model,
    threshold: float = 0.5,
    margin: float = 0.15
) -> dict:
    """
    Analyze all samples against the model.

    Returns dict with:
        - false_positives: negatives scored > threshold
        - weak_positives: positives scored < threshold
        - borderline: anything within ±margin of threshold
        - all_results: full list of (path, label, score)
    """
    results = {
        "false_positives": [],
        "weak_positives": [],
        "borderline": [],
        "all_results": [],
    }

    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}

    # Process positive samples
    if POSITIVE_DIR.exists():
        pos_files = [f for f in POSITIVE_DIR.iterdir() if f.suffix.lower() in audio_extensions]
        print(f"\nAnalyzing {len(pos_files)} positive samples...")

        for i, file_path in enumerate(pos_files):
            try:
                score = predict_file(model, str(file_path))
                results["all_results"].append((str(file_path), 1, score))

                if score < threshold:
                    results["weak_positives"].append((str(file_path), score))
                if abs(score - threshold) < margin:
                    results["borderline"].append((str(file_path), 1, score))

                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(pos_files)} positive samples...")

            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")

    # Process negative samples
    if NEGATIVE_DIR.exists():
        neg_files = [f for f in NEGATIVE_DIR.iterdir() if f.suffix.lower() in audio_extensions]
        print(f"\nAnalyzing {len(neg_files)} negative samples...")

        for i, file_path in enumerate(neg_files):
            try:
                score = predict_file(model, str(file_path))
                results["all_results"].append((str(file_path), 0, score))

                if score > threshold:
                    results["false_positives"].append((str(file_path), score))
                if abs(score - threshold) < margin:
                    results["borderline"].append((str(file_path), 0, score))

                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(neg_files)} negative samples...")

            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")

    return results


def print_report(results: dict, threshold: float):
    """Print analysis report."""
    print(f"\n{'='*60}")
    print("PRUNING ANALYSIS REPORT")
    print(f"{'='*60}")
    print(f"Threshold: {threshold}")
    print(f"Total samples analyzed: {len(results['all_results'])}")

    # False positives (sorted by severity - highest score first)
    fps = sorted(results["false_positives"], key=lambda x: x[1], reverse=True)
    print(f"\nFALSE POSITIVES (negatives scored > {threshold}): {len(fps)}")
    print("-" * 60)
    for path, score in fps[:20]:
        name = Path(path).name
        print(f"  {score:.4f}  {name}")
    if len(fps) > 20:
        print(f"  ... and {len(fps) - 20} more")

    # Weak positives (sorted by severity - lowest score first)
    wps = sorted(results["weak_positives"], key=lambda x: x[1])
    print(f"\nWEAK POSITIVES (positives scored < {threshold}): {len(wps)}")
    print("-" * 60)
    for path, score in wps[:20]:
        name = Path(path).name
        print(f"  {score:.4f}  {name}")
    if len(wps) > 20:
        print(f"  ... and {len(wps) - 20} more")

    # Borderline
    bls = sorted(results["borderline"], key=lambda x: abs(x[2] - threshold))
    print(f"\nBORDERLINE SAMPLES (within ±0.15 of threshold): {len(bls)}")
    print("-" * 60)
    for path, label, score in bls[:20]:
        name = Path(path).name
        label_str = "POS" if label == 1 else "NEG"
        print(f"  {score:.4f}  [{label_str}]  {name}")
    if len(bls) > 20:
        print(f"  ... and {len(bls) - 20} more")

    # Summary
    total = len(results["all_results"])
    flagged = len(fps) + len(wps)
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total samples:    {total}")
    print(f"  False positives:  {len(fps)}")
    print(f"  Weak positives:   {len(wps)}")
    print(f"  Borderline:       {len(bls)}")
    print(f"  Total flagged:    {flagged} ({100*flagged/max(1,total):.1f}%)")
    print(f"{'='*60}")


def interactive_review(results: dict, threshold: float):
    """Interactive review mode - play and classify each flagged sample."""
    try:
        import sounddevice as sd
    except ImportError:
        print("Error: sounddevice not installed. Install with: pip install sounddevice")
        return

    # Combine all flagged samples
    flagged = []
    for path, score in results["false_positives"]:
        flagged.append((path, 0, score, "FALSE_POSITIVE"))
    for path, score in results["weak_positives"]:
        flagged.append((path, 1, score, "WEAK_POSITIVE"))

    # Sort by severity
    flagged.sort(key=lambda x: abs(x[2] - threshold))

    if not flagged:
        print("No flagged samples to review!")
        return

    print(f"\nInteractive Review: {len(flagged)} flagged samples")
    print("Controls: [k]eep  [d]elete (quarantine)  [s]kip  [q]uit")
    print("-" * 60)

    to_quarantine = []

    for i, (path, label, score, flag_type) in enumerate(flagged):
        name = Path(path).name
        label_str = "POS" if label == 1 else "NEG"
        print(f"\n[{i+1}/{len(flagged)}] {name}")
        print(f"  Label: {label_str}  Score: {score:.4f}  Type: {flag_type}")

        # Play audio
        try:
            audio, sr = load_audio(path)
            sd.play(audio, sr)
            sd.wait()
        except Exception as e:
            print(f"  Error playing: {e}")

        # Get user input
        while True:
            choice = input("  Action [k/d/s/q]: ").strip().lower()
            if choice in ['k', 'd', 's', 'q']:
                break
            print("  Invalid choice. Use k(eep), d(elete), s(kip), or q(uit)")

        if choice == 'q':
            break
        elif choice == 'd':
            to_quarantine.append(path)
            print(f"  -> Marked for quarantine")
        elif choice == 'k':
            print(f"  -> Keeping")

    # Quarantine files
    if to_quarantine:
        print(f"\nQuarantining {len(to_quarantine)} files...")
        quarantine_files(to_quarantine)
    else:
        print("\nNo files to quarantine.")


def quarantine_files(file_paths: List[str]):
    """Move files to quarantine directory (recoverable, not deleted)."""
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    # Create subdirs to preserve origin
    pos_quarantine = QUARANTINE_DIR / "1"
    neg_quarantine = QUARANTINE_DIR / "0"
    pos_quarantine.mkdir(exist_ok=True)
    neg_quarantine.mkdir(exist_ok=True)

    moved = 0
    for path_str in file_paths:
        path = Path(path_str)
        if not path.exists():
            continue

        # Determine if positive or negative
        if str(POSITIVE_DIR) in str(path):
            dest = pos_quarantine / path.name
        else:
            dest = neg_quarantine / path.name

        shutil.move(str(path), str(dest))
        moved += 1

    print(f"Moved {moved} files to {QUARANTINE_DIR}")
    print("To recover, move files back from data/quarantine/ to data/1/ or data/0/")


def auto_quarantine(results: dict, threshold: float):
    """Automatically quarantine all flagged samples."""
    to_quarantine = []

    for path, score in results["false_positives"]:
        to_quarantine.append(path)
    for path, score in results["weak_positives"]:
        to_quarantine.append(path)

    if not to_quarantine:
        print("No samples to quarantine!")
        return

    print(f"\nAuto-quarantining {len(to_quarantine)} flagged samples...")
    quarantine_files(to_quarantine)


def export_report(results: dict, output_path: str):
    """Export full analysis to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "true_label", "predicted_score", "flag"])

        for path, label, score in results["all_results"]:
            flag = ""
            if label == 0 and any(p == path for p, _ in results["false_positives"]):
                flag = "FALSE_POSITIVE"
            elif label == 1 and any(p == path for p, _ in results["weak_positives"]):
                flag = "WEAK_POSITIVE"

            writer.writerow([path, label, f"{score:.6f}", flag])

    print(f"Report exported to {output_path}")
    print(f"Total rows: {len(results['all_results'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Post-training dataset pruning tool for Hey Ditto"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to model (default: models/HeyDittoNet-v3.keras)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification threshold (default: 0.5)"
    )
    parser.add_argument(
        "--margin", type=float, default=0.15,
        help="Margin for borderline detection (default: 0.15)"
    )
    parser.add_argument(
        "--listen", action="store_true",
        help="Interactive review mode - play and classify each flagged sample"
    )
    parser.add_argument(
        "--delete", action="store_true",
        help="Auto-quarantine all flagged samples"
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export full analysis to CSV file"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("Hey Ditto - Dataset Pruning Tool")
    print("="*60)

    model = load_model(args.model)
    results = analyze_samples(model, threshold=args.threshold, margin=args.margin)

    print_report(results, args.threshold)

    if args.export:
        export_report(results, args.export)

    if args.listen:
        interactive_review(results, args.threshold)
    elif args.delete:
        auto_quarantine(results, args.threshold)
    else:
        flagged = len(results["false_positives"]) + len(results["weak_positives"])
        if flagged > 0:
            print(f"\nTo review flagged samples: python src/prune.py --listen")
            print(f"To auto-quarantine:        python src/prune.py --delete")
            print(f"To export CSV:             python src/prune.py --export report.csv")


if __name__ == "__main__":
    main()
