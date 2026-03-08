"""
Dataset Statistics for Hey Ditto Wake Word Detection.

Shows comprehensive breakdown of:
- Raw sample counts by source
- Estimated counts after augmentation
- Class balance ratios
- Recommendations

Usage:
    python src/data/dataset_stats.py
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent.parent / "data"
POSITIVE_DIR = DATA_DIR / "1"
NEGATIVE_DIR = DATA_DIR / "0"
BACKGROUND_DIR = DATA_DIR / "background_noise"

POSITIVE_AUGMENTATION_MULTIPLIER = 9
NEGATIVE_AUGMENTATION_MULTIPLIER = 3


def count_files_by_pattern(directory: Path) -> dict:
    """Count files matching different patterns in a directory."""
    if not directory.exists():
        return {}

    patterns = {
        "edge": "edge",
        "elevenlabs": "elevenlabs",
        "openai": "openai",
        "background": "bg_",
        "mixed": "mixed_bg",
        "synthetic": "synthetic",
        "hard_negative": "hard",
        "regular_negative": "regular",
    }

    counts = defaultdict(int)
    all_files = list(directory.glob("*.wav"))
    counts["total"] = len(all_files)

    for f in all_files:
        name = f.name.lower()
        matched = False
        for pattern_name, pattern in patterns.items():
            if pattern in name:
                counts[pattern_name] += 1
                matched = True
                break
        if not matched:
            counts["other"] += 1

    return dict(counts)


def get_background_stats() -> dict:
    """Get statistics for background noise directory."""
    stats = {"total": 0, "esc50": 0, "synthetic": 0}

    if BACKGROUND_DIR.exists():
        esc50_dir = BACKGROUND_DIR / "esc50"
        synthetic_dir = BACKGROUND_DIR / "synthetic"

        if esc50_dir.exists():
            stats["esc50"] = len(list(esc50_dir.glob("*.wav")))
        if synthetic_dir.exists():
            stats["synthetic"] = len(list(synthetic_dir.glob("*.wav")))

        stats["total"] = stats["esc50"] + stats["synthetic"]

    return stats


def check_existing_dataset() -> dict:
    """Check if x_data.npy and y_data.npy exist and get their stats."""
    x_path = DATA_DIR / "x_data.npy"
    y_path = DATA_DIR / "y_data.npy"

    stats = {"exists": False}

    if x_path.exists() and y_path.exists():
        try:
            x_data = np.load(str(x_path))
            y_data = np.load(str(y_path))

            stats["exists"] = True
            stats["total_samples"] = len(y_data)
            stats["positive_samples"] = int(np.sum(y_data == 1))
            stats["negative_samples"] = int(np.sum(y_data == 0))
            stats["spectrogram_shape"] = x_data.shape[1:]
            stats["ratio"] = round(stats["negative_samples"] / max(1, stats["positive_samples"]), 2)
        except Exception as e:
            stats["error"] = str(e)

    return stats


def main():
    print("\n" + "=" * 60)
    print("  HEY DITTO DATASET STATISTICS")
    print("=" * 60)

    # RAW SAMPLES
    print("\n" + "=" * 60)
    print("  RAW SAMPLES (Before Augmentation)")
    print("=" * 60)

    pos_stats = count_files_by_pattern(POSITIVE_DIR)
    print(f"\n  POSITIVE SAMPLES (data/1/): {pos_stats.get('total', 0)}")
    print(f"  {'-' * 40}")
    if pos_stats:
        for key in ["edge", "elevenlabs", "openai", "other"]:
            if pos_stats.get(key, 0) > 0:
                print(f"    * {key.capitalize():20}: {pos_stats[key]:>6}")
    else:
        print("    (no samples found)")

    neg_stats = count_files_by_pattern(NEGATIVE_DIR)
    print(f"\n  NEGATIVE SAMPLES (data/0/): {neg_stats.get('total', 0)}")
    print(f"  {'-' * 40}")
    if neg_stats:
        print(f"    Speech (TTS):")
        for key in ["edge", "elevenlabs", "openai"]:
            if neg_stats.get(key, 0) > 0:
                print(f"      * {key.capitalize():18}: {neg_stats[key]:>6}")

        print(f"    Background/Noise:")
        for key in ["background", "mixed", "synthetic"]:
            if neg_stats.get(key, 0) > 0:
                print(f"      * {key.capitalize():18}: {neg_stats[key]:>6}")

        if neg_stats.get("other", 0) > 0:
            print(f"    Other:")
            print(f"      * {'Other':18}: {neg_stats['other']:>6}")
    else:
        print("    (no samples found)")

    bg_stats = get_background_stats()
    print(f"\n  BACKGROUND NOISE LIBRARY: {bg_stats['total']}")
    print(f"  {'-' * 40}")
    print(f"    * ESC-50:              {bg_stats['esc50']:>6}")
    print(f"    * Synthetic:           {bg_stats['synthetic']:>6}")

    # CLASS BALANCE
    print("\n" + "=" * 60)
    print("  CLASS BALANCE (Raw)")
    print("=" * 60)

    pos_total = pos_stats.get("total", 0)
    neg_total = neg_stats.get("total", 0)

    if pos_total > 0:
        ratio = neg_total / pos_total
        print(f"\n  Positive:Negative = 1:{ratio:.2f}")
        print(f"  Negative:Positive = {ratio:.2f}:1")

        if ratio < 1:
            rating = "POOR - Need more negatives!"
        elif ratio < 2:
            rating = "OKAY - Could use more negatives"
        elif ratio < 5:
            rating = "GOOD - Reasonable balance"
        else:
            rating = "EXCELLENT - Production ready"

        print(f"\n  Balance Rating: {rating}")
    else:
        print("\n  No positive samples found!")

    # AUGMENTATION ESTIMATES
    print("\n" + "=" * 60)
    print("  ESTIMATED AFTER AUGMENTATION")
    print("=" * 60)

    est_positive = pos_total * POSITIVE_AUGMENTATION_MULTIPLIER
    est_negative = neg_total * NEGATIVE_AUGMENTATION_MULTIPLIER
    est_total = est_positive + est_negative

    print(f"\n  Augmentation multipliers:")
    print(f"    * Positive: {POSITIVE_AUGMENTATION_MULTIPLIER}x (pitch, noise, stretch, mix)")
    print(f"    * Negative: {NEGATIVE_AUGMENTATION_MULTIPLIER}x (basic augmentation)")

    print(f"\n  Estimated samples after create_data.py:")
    print(f"    * Positive:  {pos_total:>6} -> {est_positive:>8}")
    print(f"    * Negative:  {neg_total:>6} -> {est_negative:>8}")
    print(f"    * Total:     {pos_total + neg_total:>6} -> {est_total:>8}")

    if est_positive > 0:
        est_ratio = est_negative / est_positive
        print(f"\n  Estimated ratio after augmentation: 1:{est_ratio:.2f}")

    # EXISTING DATASET
    dataset_stats = check_existing_dataset()

    if dataset_stats["exists"]:
        print("\n" + "=" * 60)
        print("  CURRENT DATASET (x_data.npy / y_data.npy)")
        print("=" * 60)

        print(f"\n  Dataset file exists: Yes")
        print(f"  Spectrogram shape: {dataset_stats['spectrogram_shape']}")
        print(f"\n  Actual samples in dataset:")
        print(f"    * Positive:  {dataset_stats['positive_samples']:>8}")
        print(f"    * Negative:  {dataset_stats['negative_samples']:>8}")
        print(f"    * Total:     {dataset_stats['total_samples']:>8}")
        print(f"\n  Actual ratio: 1:{dataset_stats['ratio']}")
    else:
        print("\n" + "=" * 60)
        print("  CURRENT DATASET")
        print("=" * 60)
        print("\n  Dataset not yet created. Run: python src/data/create_data.py")

    # RECOMMENDATIONS
    print("\n" + "=" * 60)
    print("  RECOMMENDATIONS")
    print("=" * 60)

    recommendations = []

    if pos_total < 500:
        recommendations.append(f"Generate more positive samples (have {pos_total}, want 500+)")

    if neg_total < 1000:
        recommendations.append(f"Generate more negative samples (have {neg_total}, want 1000+)")

    if pos_total > 0 and neg_total / pos_total < 3:
        needed = int(pos_total * 3 - neg_total)
        if needed > 0:
            recommendations.append(f"Add {needed} more negatives for 3:1 ratio")

    if bg_stats["total"] < 500:
        recommendations.append("Download background datasets: python src/data/download_backgrounds.py")

    if not dataset_stats.get("exists"):
        recommendations.append("Create the dataset: python src/data/create_data.py")

    if recommendations:
        print("\n  To improve your dataset:")
        for rec in recommendations:
            print(f"    * {rec}")
    else:
        print("\n  Dataset looks production-ready!")
        print("    Run: python src/train.py")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
