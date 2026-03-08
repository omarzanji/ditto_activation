"""
Export HeyDittoNet model to TensorFlow.js format for desktop app integration.

Converts Keras model -> SavedModel -> tfjs graph model via
tensorflowjs.converters.convert_tf_saved_model().

Output: models/HeyDittoNet-v3-tfjs/ (model.json + weight shards)

Usage:
    python export_tfjs.py
    python export_tfjs.py --model models/HeyDittoNet-v3.keras
    python export_tfjs.py --quantize
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, str(Path(__file__).parent))

import tensorflow as tf
from tensorflow import keras

MODELS_DIR = Path(__file__).parent.parent / "models"


def export_to_tfjs(
    model_path: str = None,
    output_dir: str = None,
    quantize: bool = False
) -> Path:
    """
    Export Keras model to TensorFlow.js graph model format.

    Args:
        model_path: Path to .keras model file
        output_dir: Output directory for tfjs model
        quantize: Whether to quantize weights to uint8

    Returns:
        Path to output directory containing model.json + shards
    """
    if model_path is None:
        model_path = str(MODELS_DIR / "HeyDittoNet-v3.keras")

    if output_dir is None:
        output_dir = str(MODELS_DIR / "HeyDittoNet-v3-tfjs")

    model_path = Path(model_path)
    output_path = Path(output_dir)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run training first: python src/train.py"
        )

    print(f"\n{'='*60}")
    print("Hey Ditto - Export to TensorFlow.js")
    print(f"{'='*60}")
    print(f"Input model: {model_path}")
    print(f"Output directory: {output_path}")
    print(f"Quantize: {quantize}")

    # Step 1: Load Keras model
    print("\nLoading Keras model...")
    model = keras.models.load_model(str(model_path))
    model.summary()

    # Step 2: Save as SavedModel format (required intermediate step)
    saved_model_dir = MODELS_DIR / "temp_saved_model"
    print(f"\nConverting to SavedModel format...")
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    tf.saved_model.save(model, str(saved_model_dir))
    print(f"SavedModel saved to {saved_model_dir}")

    # Step 3: Convert SavedModel to tfjs
    print(f"\nConverting to TensorFlow.js format...")

    try:
        import tensorflowjs as tfjs

        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        quantization_dtype = 'uint8' if quantize else None

        tfjs.converters.convert_tf_saved_model(
            str(saved_model_dir),
            str(output_path),
            quantization_dtype=quantization_dtype
        )

        print(f"\nTFJS model saved to {output_path}")

    except ImportError:
        print("\nError: tensorflowjs not installed.")
        print("Install with: pip install tensorflowjs")
        # Cleanup
        if saved_model_dir.exists():
            shutil.rmtree(saved_model_dir)
        raise

    # Cleanup temp SavedModel
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
        print("Cleaned up temporary SavedModel")

    # Print output files
    print(f"\nOutput files:")
    total_size = 0
    for f in sorted(output_path.iterdir()):
        size = f.stat().st_size
        total_size += size
        print(f"  {f.name}: {size / 1024:.1f} KB")

    print(f"\nTotal model size: {total_size / 1024:.1f} KB")

    print(f"\n{'='*60}")
    print("Export complete!")
    print(f"{'='*60}")
    print(f"\nTo use in the desktop app:")
    print(f"  1. Copy {output_path}/ to the desktop app's model directory")
    print(f"  2. Load with: const model = await tf.loadGraphModel('file://path/to/model.json')")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export HeyDittoNet to TensorFlow.js format"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to Keras model (default: models/HeyDittoNet-v3.keras)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: models/HeyDittoNet-v3-tfjs)"
    )
    parser.add_argument(
        "--quantize", action="store_true",
        help="Quantize weights to uint8 for smaller model size"
    )

    args = parser.parse_args()

    export_to_tfjs(
        model_path=args.model,
        output_dir=args.output,
        quantize=args.quantize
    )


if __name__ == "__main__":
    main()
