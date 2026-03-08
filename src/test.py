"""
Testing and evaluation script for Hey Ditto wake word detection.

Usage:
    python test.py
    python test.py --model models/HeyDittoNet-v3.keras
    python test.py --audio path/to/audio.wav
"""

import os
import sys
import argparse
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt

from utils.audio_utils import load_audio, normalize_audio
from utils.spec_utils import get_spectrogram

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"


def load_test_dataset(data_dir: Path = DATA_DIR) -> tuple:
    """Load the test dataset."""
    x_path = data_dir / "x_data.npy"
    y_path = data_dir / "y_data.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}. "
            "Please run 'python src/data/create_data.py' first."
        )

    x_data = np.load(str(x_path), allow_pickle=True)
    y_data = np.load(str(y_path), allow_pickle=True)

    return x_data, y_data


def load_model(model_path: str = None, use_tflite: bool = False):
    """Load a trained model."""
    if model_path is None:
        if use_tflite:
            model_path = str(MODELS_DIR / "model.tflite")
        else:
            model_path = str(MODELS_DIR / "HeyDittoNet-v3.keras")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}.")

    if use_tflite:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    else:
        return keras.models.load_model(model_path)


def predict_tflite(interpreter, x_data: np.ndarray) -> np.ndarray:
    """Run predictions using TFLite interpreter."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = []
    for i in range(len(x_data)):
        interpreter.set_tensor(
            input_details[0]['index'],
            np.expand_dims(x_data[i], 0).astype(np.float32)
        )
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(pred[0][0])

    return np.array(predictions)


def plot_roc_curve(y_true, y_pred_prob, save_path: Path = None):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")

    plt.show()

    return roc_auc


def plot_threshold_analysis(y_true, y_pred_prob, save_path: Path = None):
    """Analyze model performance at different thresholds."""
    thresholds = np.arange(0.1, 1.0, 0.05)
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for thresh in thresholds:
        y_pred = (y_pred_prob >= thresh).astype(int)
        metrics['accuracy'].append(accuracy_score(y_true, y_pred))
        metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_true, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_true, y_pred, zero_division=0))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, metrics['accuracy'], label='Accuracy', linewidth=2)
    plt.plot(thresholds, metrics['precision'], label='Precision', linewidth=2)
    plt.plot(thresholds, metrics['recall'], label='Recall', linewidth=2)
    plt.plot(thresholds, metrics['f1'], label='F1 Score', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Performance Metrics vs Detection Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Threshold analysis saved to {save_path}")

    plt.show()


def test_single_audio(model, audio_path: str, threshold: float = 0.5, use_tflite: bool = False) -> tuple:
    """Test model on a single audio file."""
    audio, sr = load_audio(audio_path)
    audio = normalize_audio(audio)
    spectrogram = get_spectrogram(audio)

    if use_tflite:
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(
            input_details[0]['index'],
            np.expand_dims(spectrogram, 0).astype(np.float32)
        )
        model.invoke()
        prob = float(model.get_tensor(output_details[0]['index'])[0][0])
    else:
        prob = float(model.predict(np.expand_dims(spectrogram, 0), verbose=0)[0][0])

    is_detected = prob >= threshold

    return prob, is_detected


def evaluate(
    model_path: str = None,
    use_tflite: bool = False,
    threshold: float = 0.5,
    test_split: float = 0.2
):
    """Evaluate model on test dataset."""
    print("\n" + "="*60)
    print("Hey Ditto - Model Evaluation")
    print("="*60)

    print(f"\nLoading model...")
    model = load_model(model_path, use_tflite)
    print(f"Model type: {'TFLite' if use_tflite else 'Keras'}")

    print("Loading dataset...")
    x_data, y_data = load_test_dataset()
    print(f"Total samples: {len(y_data)}")

    from sklearn.model_selection import train_test_split
    _, x_test, _, y_test = train_test_split(
        x_data, y_data, test_size=test_split, stratify=y_data, random_state=42
    )
    print(f"Test samples: {len(y_test)}")

    print("\nRunning predictions...")
    if use_tflite:
        y_pred_prob = predict_tflite(model, x_test)
    else:
        y_pred_prob = model.predict(x_test, verbose=0).flatten()

    y_pred = (y_pred_prob >= threshold).astype(int)

    print("\n" + "="*60)
    print(f"Results (threshold={threshold})")
    print("="*60)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    fpr = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    fnr = cm[1,0] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    print(f"\nFalse Positive Rate: {fpr:.4f}")
    print(f"False Negative Rate: {fnr:.4f}")

    print("\nGenerating plots...")
    roc_auc = plot_roc_curve(y_test, y_pred_prob)
    plot_threshold_analysis(y_test, y_pred_prob)

    return accuracy, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Test and evaluate Hey Ditto wake word model")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--tflite", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--audio", type=str, default=None)

    args = parser.parse_args()

    if args.audio:
        model = load_model(args.model, args.tflite)
        prob, detected = test_single_audio(model, args.audio, args.threshold, args.tflite)
        print(f"\nAudio: {args.audio}")
        print(f"Probability: {prob:.4f}")
        print(f"Detected: {'YES' if detected else 'NO'}")
    else:
        evaluate(
            model_path=args.model,
            use_tflite=args.tflite,
            threshold=args.threshold,
            test_split=args.test_split
        )


if __name__ == "__main__":
    main()
