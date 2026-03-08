"""
Training script for Hey Ditto wake word detection.

Usage:
    python train.py
    python train.py --epochs 100 --batch-size 64
    python train.py --resume models/HeyDittoNet-v3.keras
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

from model import create_heydittonet_v3, compile_model, get_callbacks, convert_to_tflite

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"


def load_dataset(data_dir: Path = DATA_DIR) -> tuple:
    """Load the prepared dataset from numpy files."""
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


def plot_training_history(history, save_path: Path = None) -> None:
    """Plot training history (loss and accuracy)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path: Path = None) -> None:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    classes = ['Negative (0)', 'Positive (1)']
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True Label',
        xlabel='Predicted Label',
        title='Confusion Matrix'
    )

    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def train(
    epochs: int = 150,
    batch_size: int = 32,
    validation_split: float = 0.15,
    test_split: float = 0.1,
    learning_rate: float = 0.001,
    resume_path: str = None,
    save_tflite: bool = True
) -> tuple:
    """Train the HeyDittoNet model."""
    print("\n" + "="*60)
    print("Hey Ditto - Model Training")
    print("="*60)

    print("\nLoading dataset...")
    x_data, y_data = load_dataset()

    print(f"Dataset shape: X={x_data.shape}, Y={y_data.shape}")
    n_positive = np.sum(y_data == 1)
    n_negative = np.sum(y_data == 0)
    print(f"Positive samples: {n_positive}")
    print(f"Negative samples: {n_negative}")

    total = n_positive + n_negative
    weight_for_0 = (1 / n_negative) * (total / 2.0) if n_negative > 0 else 1.0
    weight_for_1 = (1 / n_positive) * (total / 2.0) if n_positive > 0 else 1.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"Class weights: {class_weight}")

    print("\nSplitting data...")
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x_data, y_data, test_size=test_split, stratify=y_data, random_state=42
    )

    print(f"Training+Validation: {len(y_train_val)} samples")
    print(f"Test: {len(y_test)} samples")

    input_shape = x_data.shape[1:]
    print(f"\nInput shape: {input_shape}")

    if resume_path and Path(resume_path).exists():
        print(f"Resuming from {resume_path}")
        model = keras.models.load_model(resume_path)
    else:
        print("Creating new HeyDittoNet v3 model...")
        model = create_heydittonet_v3(input_shape)
        model = compile_model(model, learning_rate=learning_rate)

    model.summary()

    callbacks = get_callbacks(patience=15)

    print(f"\nTraining for up to {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Validation split: {validation_split}")
    print("-"*60)

    history = model.fit(
        x_train_val, y_train_val,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        class_weight=class_weight,
        shuffle=True,
        verbose=1
    )

    print("\n" + "="*60)
    print("Evaluation on Test Set")
    print("="*60)

    test_loss, test_acc, test_precision, test_recall = model.evaluate(
        x_test, y_test, verbose=0
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")

    y_pred_prob = model.predict(x_test, verbose=0)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "HeyDittoNet-v3.keras"
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}")

    if save_tflite:
        tflite_path = MODELS_DIR / "model.tflite"
        convert_to_tflite(model, str(tflite_path))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    history_plot_path = MODELS_DIR / f"training_history_{timestamp}.png"
    plot_training_history(history, save_path=history_plot_path)

    cm_plot_path = MODELS_DIR / f"confusion_matrix_{timestamp}.png"
    plot_confusion_matrix(y_test, y_pred, save_path=cm_plot_path)

    return model, history


def main():
    parser = argparse.ArgumentParser(
        description="Train HeyDittoNet for wake word detection"
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-tflite", action="store_true")

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        test_split=args.test_split,
        learning_rate=args.learning_rate,
        resume_path=args.resume,
        save_tflite=not args.no_tflite
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
