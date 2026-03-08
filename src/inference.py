"""
Real-time microphone inference for Hey Ditto wake word detection.

Usage:
    python inference.py
    python inference.py --sensitivity 0.95
    python inference.py --tflite
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import sounddevice as sd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from utils.audio_utils import normalize_audio
from utils.spec_utils import get_spectrogram

SAMPLE_RATE = 16000
BUFFER_SIZE = int(SAMPLE_RATE * 1.5)  # 1.5 seconds of audio
BLOCK_SIZE = int(SAMPLE_RATE / 4)  # Process every 0.25 seconds
MODELS_DIR = Path(__file__).parent.parent / "models"


class HeyDittoDetector:
    """Real-time wake word detector for 'Hey Ditto'."""

    def __init__(
        self,
        model_path: str = None,
        tflite: bool = True,
        sensitivity: float = 0.95,
        cooldown: float = 2.0
    ):
        self.tflite = tflite
        self.sensitivity = sensitivity
        self.cooldown = cooldown

        self.buffer = []
        self.frames_since_detection = 0
        self.last_detection_time = 0

        self._load_model(model_path)

        print(f"\n{'='*50}")
        print("Hey Ditto - Wake Word Detector")
        print(f"{'='*50}")
        print(f"Model type: {'TFLite' if tflite else 'Keras'}")
        print(f"Sensitivity: {sensitivity}")
        print(f"Cooldown: {cooldown}s")
        print(f"{'='*50}\n")

    def _load_model(self, model_path: str = None):
        """Load the model (Keras or TFLite)."""
        if self.tflite:
            if model_path is None:
                model_path = str(MODELS_DIR / "model.tflite")

            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"TFLite model not found at {model_path}. "
                    "Run training first or use --no-tflite flag."
                )

            with open(model_path, 'rb') as f:
                self.model = f.read()

            self.interpreter = tf.lite.Interpreter(model_content=self.model)
            self.interpreter.allocate_tensors()
            self.input_index = self.interpreter.get_input_details()[0]["index"]
            self.output_index = self.interpreter.get_output_details()[0]["index"]

            print(f"Loaded TFLite model from {model_path}")
        else:
            if model_path is None:
                model_path = str(MODELS_DIR / "HeyDittoNet-v3.keras")

            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"Keras model not found at {model_path}. "
                    "Run training first."
                )

            self.model = keras.models.load_model(model_path)
            print(f"Loaded Keras model from {model_path}")

    def _predict(self, spectrogram: np.ndarray) -> float:
        """Run prediction on a spectrogram."""
        if self.tflite:
            self.interpreter.set_tensor(
                self.input_index,
                np.expand_dims(spectrogram, 0).astype(np.float32)
            )
            self.interpreter.invoke()
            pred = self.interpreter.get_tensor(self.output_index)
            return float(pred[0][0])
        else:
            pred = self.model(np.expand_dims(spectrogram, 0), training=False)
            K.clear_session()
            return float(pred[0][0])

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream processing."""
        if status:
            print(f"Audio status: {status}")

        for sample in indata.flatten():
            self.buffer.append(sample)

        if len(self.buffer) >= BUFFER_SIZE and self.frames_since_detection == 0:
            self.frames_since_detection += frames

            self.buffer = self.buffer[-BUFFER_SIZE:]

            audio = normalize_audio(np.array(self.buffer))
            spectrogram = get_spectrogram(audio)
            confidence = self._predict(spectrogram)

            current_time = time.time()
            if confidence >= self.sensitivity:
                if current_time - self.last_detection_time > self.cooldown:
                    print(f"\nDETECTED: 'Hey Ditto' (confidence: {confidence*100:.1f}%)")
                    self.last_detection_time = current_time
                    self.on_detection(confidence)

        if self.frames_since_detection > 0:
            self.frames_since_detection += frames
            if self.frames_since_detection >= BLOCK_SIZE:
                self.frames_since_detection = 0

    def on_detection(self, confidence: float):
        """Called when wake word is detected. Override for custom behavior."""
        pass

    def listen(self, device: Optional[int] = None):
        """Start listening for the wake word."""
        if device is None:
            device = sd.default.device[0]

        print(f"Listening on device: {sd.query_devices(device)['name']}")
        print("Say 'Hey Ditto' to test detection...")
        print("Press Ctrl+C to stop.\n")

        try:
            with sd.InputStream(
                device=device,
                samplerate=SAMPLE_RATE,
                dtype='float32',
                latency='low',
                channels=1,
                callback=self.audio_callback,
                blocksize=BLOCK_SIZE
            ):
                print("Idle... (waiting for wake word)")
                while True:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nStopping...")
        except Exception as e:
            print(f"Error: {e}")


def list_audio_devices():
    """Print list of available audio devices."""
    print("\nAvailable Audio Devices:")
    print("-" * 60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            default = " (default)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {device['name']}{default}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Real-time wake word detection for 'Hey Ditto'")
    parser.add_argument("--sensitivity", type=float, default=0.95)
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--no-tflite", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    detector = HeyDittoDetector(
        model_path=args.model,
        tflite=not args.no_tflite,
        sensitivity=args.sensitivity,
        cooldown=args.cooldown
    )

    detector.listen(device=args.device)


if __name__ == "__main__":
    main()
