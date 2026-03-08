"""
TTS-based sample generation for Hey Ditto wake word detection.

Usage:
    python generate_samples.py --mode edge --count 100
    python generate_samples.py --mode elevenlabs --count 50
    python generate_samples.py --mode openai --count 50
"""

import os
import sys
import argparse
import random
import time
from pathlib import Path
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import numpy as np
import soundfile as sf

load_dotenv()

SAMPLE_RATE = 16000
DATA_DIR = Path(__file__).parent.parent.parent / "data"
POSITIVE_DIR = DATA_DIR / "1"
NEGATIVE_DIR = DATA_DIR / "0"

# Wake word variations for positive samples
WAKE_WORD_VARIATIONS = [
    "Hey Ditto",
    "Hey Ditto!",
    "Hey Ditto?",
    "hey ditto",
    "HEY DITTO",
    "Hey, Ditto",
    "Hey Ditto...",
]

# Random phrases for negative samples
NEGATIVE_PHRASES = [
    "Hello there",
    "Good morning",
    "What's the weather",
    "Turn on the lights",
    "Play some music",
    "Set a timer",
    "What time is it",
    "Tell me a joke",
    "How are you",
    "Thank you",
    "Goodbye",
    "Help me",
    "Stop",
    "Start",
    "Pause",
    "Resume",
    "Next",
    "Previous",
    "Volume up",
    "Volume down",
    "Open the door",
    "Close the window",
    "Send a message",
    "Call mom",
    "Read my emails",
    "Navigate home",
    "Order food",
    "Book a ride",
]

# HARD NEGATIVES: Similar-sounding words that could trigger false positives
HARD_NEGATIVE_PHRASES = [
    # Rhymes / near-rhymes with "Ditto"
    "Hey Kiddo",
    "Hey Ditsy",
    "Hey Disco",
    "Hey Dmitri",
    "Hey Dido",
    "Hey Diego",
    "Hey Dino",
    "Hey Dita",
    "Hey Digital",
    # Contains "Ditto" but not wake word
    "Ditto",
    "Ditto that",
    "Same ditto",
    "Where's Ditto",
    "Ditto please",
    # Contains "Hey" but not wake word
    "Hey there",
    "Hey you",
    "Hey what",
    "Hey stop",
    "Hey wait",
    "Hey look",
    "Hey listen",
    # Phonetically similar patterns
    "Did though",
    "Hit oh",
    "Bit oh",
    "Little",
    "Ditty",
    "Bitter",
    "Kitten",
    "Mitten",
    "Sitting",
    # Common wake words (to distinguish from)
    "Hey Siri",
    "Hey Google",
    "Alexa",
    "Hey Alexa",
    "OK Google",
    "Hey Cortana",
    "Hey Bixby",
    "Hey Odie",
]


class EdgeTTSGenerator:
    """Generate samples using Microsoft Edge TTS (FREE, high-quality neural voices)."""

    VOICES = [
        "en-US-GuyNeural",
        "en-US-JennyNeural",
        "en-US-AriaNeural",
        "en-US-DavisNeural",
        "en-US-AmberNeural",
        "en-US-AnaNeural",
        "en-US-AndrewNeural",
        "en-US-EmmaNeural",
        "en-US-BrianNeural",
        "en-US-ChristopherNeural",
        "en-US-EricNeural",
        "en-US-MichelleNeural",
        "en-US-RogerNeural",
        "en-US-SteffanNeural",
        "en-GB-SoniaNeural",
        "en-GB-RyanNeural",
        "en-AU-NatashaNeural",
        "en-AU-WilliamNeural",
        "en-IN-NeerjaNeural",
        "en-IN-PrabhatNeural",
    ]

    def __init__(self):
        import asyncio
        self.asyncio = asyncio
        print(f"Edge TTS initialized with {len(self.VOICES)} neural voices")

    async def _generate_audio(self, text: str, voice: str, output_path: str) -> bool:
        """Generate audio using edge-tts."""
        import os
        try:
            import edge_tts

            rate = random.choice(["+0%", "+5%", "+10%", "+15%", "+20%"])

            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(output_path)

            if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
                if os.path.exists(output_path):
                    os.remove(output_path)
                return False

            return True
        except Exception as e:
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            if "No audio was received" not in str(e) and "CancelledError" not in str(type(e).__name__):
                print(f"Error generating audio: {e}")
            return False

    def _convert_and_trim(self, mp3_path: str, wav_path: str, target_duration: float = 1.5) -> bool:
        """Convert MP3 to WAV and trim/pad to target duration."""
        import os
        try:
            import librosa

            if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) < 1000:
                if os.path.exists(mp3_path):
                    os.remove(mp3_path)
                return False

            y, sr = librosa.load(mp3_path, sr=SAMPLE_RATE)

            if len(y) < SAMPLE_RATE * 0.3:
                os.remove(mp3_path)
                return False

            target_samples = int(target_duration * SAMPLE_RATE)

            if len(y) > target_samples:
                y = y[:target_samples]
            elif len(y) < target_samples:
                y = np.pad(y, (0, target_samples - len(y)))

            sf.write(wav_path, y, SAMPLE_RATE)

            os.remove(mp3_path)

            return True
        except Exception as e:
            print(f"Error converting {mp3_path}: {e}")
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)
            return False

    def _cleanup_mp3_files(self, directory: Path) -> int:
        """Remove any orphaned MP3 files from directory."""
        count = 0
        for mp3_file in directory.glob("*.mp3"):
            try:
                mp3_file.unlink()
                count += 1
            except:
                pass
        return count

    def generate_positive(self, count: int = 100) -> List[str]:
        """Generate positive 'Hey Ditto' samples."""
        POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
        generated = []
        failed = 0

        orphaned = self._cleanup_mp3_files(POSITIVE_DIR)
        if orphaned > 0:
            print(f"Cleaned up {orphaned} orphaned MP3 files")

        async def generate_all():
            nonlocal failed
            for i in range(count):
                text = random.choice(WAKE_WORD_VARIATIONS)
                voice = random.choice(self.VOICES)

                timestamp = int(time.time() * 1000) + i
                mp3_path = str(POSITIVE_DIR / f"hey_ditto_edge_{i}_{timestamp}.mp3")
                wav_path = str(POSITIVE_DIR / f"hey_ditto_edge_{i}_{timestamp}.wav")

                success = False
                try:
                    if await self._generate_audio(text, voice, mp3_path):
                        if self._convert_and_trim(mp3_path, wav_path, target_duration=1.5):
                            generated.append(wav_path)
                            success = True
                except Exception:
                    pass

                if os.path.exists(mp3_path):
                    try:
                        os.remove(mp3_path)
                    except:
                        pass

                if not success:
                    failed += 1
                    if os.path.exists(wav_path):
                        try:
                            os.remove(wav_path)
                        except:
                            pass

                await self.asyncio.sleep(0.2)

                if (i + 1) % 10 == 0:
                    print(f"Generated {len(generated)}/{i+1} positive samples ({failed} failed)")

        try:
            self.asyncio.run(generate_all())
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            remaining = self._cleanup_mp3_files(POSITIVE_DIR)
            if remaining > 0:
                print(f"Cleaned up {remaining} remaining MP3 files")

        print(f"Completed: {len(generated)} samples generated, {failed} failed")
        return generated

    def generate_negative(self, count: int = 100, hard_negative_ratio: float = 0.5) -> List[str]:
        """Generate negative (non-wake-word) samples including hard negatives."""
        NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)
        generated = []
        failed = 0

        orphaned = self._cleanup_mp3_files(NEGATIVE_DIR)
        if orphaned > 0:
            print(f"Cleaned up {orphaned} orphaned MP3 files")

        hard_negative_count = int(count * hard_negative_ratio)
        regular_count = count - hard_negative_count

        all_phrases = (
            [random.choice(HARD_NEGATIVE_PHRASES) for _ in range(hard_negative_count)] +
            [random.choice(NEGATIVE_PHRASES) for _ in range(regular_count)]
        )
        random.shuffle(all_phrases)

        async def generate_all():
            nonlocal failed
            for i, text in enumerate(all_phrases):
                voice = random.choice(self.VOICES)

                neg_type = "hard" if text in HARD_NEGATIVE_PHRASES else "regular"
                timestamp = int(time.time() * 1000) + i
                mp3_path = str(NEGATIVE_DIR / f"negative_{neg_type}_edge_{i}_{timestamp}.mp3")
                wav_path = str(NEGATIVE_DIR / f"negative_{neg_type}_edge_{i}_{timestamp}.wav")

                success = False
                try:
                    if await self._generate_audio(text, voice, mp3_path):
                        if self._convert_and_trim(mp3_path, wav_path, target_duration=1.5):
                            generated.append(wav_path)
                            success = True
                except Exception:
                    pass

                if os.path.exists(mp3_path):
                    try:
                        os.remove(mp3_path)
                    except:
                        pass

                if not success:
                    failed += 1
                    if os.path.exists(wav_path):
                        try:
                            os.remove(wav_path)
                        except:
                            pass

                await self.asyncio.sleep(0.2)

                if (i + 1) % 10 == 0:
                    print(f"Generated {len(generated)}/{i+1} negative samples ({failed} failed)")

        try:
            self.asyncio.run(generate_all())
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            remaining = self._cleanup_mp3_files(NEGATIVE_DIR)
            if remaining > 0:
                print(f"Cleaned up {remaining} remaining MP3 files")

        print(f"Completed: {len(generated)} samples generated, {failed} failed")
        return generated


class ElevenLabsTTSGenerator:
    """Generate samples using ElevenLabs API (high quality, requires API key)."""

    def __init__(self):
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key or api_key == "your_elevenlabs_api_key_here":
            raise ValueError("ELEVENLABS_API_KEY not found in .env file.")

        from elevenlabs import ElevenLabs
        self.client = ElevenLabs(api_key=api_key)

        response = self.client.voices.get_all()
        self.voices = response.voices
        print(f"Found {len(self.voices)} ElevenLabs voices")

    def _save_audio(self, audio_bytes: bytes, file_path: str) -> bool:
        """Save audio bytes to file and ensure 1.5-second duration."""
        try:
            temp_path = file_path + ".temp"
            with open(temp_path, 'wb') as f:
                for chunk in audio_bytes:
                    f.write(chunk)

            import librosa
            y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
            target_samples = int(SAMPLE_RATE * 1.5)

            if len(y) > target_samples:
                y = y[:target_samples]
            elif len(y) < target_samples:
                y = np.pad(y, (0, target_samples - len(y)))

            sf.write(file_path, y, SAMPLE_RATE)
            os.remove(temp_path)
            return True
        except Exception as e:
            print(f"Error saving audio: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

    def generate_positive(self, count: int = 50) -> List[str]:
        """Generate positive 'Hey Ditto' samples using ElevenLabs."""
        POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
        generated = []

        for i in range(count):
            text = random.choice(WAKE_WORD_VARIATIONS)
            voice = random.choice(self.voices)

            try:
                audio = self.client.text_to_speech.convert(
                    voice_id=voice.voice_id,
                    text=text,
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128",
                )

                filename = f"hey_ditto_elevenlabs_{i}_{int(time.time())}.wav"
                file_path = str(POSITIVE_DIR / filename)

                if self._save_audio(audio, file_path):
                    generated.append(file_path)
                    if (i + 1) % 10 == 0:
                        print(f"Generated {i + 1}/{count} positive samples")

            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue

        return generated

    def generate_negative(self, count: int = 50, hard_negative_ratio: float = 0.6) -> List[str]:
        """Generate negative samples using ElevenLabs including hard negatives."""
        NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)
        generated = []

        hard_negative_count = int(count * hard_negative_ratio)
        regular_count = count - hard_negative_count

        all_phrases = (
            [random.choice(HARD_NEGATIVE_PHRASES) for _ in range(hard_negative_count)] +
            [random.choice(NEGATIVE_PHRASES) for _ in range(regular_count)]
        )
        random.shuffle(all_phrases)

        for i, text in enumerate(all_phrases):
            voice = random.choice(self.voices)

            try:
                audio = self.client.text_to_speech.convert(
                    voice_id=voice.voice_id,
                    text=text,
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128",
                )

                neg_type = "hard" if text in HARD_NEGATIVE_PHRASES else "regular"
                filename = f"negative_{neg_type}_elevenlabs_{i}_{int(time.time())}.wav"
                file_path = str(NEGATIVE_DIR / filename)

                if self._save_audio(audio, file_path):
                    generated.append(file_path)
                    if (i + 1) % 10 == 0:
                        print(f"Generated {i + 1}/{count} negative samples")

            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue

        return generated


class OpenAITTSGenerator:
    """Generate samples using OpenAI TTS API."""

    VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY not found in .env file.")

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        print(f"OpenAI TTS initialized with {len(self.VOICES)} voices")

    def _save_audio(self, response, file_path: str) -> bool:
        """Save OpenAI response to file and ensure 1.5-second duration."""
        try:
            temp_path = file_path + ".temp.mp3"
            response.stream_to_file(temp_path)

            import librosa
            y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
            target_samples = int(SAMPLE_RATE * 1.5)

            if len(y) > target_samples:
                y = y[:target_samples]
            elif len(y) < target_samples:
                y = np.pad(y, (0, target_samples - len(y)))

            sf.write(file_path, y, SAMPLE_RATE)
            os.remove(temp_path)
            return True
        except Exception as e:
            print(f"Error saving audio: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

    def generate_positive(self, count: int = 50) -> List[str]:
        """Generate positive 'Hey Ditto' samples using OpenAI TTS."""
        POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
        generated = []

        for i in range(count):
            text = random.choice(WAKE_WORD_VARIATIONS)
            voice = random.choice(self.VOICES)

            try:
                response = self.client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=text,
                )

                filename = f"hey_ditto_openai_{i}_{int(time.time())}.wav"
                file_path = str(POSITIVE_DIR / filename)

                if self._save_audio(response, file_path):
                    generated.append(file_path)
                    if (i + 1) % 10 == 0:
                        print(f"Generated {i + 1}/{count} positive samples")

            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue

        return generated

    def generate_negative(self, count: int = 50, hard_negative_ratio: float = 0.6) -> List[str]:
        """Generate negative samples using OpenAI TTS including hard negatives."""
        NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)
        generated = []

        hard_negative_count = int(count * hard_negative_ratio)
        regular_count = count - hard_negative_count

        all_phrases = (
            [random.choice(HARD_NEGATIVE_PHRASES) for _ in range(hard_negative_count)] +
            [random.choice(NEGATIVE_PHRASES) for _ in range(regular_count)]
        )
        random.shuffle(all_phrases)

        for i, text in enumerate(all_phrases):
            voice = random.choice(self.VOICES)

            try:
                response = self.client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=text,
                )

                neg_type = "hard" if text in HARD_NEGATIVE_PHRASES else "regular"
                filename = f"negative_{neg_type}_openai_{i}_{int(time.time())}.wav"
                file_path = str(NEGATIVE_DIR / filename)

                if self._save_audio(response, file_path):
                    generated.append(file_path)
                    if (i + 1) % 10 == 0:
                        print(f"Generated {i + 1}/{count} negative samples")

            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue

        return generated


def main():
    parser = argparse.ArgumentParser(
        description="Generate training samples for Hey Ditto wake word detection"
    )
    parser.add_argument("--mode", choices=["edge", "elevenlabs", "openai"], default="edge")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--negative-only", action="store_true")

    args = parser.parse_args()

    POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Hey Ditto - Sample Generator")
    print(f"{'='*50}")
    print(f"Mode: {args.mode}")
    print(f"Count per class: {args.count}")
    print(f"{'='*50}\n")

    if args.mode == "edge":
        generator = EdgeTTSGenerator()
    elif args.mode == "elevenlabs":
        generator = ElevenLabsTTSGenerator()
    elif args.mode == "openai":
        generator = OpenAITTSGenerator()

    if not args.negative_only:
        print("\nGenerating positive samples ('Hey Ditto')...")
        positive_files = generator.generate_positive(args.count)
        print(f"Generated {len(positive_files)} positive samples")

    if not args.positive_only:
        print("\nGenerating negative samples...")
        negative_files = generator.generate_negative(args.count)
        print(f"Generated {len(negative_files)} negative samples")

    print("\nDone!")


if __name__ == "__main__":
    main()
