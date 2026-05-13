"""
MoodSyncAI - Audio Transcription
Uses Whisper (tiny model) to transcribe speech to text.
Output flows into the existing RoBERTa text sentiment pipeline.
"""

import whisper
import tempfile
import os


_MODEL = None


def get_whisper():
    """Load Whisper tiny model. Cached."""
    global _MODEL
    if _MODEL is None:
        print("Loading Whisper tiny model... (first run downloads ~39MB)")
        _MODEL = whisper.load_model("tiny")
    return _MODEL


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio bytes (any common format) to text.

    Returns:
        Transcribed text as a single string.
    """

    # Whisper needs a file path, so write to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        model = get_whisper()
        result = model.transcribe(tmp_path, fp16=False)
        text = result["text"].strip()
        return text
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    """Standalone test - requires a test audio file."""
    test_file = "test_audio.wav"
    if not os.path.exists(test_file):
        print(f"No test file at {test_file} - skipping standalone test.")
        print("Will be tested in Streamlit instead.")
        return

    with open(test_file, "rb") as f:
        audio_bytes = f.read()

    text = transcribe_audio(audio_bytes)
    print(f"Transcription: \"{text}\"")


if __name__ == "__main__":
    main()