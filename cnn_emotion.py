"""
MoodSyncAI - Facial Emotion Detection
Uses pretrained CNN (ViT-based) from HuggingFace to detect emotion from a face image.
"""

from transformers import pipeline
from PIL import Image
import sys


def detect_emotion(image_path: str):
    """Takes path to an image, returns list of (emotion, score) pairs."""

    print(f"Loading model... (first run downloads ~350MB, takes 2-3 minutes)")
    classifier = pipeline(
        "image-classification",
        model="dima806/facial_emotions_image_detection"
    )

    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    print("Running prediction...")
    results = classifier(image)

    return results


def main():
    # Use command-line argument if given, else default
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_face.jpg"

    results = detect_emotion(image_path)

    print("\n=== FACIAL EMOTION RESULTS ===")
    for r in results:
        confidence_pct = r["score"] * 100
        print(f"  {r['label']:12s} : {confidence_pct:5.1f}%")

    top_emotion = results[0]
    print(f"\nTop emotion: {top_emotion['label']} ({top_emotion['score']*100:.1f}% confidence)")


if __name__ == "__main__":
    main()