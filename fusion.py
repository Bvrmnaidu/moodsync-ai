"""
MoodSyncAI - Fusion Layer
Combines CNN facial emotion + RoBERTa text sentiment into a unified assessment.
Detects ALIGNED vs MISMATCH between visual and verbal signals.
"""

from cnn_emotion import detect_emotion
from text_sentiment import analyze_sentiment


# Map each facial emotion to its sentiment polarity
EMOTION_TO_SENTIMENT = {
    "happy":    "positive",
    "surprise": "positive",
    "neutral":  "neutral",
    "sad":      "negative",
    "angry":    "negative",
    "fear":     "negative",
    "disgust":  "negative",
}


def fuse(image_path: str, text: str, mismatch_threshold: float = 0.5):
    """
    Run both models, compare results, return a fusion verdict.

    Returns:
        dict with keys:
            visual_emotion, visual_confidence,
            textual_sentiment, textual_confidence,
            visual_polarity (positive/neutral/negative),
            fusion_status (ALIGNED / MISMATCH DETECTED),
            agreement_score (0-1)
    """

    # --- Run CNN ---
    visual_results = detect_emotion(image_path)
    top_visual = visual_results[0]
    visual_emotion = top_visual["label"].lower()
    visual_confidence = top_visual["score"]

    # --- Run text sentiment ---
    textual_results = analyze_sentiment(text)
    top_textual = max(textual_results, key=lambda r: r["score"])
    textual_sentiment = top_textual["label"].lower()
    textual_confidence = top_textual["score"]

    # --- Map visual emotion → polarity ---
    visual_polarity = EMOTION_TO_SENTIMENT.get(visual_emotion, "neutral")

    # --- Compute alignment ---
    if visual_polarity == textual_sentiment:
        fusion_status = "ALIGNED"
        agreement_score = (visual_confidence + textual_confidence) / 2
    else:
        # Special case: one is neutral → soft mismatch, not hard mismatch
        if "neutral" in (visual_polarity, textual_sentiment):
            fusion_status = "PARTIAL MISMATCH"
        else:
            fusion_status = "MISMATCH DETECTED"
        # Lower agreement when there is disagreement
        agreement_score = 1 - ((visual_confidence + textual_confidence) / 2)

    return {
        "visual_emotion": visual_emotion,
        "visual_confidence": visual_confidence,
        "visual_polarity": visual_polarity,
        "textual_sentiment": textual_sentiment,
        "textual_confidence": textual_confidence,
        "fusion_status": fusion_status,
        "agreement_score": agreement_score,
        "visual_all": visual_results,
        "textual_all": textual_results,
    }


def main():
    # Default demo: assignment example (sad face + positive words = mismatch)
    image_path = "test_face.jpg"
    text = "No, I think the project is going really well."

    print(f"\nRunning fusion on:")
    print(f"  Image: {image_path}")
    print(f"  Text:  {text}\n")

    result = fuse(image_path, text)

    print("\n=== FUSION RESULT ===")
    print(f"  Visual emotion    : {result['visual_emotion']} ({result['visual_confidence']*100:.1f}%)")
    print(f"  Visual polarity   : {result['visual_polarity']}")
    print(f"  Textual sentiment : {result['textual_sentiment']} ({result['textual_confidence']*100:.1f}%)")
    print(f"  Fusion status     : {result['fusion_status']}")
    print(f"  Agreement score   : {result['agreement_score']*100:.1f}%")


if __name__ == "__main__":
    main()
