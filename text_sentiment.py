"""
MoodSyncAI - Text Sentiment Analysis
Uses pretrained Transformer (RoBERTa) from HuggingFace to detect sentiment from text.
"""

from transformers import pipeline
import sys


def analyze_sentiment(text: str):
    """Takes text, returns list of (sentiment_label, score) pairs."""

    print(f"Loading model... (first run downloads ~500MB, takes 2-3 minutes)")
    classifier = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        top_k=None
    )

    print(f"Analyzing text: \"{text}\"")
    results = classifier(text)

    return results[0]


def main():
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "No, I think the project is going really well."

    results = analyze_sentiment(text)

    print("\n=== TEXT SENTIMENT RESULTS ===")
    for r in results:
        confidence_pct = r["score"] * 100
        print(f"  {r['label']:10s} : {confidence_pct:5.1f}%")

    top_sentiment = max(results, key=lambda r: r["score"])
    print(f"\nTop sentiment: {top_sentiment['label']} ({top_sentiment['score']*100:.1f}% confidence)")


if __name__ == "__main__":
    main()
