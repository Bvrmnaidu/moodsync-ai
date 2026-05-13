"""
MoodSyncAI - Generative Summary
Hybrid approach: rule-based template + flan-t5 explanation generation.
More reliable than asking flan-t5-base to follow complex instructions directly.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fusion import fuse


_TOKENIZER = None
_MODEL = None


def get_model():
    global _TOKENIZER, _MODEL
    if _MODEL is None:
        print("Loading generative model flan-t5-base...")
        _TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-base")
        _MODEL = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return _TOKENIZER, _MODEL


def _llm_explain(question: str) -> str:
    """Helper: ask flan-t5 a simple, single-focus question."""
    tokenizer, model = get_model()
    inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def generate_summary(fusion_result: dict) -> str:
    """
    Two-stage summary:
      1. Rule-based opening sentence stating the facts.
      2. LLM-generated interpretation sentence about what it likely means.
    """

    visual = fusion_result["visual_emotion"]
    visual_conf = int(fusion_result["visual_confidence"] * 100)
    textual = fusion_result["textual_sentiment"]
    textual_conf = int(fusion_result["textual_confidence"] * 100)
    status = fusion_result["fusion_status"]

    # ---- Sentence 1: factual statement (rule-based) ----
    sentence_1 = (
        f"The visual analysis detected '{visual}' ({visual_conf}% confidence) "
        f"while the verbal sentiment was classified as '{textual}' ({textual_conf}% confidence)."
    )

    # ---- Sentence 2: status statement (rule-based) ----
    if status == "ALIGNED":
        sentence_2 = (
            f"Both modalities are aligned, suggesting the person's expressed emotion "
            f"is consistent with what they communicated verbally."
        )
        llm_question = (
            f"Explain in one sentence why a person showing '{visual}' on their face "
            f"and speaking with '{textual}' sentiment is consistent."
        )
    elif status == "PARTIAL MISMATCH":
        sentence_2 = (
            f"There is a partial mismatch between facial and verbal signals — "
            f"one channel appears neutral while the other is more emotionally charged."
        )
        llm_question = (
            f"Explain in one sentence what it means when a person's face shows '{visual}' "
            f"but their words sound '{textual}'."
        )
    else:  # MISMATCH DETECTED
        sentence_2 = (
            f"This is a clear incongruence: the facial cues suggest one emotional state "
            f"while the verbal content suggests another."
        )
        llm_question = (
            f"Why might a person say something '{textual}' while their face shows '{visual}'? "
            f"Answer in one sentence."
        )

    # ---- Sentence 3: LLM interpretation ----
    sentence_3 = _llm_explain(llm_question)

    return f"{sentence_1} {sentence_2} {sentence_3}"


def main():
    image_path = "test_face.jpg"
    text = "No, I think the project is going really well."

    print(f"\nRunning full MoodSyncAI pipeline on:")
    print(f"  Image: {image_path}")
    print(f"  Text:  {text}\n")

    fusion_result = fuse(image_path, text)

    print("\n=== FUSION RESULT ===")
    print(f"  Visual    : {fusion_result['visual_emotion']} ({fusion_result['visual_confidence']*100:.1f}%)")
    print(f"  Textual   : {fusion_result['textual_sentiment']} ({fusion_result['textual_confidence']*100:.1f}%)")
    print(f"  Status    : {fusion_result['fusion_status']}")

    print("\n=== GENERATIVE SUMMARY ===")
    summary = generate_summary(fusion_result)
    print(summary)


if __name__ == "__main__":
    main()
