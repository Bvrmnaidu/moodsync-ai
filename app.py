"""
MoodSyncAI - Streamlit UI
Multimodal sentiment & emotion analyser with CNN + Transformer + Fusion + GenAI.
"""

import streamlit as st
import pandas as pd
from PIL import Image
import tempfile
import os

from cnn_emotion import detect_emotion
from text_sentiment import analyze_sentiment
from fusion import fuse, EMOTION_TO_SENTIMENT
from summary import generate_summary


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="MoodSyncAI - Multimodal Emotion Analyser",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# Cached model loading - prevents reload on every interaction
# ============================================================
@st.cache_resource(show_spinner="Loading AI models... (~1 minute first time)")
def warmup_models():
    """Pre-load all three models into memory once."""
    detect_emotion("test_face.jpg")  # warms CNN
    analyze_sentiment("warmup")       # warms RoBERTa
    return True


# ============================================================
# Header
# ============================================================
st.title("🧠 MoodSyncAI")
st.markdown("**Multimodal Sentiment & Emotion Analyser**")
st.caption("CNN (ViT) for facial emotion · Transformer (RoBERTa) for text sentiment · Fusion layer · Generative explanation")
st.divider()


# ============================================================
# Input section - 2 columns
# ============================================================
col_input_l, col_input_r = st.columns(2)

with col_input_l:
    st.subheader("📷 Visual Input")
    uploaded_image = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear photo of a person's face"
    )
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded image", use_container_width=True)

with col_input_r:
    st.subheader("💬 Verbal Input")
    user_text = st.text_area(
        "Type what the person said",
        value="No, I think the project is going really well.",
        height=120,
        help="Type the sentence the person spoke"
    )

st.divider()

# ============================================================
# Analyze button - centered
# ============================================================
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze_clicked = st.button(
        "🔍 Analyze Emotional State",
        type="primary",
        use_container_width=True
    )


# ============================================================
# Run analysis when button clicked
# ============================================================
if analyze_clicked:
    if uploaded_image is None:
        st.error("⚠️ Please upload an image first.")
        st.stop()
    if not user_text.strip():
        st.error("⚠️ Please type a sentence first.")
        st.stop()

    # Save uploaded image to a temp file so fuse() can read it from path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_image.getbuffer())
        tmp_path = tmp.name

    try:
        with st.spinner("Running multimodal analysis..."):
            warmup_models()
            result = fuse(tmp_path, user_text)

        st.divider()
        st.header("📊 Results")

        # ----------------------------------------------------
        # Two charts side-by-side
        # ----------------------------------------------------
        col_chart_l, col_chart_r = st.columns(2)

        with col_chart_l:
            st.subheader("Visual Emotion (CNN/ViT)")
            visual_df = pd.DataFrame({
                "Emotion": [r["label"] for r in result["visual_all"]],
                "Confidence": [r["score"] * 100 for r in result["visual_all"]],
            }).sort_values("Confidence", ascending=False)
            st.bar_chart(visual_df.set_index("Emotion"), height=300)
            st.metric(
                "Top Visual Emotion",
                result["visual_emotion"].capitalize(),
                f"{result['visual_confidence']*100:.1f}% confidence"
            )

        with col_chart_r:
            st.subheader("Text Sentiment (RoBERTa)")
            textual_df = pd.DataFrame({
                "Sentiment": [r["label"] for r in result["textual_all"]],
                "Confidence": [r["score"] * 100 for r in result["textual_all"]],
            }).sort_values("Confidence", ascending=False)
            st.bar_chart(textual_df.set_index("Sentiment"), height=300)
            st.metric(
                "Top Text Sentiment",
                result["textual_sentiment"].capitalize(),
                f"{result['textual_confidence']*100:.1f}% confidence"
            )

        st.divider()

        # ----------------------------------------------------
        # Fusion status badge
        # ----------------------------------------------------
        st.subheader("🔗 Fusion Analysis")

        status = result["fusion_status"]
        agreement = result["agreement_score"] * 100

        if status == "ALIGNED":
            st.success(f"✅ **ALIGNED** — Agreement score: {agreement:.1f}%")
            st.caption("Visual and verbal signals are consistent with each other.")
        elif status == "PARTIAL MISMATCH":
            st.warning(f"⚠️ **PARTIAL MISMATCH** — Agreement score: {agreement:.1f}%")
            st.caption("One modality is neutral while the other is emotionally charged.")
        else:  # MISMATCH DETECTED
            st.error(f"🚨 **MISMATCH DETECTED** — Agreement score: {agreement:.1f}%")
            st.caption("Strong incongruence between facial cues and verbal content.")

        st.divider()

        # ----------------------------------------------------
        # Generative summary
        # ----------------------------------------------------
        st.subheader("💡 Generative Summary")
        with st.spinner("Generating natural-language explanation..."):
            summary_text = generate_summary(result)
        st.info(summary_text)

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ============================================================
# Footer
# ============================================================
st.divider()
st.caption(
    "MoodSyncAI · Data Analytics 3 Final Project · SRH Hamburg · "
    "Models: dima806/facial_emotions_image_detection · "
    "cardiffnlp/twitter-roberta-base-sentiment-latest · google/flan-t5-base"
)
