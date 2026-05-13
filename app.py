"""
MoodSyncAI - Streamlit UI
Multimodal sentiment & emotion analyser with CNN + Transformer + Fusion + GenAI.
Extended: webcam input, ViT attention rollout, audio transcription via Whisper.
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
from attention_rollout import compute_attention_rollout
from audio_transcribe import transcribe_audio


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="MoodSyncAI - Multimodal Emotion Analyser",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# Cached model loading
# ============================================================
@st.cache_resource(show_spinner="Loading AI models... (~1 minute first time)")
def warmup_models():
    detect_emotion("test_face.jpg")
    analyze_sentiment("warmup")
    return True


# ============================================================
# Session state for transcribed text
# ============================================================
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = "No, I think the project is going really well."


# ============================================================
# Header
# ============================================================
st.title("🧠 MoodSyncAI")
st.markdown("**Multimodal Sentiment & Emotion Analyser**")
st.caption("CNN (ViT) for facial emotion · Transformer (RoBERTa) for text sentiment · Whisper for audio · Fusion layer · Generative explanation · Attention rollout")
st.divider()


# ============================================================
# Input section
# ============================================================
col_input_l, col_input_r = st.columns(2)

with col_input_l:
    st.subheader("📷 Visual Input")

    input_mode = st.radio(
        "Visual input mode",
        ["Upload image", "Take photo (webcam)"],
        horizontal=True,
    )

    uploaded_image = None

    if input_mode == "Upload image":
        uploaded_image = st.file_uploader(
            "Upload a face image",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded image", use_container_width=True)
    else:
        uploaded_image = st.camera_input(
            "Click 'Take Photo' below to capture from webcam",
        )
        if uploaded_image is not None:
            st.success("✅ Photo captured")

with col_input_r:
    st.subheader("💬 Verbal Input")

    text_mode = st.radio(
        "Verbal input mode",
        ["Type text", "Record audio (Whisper)"],
        horizontal=True,
    )

    if text_mode == "Record audio (Whisper)":
        audio_bytes = st.audio_input("Click to record (then click stop)")
        if audio_bytes is not None:
            with st.spinner("Transcribing with Whisper..."):
                try:
                    transcript = transcribe_audio(audio_bytes.getvalue())
                    if transcript:
                        st.session_state.transcribed_text = transcript
                        st.success(f"✅ Transcribed: \"{transcript}\"")
                    else:
                        st.warning("Whisper returned empty transcript - try recording again with clearer audio.")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")

    user_text = st.text_area(
        "Text used for sentiment analysis",
        value=st.session_state.transcribed_text,
        height=120,
        help="If using audio mode, the transcription appears here. You can also edit it.",
    )
    st.session_state.transcribed_text = user_text

st.divider()

# ============================================================
# Analyze button
# ============================================================
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze_clicked = st.button(
        "🔍 Analyze Emotional State",
        type="primary",
        use_container_width=True
    )


# ============================================================
# Run analysis
# ============================================================
if analyze_clicked:
    if uploaded_image is None:
        st.error("⚠️ Please upload an image or take a photo first.")
        st.stop()
    if not user_text.strip():
        st.error("⚠️ Please type or record a sentence first.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_image.getbuffer())
        tmp_path = tmp.name

    try:
        with st.spinner("Running multimodal analysis..."):
            warmup_models()
            result = fuse(tmp_path, user_text)

        st.divider()
        st.header("📊 Results")

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
        # ATTENTION ROLLOUT VISUALIZATION (extended feature)
        # ----------------------------------------------------
        st.subheader("🔥 ViT Attention Rollout")
        st.caption(
            "Heatmap showing which facial regions the Vision Transformer "
            "focused on for its emotion prediction. Warm colors (red/yellow) = "
            "high attention; cool colors (blue/green) = low attention."
        )

        col_orig, col_heat = st.columns(2)

        with col_orig:
            st.markdown("**Original Image**")
            st.image(tmp_path, use_container_width=True)

        with col_heat:
            st.markdown("**Attention Heatmap**")
            with st.spinner("Computing attention rollout..."):
                try:
                    overlay, _, _ = compute_attention_rollout(tmp_path)
                    st.image(overlay, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not compute attention heatmap: {e}")

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
        else:
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
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ============================================================
# Footer
# ============================================================
st.divider()
st.caption(
    "MoodSyncAI · Data Analytics 3 Final Project · SRH Hamburg · "
    "Models: dima806/facial_emotions_image_detection · "
    "cardiffnlp/twitter-roberta-base-sentiment-latest · google/flan-t5-base · "
    "openai/whisper-tiny"
)