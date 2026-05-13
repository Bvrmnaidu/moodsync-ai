"""
MoodSyncAI - Attention Rollout for ViT
Visualizes which facial regions the Vision Transformer attended to when
predicting the emotion.

Reference: Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (ACL 2020)
"""

import torch
import numpy as np
from PIL import Image
import cv2
from transformers import AutoImageProcessor, AutoModelForImageClassification


_PROCESSOR = None
_MODEL = None
MODEL_NAME = "dima806/facial_emotions_image_detection"


def get_vit_model():
    """Load processor + model with output_attentions enabled. Cached."""
    global _PROCESSOR, _MODEL
    if _MODEL is None:
        print("Loading ViT model with attention output enabled...")
        _PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _MODEL = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME, output_attentions=True
        )
        _MODEL.eval()
    return _PROCESSOR, _MODEL


def compute_attention_rollout(image_path: str):
    """
    Run the ViT, capture all-layer attentions, perform attention rollout,
    and return a heatmap aligned to the original image.

    Returns:
        heatmap_overlay : np.ndarray (H, W, 3) uint8  -- original image with red heatmap blended
        top_label       : str  -- predicted emotion
        top_score       : float -- confidence 0-1
    """

    processor, model = get_vit_model()

    # 1. Load and preprocess image
    original = Image.open(image_path).convert("RGB")
    orig_np = np.array(original)
    inputs = processor(images=original, return_tensors="pt")

    # 2. Forward pass with attentions
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # 3. Get prediction
    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=-1)
    top_idx = int(torch.argmax(probs))
    top_label = model.config.id2label[top_idx]
    top_score = float(probs[top_idx])

    # 4. Attention rollout: multiply attention matrices across layers
    # outputs.attentions is a tuple of length num_layers,
    # each shape (batch=1, heads, tokens, tokens)
    attentions = outputs.attentions  # tuple of tensors
    num_layers = len(attentions)
    num_tokens = attentions[0].shape[-1]  # e.g. 197 = 1 CLS + 196 patches

    # Average over heads, add identity (residual connection)
    rollout = torch.eye(num_tokens)
    for att in attentions:
        att_mean = att[0].mean(dim=0)              # (tokens, tokens)
        att_mean = att_mean + torch.eye(num_tokens)  # residual
        att_mean = att_mean / att_mean.sum(dim=-1, keepdim=True)
        rollout = att_mean @ rollout

    # 5. Extract CLS-token attention to all patches, drop CLS itself
    cls_attn = rollout[0, 1:]  # shape (196,)
    grid_size = int(np.sqrt(cls_attn.shape[0]))   # 14
    heatmap = cls_attn.reshape(grid_size, grid_size).numpy()

    # Normalize 0-1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # 6. Resize heatmap to original image size
    h, w = orig_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    # 7. Apply red colormap and overlay on original
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend: 60% original + 40% heatmap
    overlay = (0.6 * orig_np + 0.4 * heatmap_color).astype(np.uint8)

    return overlay, top_label, top_score


def main():
    """Standalone test."""
    overlay, label, score = compute_attention_rollout("test_face.jpg")
    print(f"Top emotion: {label} ({score*100:.1f}%)")
    print(f"Overlay shape: {overlay.shape}")

    # Save overlay as a sanity check
    Image.fromarray(overlay).save("attention_overlay_test.png")
    print("Saved: attention_overlay_test.png")


if __name__ == "__main__":
    main()
