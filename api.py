from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# Import model and configuration
from models.cnn_model import CNNModel
from config.config import *

# ============================================================
# Flask App
# ============================================================
app = Flask(__name__)
CORS(app)

# ============================================================
# Load Model
# ============================================================
model = CNNModel()

model.load_state_dict(
    torch.load(
        MODEL_SAVE_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

# ============================================================
# Image Transform
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================================================
# Validate Chest X-ray (Heuristic)
# ============================================================
def validate_xray(image):
    """
    Validates whether the uploaded image appears to be a chest X-ray
    using heuristic checks on pixel statistics.

    Returns:
        (bool, str, dict) — (is_valid, reason_if_invalid, debug_stats)
    """

    # Convert to numpy arrays
    rgb = np.array(image.convert("RGB")).astype(np.float32)
    gray = np.array(image.convert("L")).astype(np.float32)

    h, w = gray.shape

    # Compute HSV values for color analysis
    r, g, b = rgb[:, :, 0] / 255.0, rgb[:, :, 1] / 255.0, rgb[:, :, 2] / 255.0
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Saturation
    saturation = np.where(cmax > 0, delta / cmax, 0)
    mean_sat = float(saturation.mean())

    # Hue (0-360)
    hue = np.zeros_like(delta)
    mask = delta > 0.01
    # Where R is max
    r_max = mask & (cmax == r)
    hue[r_max] = (60 * ((g[r_max] - b[r_max]) / delta[r_max])) % 360
    # Where G is max
    g_max = mask & (cmax == g)
    hue[g_max] = (60 * ((b[g_max] - r[g_max]) / delta[g_max]) + 120) % 360
    # Where B is max
    b_max = mask & (cmax == b)
    hue[b_max] = (60 * ((r[b_max] - g[b_max]) / delta[b_max]) + 240) % 360

    # Hue diversity: std dev of hue in saturated regions only
    sat_mask = saturation > 0.10
    if sat_mask.sum() > 100:
        hue_in_sat = hue[sat_mask]
        hue_std = float(np.std(hue_in_sat))
    else:
        # Very few saturated pixels — image is essentially grayscale
        hue_std = 0.0

    # Intensity stats
    std_dev = float(gray.std())
    aspect = float(w / h) if h > 0 else 0

    # Brightness stats
    mean_brightness = float(gray.mean())
    dark_ratio = float((gray < 60).mean())

    debug_stats = {
        "mean_saturation": round(mean_sat, 4),
        "hue_std": round(hue_std, 2),
        "intensity_std": round(std_dev, 2),
        "mean_brightness": round(mean_brightness, 2),
        "dark_pixel_ratio": round(dark_ratio, 4),
        "aspect_ratio": round(aspect, 3),
        "size": f"{w}x{h}"
    }

    # ----------------------------------------------------------
    # Check 1: Aspect Ratio
    # ----------------------------------------------------------
    if aspect < 0.5 or aspect > 2.0:
        return False, "Image aspect ratio is not consistent with a chest X-ray.", debug_stats

    # ----------------------------------------------------------
    # Check 2: Color Check (Saturation + Hue Diversity)
    #
    # Strategy:
    #   - Low saturation (< 0.15) → Pass (clearly grayscale/near-gray)
    #   - Medium saturation (0.15–0.45) → Check hue diversity
    #       - Low hue std (< 50) → Pass (uniform tint, like a warm X-ray)
    #       - High hue std (>= 50) → Reject (diverse colors, like a photo)
    #   - High saturation (> 0.45) → Reject (definitely a color photo)
    # ----------------------------------------------------------
    if mean_sat > 0.45:
        return False, "Image contains too much color to be a chest X-ray.", debug_stats

    if mean_sat > 0.15 and hue_std >= 50.0:
        return False, "Image has too many different colors to be a chest X-ray.", debug_stats

    # ----------------------------------------------------------
    # Check 3: Intensity Distribution
    # ----------------------------------------------------------
    if std_dev < 15.0:
        return False, "Image appears blank or uniform.", debug_stats

    if std_dev > 130.0:
        return False, "Image intensity is not consistent with a medical image.", debug_stats

    # ----------------------------------------------------------
    # Check 4: Dark Region Check
    # X-rays always have dark background regions (collimation,
    # lung fields). At least 20% of pixels should be dark.
    # Bright photos (hands, faces, objects in daylight) fail this.
    # ----------------------------------------------------------
    if dark_ratio < 0.15 and mean_brightness > 150:
        return False, "Image is too bright — chest X-rays have dark background regions.", debug_stats

    # ----------------------------------------------------------
    # All checks passed
    # ----------------------------------------------------------
    return True, "", debug_stats


# ============================================================
# Detect Lung Side
# ============================================================
def detect_lung_side(image):
    img = np.array(image.convert("L"))

    h, w = img.shape

    left = img[:, :w // 2].mean()
    right = img[:, w // 2:].mean()

    if abs(left - right) < 5:
        return "Both Lungs"
    elif left > right:
        return "Left Lung"
    else:
        return "Right Lung"

# ============================================================
# Health Check Route
# ============================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "Pneumonia Detection API is running"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

# ============================================================
# Prediction Route
# ============================================================
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded"
        }), 400

    try:

        file = request.files["file"]

        image = Image.open(file).convert("RGB")

        # ── Validate that the image is a chest X-ray ──
        is_valid, reason, debug_stats = validate_xray(image)

        if not is_valid:
            return jsonify({
                "error": "Invalid image. Please upload a chest X-ray image.",
                "detail": reason,
                "validation": "failed",
                "debug": debug_stats
            }), 400

        # ── Run CNN prediction ──
        input_img = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            output = model(input_img)

            probs = torch.softmax(output, dim=1)

            confidence, pred = torch.max(probs, 1)

        confidence = float(confidence.item() * 100)

        prediction = "PNEUMONIA" if pred.item() == 1 else "NORMAL"

        lung = (
            detect_lung_side(image)
            if prediction == "PNEUMONIA"
            else "None"
        )

        return jsonify({
            "result": prediction,
            "confidence": round(confidence, 2),
            "lung": lung
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# ============================================================
# Run Local Development Server
# ============================================================
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )