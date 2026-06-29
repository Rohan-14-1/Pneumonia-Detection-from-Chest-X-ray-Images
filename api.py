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
        (bool, str) — (is_valid, reason_if_invalid)
    """

    # Convert to numpy arrays
    rgb = np.array(image.convert("RGB")).astype(np.float32)
    gray = np.array(image.convert("L")).astype(np.float32)

    h, w = gray.shape

    # ----------------------------------------------------------
    # Check 1: Aspect Ratio (chest X-rays are roughly square)
    # ----------------------------------------------------------
    aspect = w / h if h > 0 else 0
    if aspect < 0.5 or aspect > 2.0:
        return False, "Image aspect ratio is not consistent with a chest X-ray."

    # ----------------------------------------------------------
    # Check 2: Low Saturation (X-rays are near-monochrome)
    # Even warm-tinted X-rays have low color saturation.
    # Real photos (feet, faces, objects) have high saturation.
    # Uses HSV color space for robust color detection.
    # ----------------------------------------------------------
    r, g, b = rgb[:, :, 0] / 255.0, rgb[:, :, 1] / 255.0, rgb[:, :, 2] / 255.0

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Saturation = delta / cmax (where cmax > 0)
    saturation = np.where(cmax > 0, delta / cmax, 0)
    mean_saturation = saturation.mean()

    # X-rays (even warm-tinted) typically have saturation < 0.18
    # Real-world color photos typically have saturation > 0.25
    if mean_saturation > 0.22:
        return False, "Image contains too much color saturation to be a chest X-ray."

    # ----------------------------------------------------------
    # Check 3: Intensity Distribution
    # X-rays have a wide spread of intensities (dark background,
    # bright bones). StdDev should be in a characteristic range.
    # ----------------------------------------------------------
    std_dev = gray.std()

    if std_dev < 15.0:
        return False, "Image intensity range is too narrow (appears blank or uniform)."

    if std_dev > 130.0:
        return False, "Image intensity distribution is not consistent with a medical image."

    # ----------------------------------------------------------
    # All checks passed
    # ----------------------------------------------------------
    return True, ""


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
        is_valid, reason = validate_xray(image)

        if not is_valid:
            return jsonify({
                "error": "Invalid image. Please upload a chest X-ray image.",
                "detail": reason,
                "validation": "failed"
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