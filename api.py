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
    if aspect < 0.6 or aspect > 1.7:
        return False, "Image aspect ratio is not consistent with a chest X-ray."

    # ----------------------------------------------------------
    # Check 2: Near-Grayscale (X-rays are monochrome)
    # R, G, B channels should be nearly identical.
    # ----------------------------------------------------------
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Mean absolute difference across channels
    rg_diff = np.abs(r - g).mean()
    rb_diff = np.abs(r - b).mean()
    gb_diff = np.abs(g - b).mean()
    color_deviation = (rg_diff + rb_diff + gb_diff) / 3.0

    if color_deviation > 15.0:
        return False, "Image contains too much color to be a chest X-ray."

    # ----------------------------------------------------------
    # Check 3: Intensity Distribution
    # X-rays have a wide spread of intensities (dark background,
    # bright bones). StdDev should be in a characteristic range.
    # ----------------------------------------------------------
    std_dev = gray.std()

    if std_dev < 25.0:
        return False, "Image intensity range is too narrow (appears blank or uniform)."

    if std_dev > 110.0:
        return False, "Image intensity distribution is not consistent with a chest X-ray."

    # ----------------------------------------------------------
    # Check 4: Dark Border Ratio
    # X-rays typically have dark collimation borders.
    # Sample border strips and check fraction of dark pixels.
    # ----------------------------------------------------------
    border_size = max(int(min(h, w) * 0.08), 5)

    top = gray[:border_size, :]
    bottom = gray[-border_size:, :]
    left = gray[:, :border_size]
    right = gray[:, -border_size:]

    border_pixels = np.concatenate([
        top.flatten(),
        bottom.flatten(),
        left.flatten(),
        right.flatten()
    ])

    dark_fraction = (border_pixels < 50).mean()

    if dark_fraction < 0.30:
        return False, "Image lacks the dark borders typical of a chest X-ray."

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