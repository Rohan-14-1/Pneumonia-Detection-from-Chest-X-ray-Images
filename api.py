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