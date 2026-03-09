from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Ensure these files exist in your local directory structure
from models.cnn_model import CNNModel
from config.config import *

app = Flask(__name__)
CORS(app)

# Model Initialization
model = CNNModel()
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def detect_lung_side(image):
    img = np.array(image.convert("L"))
    h, w = img.shape
    left = img[:, :w//2].mean()
    right = img[:, w//2:].mean()

    if abs(left - right) < 5:
        return "Both Lungs"
    elif left > right:
        return "Left Lung"
    else:
        return "Right Lung"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    input_img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_img)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    confidence_val = float(confidence.item() * 100)
    result = "PNEUMONIA" if pred.item() == 1 else "NORMAL"
    lung = detect_lung_side(image) if result == "PNEUMONIA" else "None"

    return jsonify({
        "result": result,
        "confidence": round(confidence_val, 2),
        "lung": lung
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)