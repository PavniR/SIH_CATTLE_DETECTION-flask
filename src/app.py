# app.py
from flask import Flask, request, jsonify
from detection import detect_breed, detect_keypoints
from keypoints_atc import extract_traits, calculate_atc_scores
import cv2
import numpy as np
import os

app = Flask(__name__)

# Temporary folder to save uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/analyze", methods=["POST"])
def analyze_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Run breed detection
    breed, confidence = detect_breed(file_path)

    # Run keypoint detection
    keypoints = detect_keypoints(file_path)
    if keypoints is None:
        return jsonify({"error": "Keypoints not detected"}), 400

    # Calculate traits and ATC scores
    traits = extract_traits(keypoints)
    scores = calculate_atc_scores(traits)

    # Return results as JSON
    return jsonify({
        "breed": breed,
        "confidence": confidence,
        "scores": scores
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
