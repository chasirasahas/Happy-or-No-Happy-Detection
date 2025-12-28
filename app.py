from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import RandomHeight, RandomWidth  

app = Flask(__name__)


custom_objects = {
    "RandomHeight": RandomHeight,
    "RandomWidth": RandomWidth
}

try:
    model = load_model("new_model.h5", custom_objects=custom_objects)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Methanata folders name tika danna thiyenne (train karana images tika)
class_labels = [
    'happy', 
    'not_happy',
]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_disease(image_path):
    if model is None:
        return {"error": "Model not loaded properly"}

    try:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))  
        image = np.array(image) / 255.0  
        image = np.expand_dims(image, axis=0)  

        prediction = model.predict(image)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        result = {"class": class_labels[class_index], "confidence": float(confidence)}
        return result

    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html') # methanata HTML eke name eka danna thiyenne

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = predict_disease(filepath)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)