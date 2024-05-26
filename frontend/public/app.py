from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import json


app = Flask(__name__)
model = tf.keras.models.load_model("animal_classifier_model.keras")

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)  # Preprocess for EfficientNet
    return img_array

@app.route('/')
def index():
    return render_template('species.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img_array = load_and_preprocess_image(filepath, (224, 224))
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Load the class label map
        with open('C:/Users/tanya/OneDrive/Desktop/BioSpotter/frontend/public/class_labels.json', 'r') as f:
            label_map = json.load(f)
        predicted_label = label_map[str(predicted_class)]

        return jsonify(prediction=predicted_label)
    else:
        return jsonify(error="File type not allowed"), 400

if __name__ == '__main__':
    app.run(debug=True)
