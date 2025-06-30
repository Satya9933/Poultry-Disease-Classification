from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
MODEL_PATH = 'poultry_model.h5'  
model = load_model(MODEL_PATH)

# Define your class labels (must match training order)
class_names = ['Coccidiosis', 'Salmonella', 'Healthy']  

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part in the request'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No file selected'
    
    if file:
        # Save file temporarily
        file_path = os.path.join('static', file.filename)
        file.save(file_path)

        # Load and preprocess image
        img = image.load_img(file_path, target_size=(224, 224))  # Change size if needed
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize

        # Prediction
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_label = class_names[predicted_index]
        confidence = round(float(np.max(prediction)) * 100, 2)

        return render_template('result.html', 
                               label=predicted_label,
                               confidence=confidence,
                               image_path=file_path)

    return 'Something went wrong'

if __name__ == '__main__':
    app.run(debug=True)