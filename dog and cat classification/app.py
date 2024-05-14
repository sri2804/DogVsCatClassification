import os
import cv2
import numpy as np
from flask import Flask, request, render_template, url_for

# Load the saved model
from tensorflow.keras.models import load_model
model = load_model('dog_animal_classifier.h5')

# Function to classify an image as dog or cat
def classify_image(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Perform prediction
    prediction = model.predict(img)

    # Classify the prediction
    if prediction[0] < 0.5:
        return 'Dog', img
    else:
        return 'Cat', img

# Initialize Flask application
app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling image upload and classification
@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        # If the user does not select a file, the browser may submit an empty file without a filename
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file:
            # Save the uploaded file to a temporary location
            temp_path = 'static/temp.jpg'  # Update the path to save the file in the static folder
            file.save(temp_path)

            # Classify the uploaded image
            classification = classify_image(temp_path)

            # Return the classification result and pass the path to the uploaded image
            return render_template('index.html', message='File uploaded successfully', classification=classification[0])

if __name__ == '__main__':
    app.run(debug=True)
