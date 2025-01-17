from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load the Keras model
model = tf.keras.models.load_model('model.h5')

# Define class labels (replace with your actual class labels)
class_labels = [
    'Aloevera', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha',
    'Basale', 'Betel', 'Brahmi', 'Bringaraja', 'Butterfly pea',
    'Caricature', 'Castor', 'Chakte', 'Citron lime', 'Coffee',
    'Coriender', 'Cuban oregano', 'Curry', 'Doddapatre', 'Ekka',
    'Ganike', 'Gasagase', 'Guava', 'Ginger', 'Green chiretta',
    'Henna', 'Honge', 'Insulin', 'Jasmine', 'Kachhar',
    'Kamakasturi', 'Kasambruga', 'Kepala', 'Lemon', 'Lemon grass',
    'Malabar Spinach', 'Marigold', 'Mint', 'Nagadali', 'Neem',
    'Nerale', 'Nithyapushpa', 'Nooni', 'Palak', 'Papaya',
    'Parijatha', 'Pepper', 'Radish', 'Raktachandini', 'Red flame ivy',
    'Sampige', 'Tamarind', 'Thumbe', 'Tulasi', 'Turmeric'
]

def preprocess_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = img.resize((224, 224))  # Resize to model input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img_array = preprocess_image(file)

        # Debug: Ensure the image array is correct
        print(f'Image array shape: {img_array.shape}')

        # Prediction
        prediction = model.predict(img_array)

        # Debug: Check the prediction result
        print(f'Prediction: {prediction}')

        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]

        return jsonify({'label': predicted_class_label})
    except Exception as e:
        # Debug: Print the error message
        print(f'Error during prediction: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)





