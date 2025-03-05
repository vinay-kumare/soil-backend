from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Soil information dictionary
soil_info = {
    0: {
        "soil_type": "Alluvial Soil",
        "description": "Highly fertile soil formed by water deposits.",
    },
    1: {
        "soil_type": "Black Soil",
        "description": "Also known as Regur soil, it is rich in clay and moisture-retaining properties",
    },
    2: {
        "soil_type": "Laterite Soil",
        "description": "Rich in iron and aluminum, It is low in fertility but can be improved with organic matter.",
    },
    3: {
        "soil_type": "Yellow Soil",
        "description": "Moderately fertile soil derived from crystalline rocks.",
    }
}

# Load the pre-trained CNN model
model = load_model('soil_model.h5')

# Define the image preprocessing function


def preprocess_image(image):
    image = image.resize((200, 200))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define the prediction route


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)

        # Make a prediction
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Get soil information
        soil_details = soil_info[predicted_class]

        # Return the result with additional details
        return jsonify({
            'soil_type': soil_details['soil_type'],
            'description': soil_details['description'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
