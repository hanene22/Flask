from flask import Flask, render_template, send_file, redirect, url_for, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

generator = None

def load_model():
    global generator
    generator = tf.keras.models.load_model('model_generator3.h5')

def generate_image():
    global generator
    if generator is None:
        raise ValueError("The model is not loaded. Please load the model first.")

    noise = np.random.normal(0, 1, (1, 100))
    generated_image = generator.predict(noise)
    generated_image = (generated_image * 127.5 + 127.5).astype(np.uint8)
    pil_image = Image.fromarray(generated_image[0])
    return pil_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_model', methods=['POST'])
def load_model_route():
    load_model()
    return jsonify({'status': 'Model loaded successfully'})

@app.route('/generate_image', methods=['POST'])
def generate_image_route():
    image = generate_image()
    image_path = 'static/generated_image.png'
    image.save(image_path)
    return jsonify({'status': 'Image generated successfully', 'image_url': url_for('image')})

@app.route('/image')
def image():
    return send_file('static/generated_image.png', mimetype='image/png')



if __name__ == '__main__':
    app.run(debug=True)