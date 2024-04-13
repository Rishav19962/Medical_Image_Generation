from flask import Flask, render_template, request
from keras.models import load_model
from numpy.random import randn
import numpy as np
from PIL import Image
import io
import base64
app = Flask(__name__)

# Load the generator model
model = load_model('generator_model_045.h5')

# Function to generate images
def generate_images(num_images):
    latent_points = generate_latent_points(100, num_images)
    X = model.predict(latent_points)
    return X

# Generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# Function to convert image array to HTML img tags
def get_image_html(image_array):
    images_html = ""
    for img_array in image_array:
        # Convert image to grayscale
        grayscale_img_array = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])  # Convert RGB to grayscale
        
        # Scale pixel values to [0, 255]
        grayscale_img_array = ((grayscale_img_array+1)/2*255).astype(np.uint8)
        img = Image.fromarray(grayscale_img_array)
        
        # Convert image to base64
        with io.BytesIO() as buffer:
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Example HTML output using base64 encoding
        images_html += f'<img src="data:image/png;base64,{img_base64}" style="margin: 10px;">'
    return images_html


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    num_images = int(request.form['num_images'])
    generated_images = generate_images(num_images)
    images_html = get_image_html(generated_images)
    return render_template('result.html', images_html=images_html)

if __name__ == '__main__':
    app.run(debug=True)