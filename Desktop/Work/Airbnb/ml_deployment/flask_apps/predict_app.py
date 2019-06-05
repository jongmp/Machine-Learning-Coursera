import base64
import numpy as np
import io
from PIL import Image

import os
os.environ['KERAS_BACKEND'] = 'theano'

import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
import json


app = Flask(__name__)

{def get_model():
    global model
    model = load_model('model_keras.h5')
    print(" * Model Loaded!")}

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image

print(" * Loading Keras model...")
get_model()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(150,150))

    prediction = model.predict(processed_image)
    print(prediction)
    response = prediction[0][0]
    print(response)
    if response == 1.0:
        response = 'This picture is likely to succeed'
    else:
        response = "This picture is not likely to succeed"
    return json.dumps(str(response))


'''
# Get the image from my computer to test
img = Image.open('/Users/jpar746/Desktop/5065.0nots.png')
predict(img)
'''

