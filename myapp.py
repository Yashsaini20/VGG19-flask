from __future__ import division, print_function

import imp
import numpy as np
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np


# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
model_path = 'vgg19.h5'

##load
model = load_model(model_path)
model._make_predict_function()  ##Necessary


## Preprocessing function
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)  ## Expanding dimension


    x = preprocess_input(x)

    preds = model.predict(x)
    return preds

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method == "POST":
        ## get file from post
        f=request.files['file']

        #save the file to ./uploads

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        ## here we make prediction
        pred = model_predict(file_path,model)  ## givesa class label
       
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(pred, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])
        return result
    return None               # Convert to string

if __name__ == '__main__':
    app.run(debug=True)