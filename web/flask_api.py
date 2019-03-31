# https://github.com/mtobeiyf/keras-flask-deploy-webapp/blob/master/app.py


from __future__ import division, print_function
# coding=utf-8
import sys
sys.stdout = sys.stderr

import os
import glob
import re
import argparse


# Model
import keras
import numpy as np

import matplotlib

import vis
import vis.utils
import vis.visualization

from PIL import Image as pil_image
import scipy.misc

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import tensorflow as tf


# Define a flask app
app = Flask(__name__)
model = keras.models.load_model('/server/model.hdf5')
model._make_predict_function()
graph = tf.get_default_graph()
model.summary()
layer_idx = vis.utils.utils.find_layer_idx(model, 'dense_1')
penultimate_layer = vis.utils.utils.find_layer_idx(model, 'Conv_1')


def parse_args(argv):
    """parse input arguments"""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('model', type=str,
                        help='Input path to model file')

    args = parser.parse_args(args=argv)
    return args


def preproc(data):
    data /= 127.5
    data -= 1.
    return data


def model_predict(img_path, model):
### This block is executed for each request
    # Load image
    img = keras.preprocessing.image.load_img(img_path, grayscale=True)
    size = img.size
    img = img.resize((224, 224), pil_image.NEAREST)
    img = keras.preprocessing.image.img_to_array(img)
    img = preproc(img)
    x = np.expand_dims(img, axis=0)

    # Predict
    with graph.as_default():
        res = model.predict(x)

        grads = vis.visualization.visualize_cam(model,
                layer_idx,
                penultimate_layer_idx=penultimate_layer,
                filter_indices=0,
                seed_input=img)

    jet_heatmap = np.uint8(matplotlib.cm.jet(grads)[..., :3] * 255)
    jet_heatmap = scipy.misc.imresize(jet_heatmap, size)

    # Output
    fpath = os.path.join('static', secure_filename('heatmap.png'))
    matplotlib.image.imsave(fpath, jet_heatmap)

    return res[0][0], fpath


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        try:
            pred, fpath = model_predict(file_path, model)
        except Exception as e:
            print(e)
        print(pred)
        print(fpath)
        return str(pred) + ',' + str(fpath)
    return None


from werkzeug.serving import run_with_reloader
from werkzeug.debug import DebuggedApplication

@run_with_reloader
def run_server():
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), DebuggedApplication(app))
    http_server.serve_forever()
