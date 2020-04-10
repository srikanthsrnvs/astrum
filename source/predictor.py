from pathlib import Path

import numpy as np
import requests
import tensorflow as tf
from flask import jsonify, request
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.python import keras

from firebase import FirebaseHelper
from lrn import LRN
from pool_helper import PoolHelper


class Predictor:

    def __init__(self, model_path, job_id, job_type, label_map=[]):

        self.job_type = job_type

        if job_type == 'image_classification':
            home_dir = str(Path.home())
            models_dir = home_dir + '/models/'
            model_local_location = models_dir + job_id

            FirebaseHelper().get_file(model_path, model_local_location)
            self.model = load_model(model_local_location, custom_objects={
                                    'PoolHelper': PoolHelper, 'LRN': LRN})
            self.label_map = label_map

        elif job_type == 'structured_classification':
            pass
        elif job_type == 'structured_prediction':
            pass
        elif job_type == 'custom':
            pass

    def predict(self, image):
        if self.job_type == 'image_classification':
            img = Image.open(image)
            img = img.resize((224, 224))
            img_tensor = (keras_image.img_to_array(img))/255.
            img_tensor = np.expand_dims(img_tensor, axis=0)
            prediction = self.model.predict(img_tensor)
            confidence = "{:.2f}".format(np.max(prediction))
            class_prediction = self.label_map[np.argmax(prediction)]
            return jsonify({"prediction": class_prediction, 'confidence': confidence}), 200
