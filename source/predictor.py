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

    def __init__(self, job):

        self.job = job

        if self.job.type == 'image_classification':
            home_dir = str(Path.home())
            models_dir = home_dir + '/models/'
            model_local_location = models_dir + self.job.id
            if not Path(model_local_location).is_file():
                FirebaseHelper().get_file(self.job.model, model_local_location)
            self.model = load_model(model_local_location, custom_objects={
                                    'PoolHelper': PoolHelper, 'LRN': LRN})
            self.label_map = self.job.label_map

        elif self.job.type == 'object_detection':
            home_dir = str(Path.home())
            models_dir = home_dir + '/models/'
            model_local_location = models_dir + self.job.id
            if not Path(model_local_location).is_file():
                FirebaseHelper().get_file(self.job.model, model_local_location)
            self.model = load_model(model_local_location, custom_objects={
                                    'PoolHelper': PoolHelper, 'LRN': LRN})
            self.label_map = self.job.label_map

        elif self.job.type == 'structured_classification':
            pass
        elif self.job.type == 'structured_prediction':
            pass
        elif self.job.type == 'custom':
            pass

    def predict(self, input_data):
        if self.job.type == 'image_classification':
            img = Image.open(input_data)
            img = img.resize((224, 224))
            img_tensor = (keras_image.img_to_array(img))/255.
            img_tensor = np.expand_dims(img_tensor, axis=0)
            prediction = self.model.predict(img_tensor)
            confidence = "{:.2f}".format(np.max(prediction))
            class_prediction = self.label_map[np.argmax(prediction)]
            return jsonify({"prediction": class_prediction, 'confidence': confidence}), 200

        elif self.job.type == 'object_detection':
            img = Image.open(input_data)
            img = img.resize((224, 224))
            img_tensor = (keras_image.img_to_array(img))/255.
            img_tensor = np.expand_dims(img_tensor, axis=0)
            prediction = self.model.predict(img_tensor)
            confidence_dict = {}
            print(prediction[0])
            for index in range(0, len(prediction[0])):
                class_name = self.label_map[index]
                confidence_dict[class_name] = "{:.2f}".format(prediction[0][index])
            return jsonify({'predictions': confidence_dict}), 200