import io
import os
import random
import re
import shutil
import zipfile

import numpy as np
import requests
import tensorflow as tf
from PIL import Image
from tensorflow.python import keras
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from custom_lenet import CustomLeNet


class ImageClassifier:

    def __init__(self, urls, output_classes, job_id, log_dir):
        self.urls = urls
        self.log_dir = log_dir
        self.job_id = job_id
        self.hyperparameters = {}
        self.output_classes = output_classes

    def _save(self):
        # TODO: Need to save the model to cloud storage as well as locally and write to the users referece
        pass

    def build_predictor(self):
        # TODO: Need to use flask to programatically create an API endpoint where the user can send new images
        # This method should return the API endpoint where the user can send urls for prediction
        pass

    def build(self):
        self._prepare_data()
        self._prepare_hyperparameters()

        model = CustomLeNet(self.input_size, self.output_classes,
                            self.hyperparameters['optimizer'], self.hyperparameters['loss']).model
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=True,
            vertical_flip=True
        )

        test_datagen = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=True,
            vertical_flip=True
        )

        train_generator = train_datagen.flow_from_directory(
            'datasets/train',
            target_size=(self.input_size[0], self.input_size[1]),
            batch_size=self.train_batch_size
        )

        validation_generator = test_datagen.flow_from_directory(
            'datasets/test',
            target_size=(self.input_size[0], self.input_size[1]),
            batch_size=self.test_batch_size
        )

        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=self.log_dir+'scalars/'+self.job_id)

        model.fit_generator(
            train_generator,
            steps_per_epoch=self.train_img_count // self.train_batch_size,
            epochs=self.hyperparameters['epochs'],
            validation_data=validation_generator,
            validation_steps=self.test_img_count // self.test_batch_size,
            callbacks=[tensorboard_callback]
        )
        self.model = model
        self._save()

    def _prepare_hyperparameters(self):
        hyperparameters = {}
        hyperparameters['epochs'] = 100
        hyperparameters['learning_rate'] = 0.01
        hyperparameters['loss'] = 'categorical_crossentropy'
        hyperparameters['momentum'] = 0.0
        hyperparameters['decay'] = 0.0
        hyperparameters['optimizer'] = SGD(
            lr=hyperparameters['learning_rate'], momentum=hyperparameters['momentum'], decay=hyperparameters['decay'])

        self.hyperparameters = hyperparameters

    def _prepare_data(self):
        os.makedirs("datasets")
        total_img_count = 0
        cumalative_img_height = 0
        cumalative_img_width = 0
        imgs = {}

        for url in self.urls:
            r = requests.get(url)
            f = io.BytesIO(r.content)
            z = zipfile.ZipFile(f)
            z.extractall()
            filename = z.filelist[0].filename.strip('/')
            path = 'datasets/'+filename
            os.rename(filename, path)
            imgs[filename] = []
            for img_name in os.listdir(path):
                img = Image.open(os.path.join(path, img_name))
                imgs[filename].append({'image': img, 'name': img_name})
                total_img_count += 1
                img_height, img_width = img.size
                cumalative_img_height += img_height
                cumalative_img_width += img_width

        # img_size = int(max(cumalative_img_height/total_img_count,
        #                cumalative_img_width/total_img_count))
        # TODO: Image size is constant here, need to make dynamic
        img_size = 224

        # Save all images by splitting into /test & /train
        train_img_count = 0
        test_img_count = 0
        for key, img_data in imgs.items():
            os.makedirs('datasets/train/'+key)
            os.makedirs('datasets/test/'+key)
            # Reshape all images
            dataset_size = len(img_data)
            split = int(dataset_size * 0.7)
            train_imgs = img_data[0:split]
            test_imgs = img_data[split:]

            for im in train_imgs:
                train_img_count += 1
                img = im['image'].resize((img_size, img_size))
                img.save('datasets/train/{}/{}'.format(key,
                                                       im['name']))
            for im in test_imgs:
                test_img_count += 1
                img = im['image'].resize((img_size, img_size))
                img.save('datasets/test/{}/{}'.format(key,
                                                      im['name']))

            self.train_batch_size = min(16, train_img_count)
            self.test_batch_size = min(16, test_img_count)
            self.train_img_count = train_img_count
            self.test_img_count = test_img_count

            self.input_size = (img_size, img_size, 3)
            # cleanup
            shutil.rmtree('datasets/'+key)
