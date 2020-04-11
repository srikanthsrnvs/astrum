import io
import os
import random
import re
import shutil
import zipfile
from job import Job
import numpy as np
import requests
import tensorflow as tf
from PIL import Image
from tensorflow.python import keras
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from custom_lenet import CustomLeNet
from firebase import FirebaseHelper

BASE_URL = 'https://astrumdashboard.appspot.com'


class ImageClassifier:

    def __init__(self, job, log_dir):
        self.log_dir = log_dir
        self.job = job
        self.hyperparameters = {}
        self.firebase_helper = FirebaseHelper()

    def __save(self):
        self.model.save('{}.h5'.format(self.job.id))
        self.saved_model_location = self.firebase_helper.save_model(
            self.job.id)
        self.saved_logs_location = self.firebase_helper.save_logs(self.job.id)
        self.saved_tb_logs_location = self.firebase_helper.save_tb_logs(self.job.id)
        self.__notify_backend_for_completion()

    def __notify_backend_for_completion(self):
        requests.put(
            BASE_URL+'/jobs/'+self.job.id,
            json={
                'model': self.saved_model_location,
                'logs': self.saved_logs_location,
                'tb_logs': self.saved_tb_logs_location,
                'label_map': self.label_map,
                'status': 2
            }
        )
        self.__cleanup()

    def __cleanup(self):
        shutil.rmtree(self.job.filename)
        shutil.rmtree(self.job.id+'_logs')
        os.remove(self.job.id+'.h5')
        os.remove(self.job.id+'_output.txt')
        os.remove(self.job.id+'_tensorboard.zip')

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
            self.job.filename+'/train',
            target_size=(self.input_size[0], self.input_size[1]),
            batch_size=self.train_batch_size
        )

        validation_generator = test_datagen.flow_from_directory(
            self.job.filename+'/test',
            target_size=(self.input_size[0], self.input_size[1]),
            batch_size=self.test_batch_size
        )

        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=self.log_dir+'/scalars/')

        model.fit_generator(
            train_generator,
            steps_per_epoch=self.train_img_count // self.train_batch_size,
            epochs=self.hyperparameters['epochs'],
            validation_data=validation_generator,
            validation_steps=self.test_img_count // self.test_batch_size,
            callbacks=[tensorboard_callback]
        )
        self.model = model
        self.label_map = train_generator.class_indices
        self.__save()

    def _prepare_hyperparameters(self):
        hyperparameters = {}
        hyperparameters['epochs'] = 100
        hyperparameters['learning_rate'] = 0.1
        hyperparameters['loss'] = 'categorical_crossentropy'
        hyperparameters['momentum'] = 0.0
        hyperparameters['decay'] = 0.0
        hyperparameters['optimizer'] = SGD(
            lr=hyperparameters['learning_rate'], momentum=hyperparameters['momentum'])

        self.hyperparameters = hyperparameters

    def _prepare_data(self):
        
        total_img_count = 0
        cumalative_img_height = 0
        cumalative_img_width = 0
        imgs = {}

        r = requests.get(self.job.download_link)
        f = io.BytesIO(r.content)
        z = zipfile.ZipFile(f)
        z.extractall()
        filename = z.filelist[0].filename.strip('/')
        self.job.set_filename(filename)

        for folder in os.listdir(filename):
            path = filename+'/'+folder
            imgs[folder] = []
            for img_name in os.listdir(path):
                # TODO: Error handling if a file is not an image
                img = Image.open(os.path.join(path, img_name))
                imgs[folder].append({'image': img, 'name': img_name})
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
            os.makedirs(filename+'/train/'+key)
            os.makedirs(filename+'/test/'+key)
            # Reshape all images
            dataset_size = len(img_data)
            split = int(dataset_size * 0.7)
            train_imgs = img_data[0:split]
            test_imgs = img_data[split:]

            for im in train_imgs:
                train_img_count += 1
                img = im['image'].resize((img_size, img_size))
                img.save(filename+'/train/{}/{}'.format(key,
                                                       im['name']))
            for im in test_imgs:
                test_img_count += 1
                img = im['image'].resize((img_size, img_size))
                img.save(filename+'/test/{}/{}'.format(key,
                                                      im['name']))
            # cleanup
            shutil.rmtree(filename+'/'+key)

        self.train_batch_size = min(16, train_img_count)
        self.test_batch_size = min(16, test_img_count)
        self.train_img_count = train_img_count
        self.test_img_count = test_img_count
        self.input_size = (img_size, img_size, 3)
        self.output_classes = len(imgs.keys())
