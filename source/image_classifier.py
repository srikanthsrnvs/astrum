import io
import os
import random
import re
import shutil
import zipfile

import numpy as np
from pathlib import Path
import requests
import tensorflow as tf
from PIL import Image
from tensorflow.python import keras
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from saving_worker import SavingWorker
from custom_lenet import CustomLeNet
from firebase import FirebaseHelper
from job import Job


class ImageClassifier:

    def __init__(self, job, log_dir, finished_queue, cv):
        self.cv = cv
        self.log_dir = log_dir
        self.job = job
        self.finished_queue = finished_queue
        self.hyperparameters = {}
        self.firebase_helper = FirebaseHelper()
        self.job_files_path = Path(str(Path.home())+'/JobFiles/'+self.job.id)

    def __save(self):
        self.model.save(str(self.job_files_path)+'/model.h5')
        with tf.keras.backend.get_session() as sess:
            tf.saved_model.simple_save(
                sess,
                str(self.job_files_path)+'/ServingModel/1',
                inputs={'input_image': self.model.input},
                outputs={t.name: t for t in self.model.outputs}
            )
        self.finished_queue.append(
            {'job': self.job, 'label_map': self.label_map, 'stats': self.stats})
        self.cv.notifyAll()
        shutil.rmtree('./'+self.job.filename)

    def build(self):
        self._prepare_data()
        self._prepare_hyperparameters()

        model = CustomLeNet(self.output_classes,
                            self.hyperparameters['optimizer'], self.hyperparameters['output_activation'], self.hyperparameters['loss']).model
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
        )

        test_datagen = ImageDataGenerator(
            rescale=1. / 255,
            width_shift_range=0.2,
            height_shift_range=0.2,
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

        stats = model.fit_generator(
            train_generator,
            steps_per_epoch=self.train_img_count // self.train_batch_size,
            epochs=self.hyperparameters['epochs'],
            validation_data=validation_generator,
            validation_steps=self.test_img_count // self.test_batch_size,
            callbacks=[tensorboard_callback]
        )

        stats = stats.history

        train_loss = stats.get('loss', '')[-1]
        test_loss = stats.get('val_loss', '')[-1]
        train_acc = stats.get('acc', '')[-1]
        test_acc = stats.get('val_acc', '')[-1]

        self.stats = {
            'train': {
                'accuracy': train_acc,
                'loss': train_loss
            },
            'test': {
                'accuracy': test_acc,
                'loss': test_loss
            }
        }
        self.model = model
        self.label_map = train_generator.class_indices
        self.__save()

    def _prepare_hyperparameters(self):
        hyperparameters = {}
        hyperparameters['epochs'] = 100
        hyperparameters['learning_rate'] = 0.01
        hyperparameters['loss'] = 'categorical_crossentropy'
        hyperparameters['momentum'] = 0.9
        hyperparameters['decay'] = 0.0
        hyperparameters['optimizer'] = SGD(
            lr=hyperparameters['learning_rate'], momentum=hyperparameters['momentum'])
        hyperparameters['output_activation'] = 'softmax'

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
        img_size = 299

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
