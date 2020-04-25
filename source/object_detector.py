import io
import os
import random
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from PIL import Image
from tensorflow.python import keras
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from custom_lenet import CustomLeNet
from firebase import FirebaseHelper
from job import Job
from saving_worker import SavingWorker


class ObjectDetector:

    def __init__(self, job, log_dir, finished_queue, cv):
        self.log_dir = log_dir
        self.cv = cv
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
            {'job': self.job, 'label_map': self.label_map})
        self.cv.notifyAll()
        shutil.rmtree('./'+self.job.filename)

    def build(self):
        self._prepare_data()
        self._prepare_hyperparameters()

        model = CustomLeNet(len(self.output_classes),
                            self.hyperparameters['optimizer'], self.hyperparameters['output_activation'], self.hyperparameters['loss']).model
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

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=self.train,
            directory='./',
            x_col=self.filename_col_header,
            y_col=self.label_col_header,
            classmode='categorical',
            classes=self.output_classes,
            shuffle=True,
            target_size=(self.input_size[0], self.input_size[1]),
            batch_size=self.train_batch_size
        )

        validation_generator = test_datagen.flow_from_dataframe(
            dataframe=self.test,
            directory='./',
            x_col=self.filename_col_header,
            y_col=self.label_col_header,
            classmode='categorical',
            classes=self.output_classes,
            shuffle=True,
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
        hyperparameters['learning_rate'] = 0.01
        hyperparameters['loss'] = 'binary_crossentropy'
        hyperparameters['momentum'] = 0.9
        hyperparameters['decay'] = 0
        hyperparameters['optimizer'] = RMSprop(
            lr=hyperparameters['learning_rate'], decay=hyperparameters['decay'])
        hyperparameters['output_activation'] = 'sigmoid'

        self.hyperparameters = hyperparameters

    def _prepare_data(self):

        total_img_count = 0
        cumalative_img_height = 0
        cumalative_img_width = 0
        imgs = []

        r = requests.get(self.job.download_link)
        f = io.BytesIO(r.content)
        z = zipfile.ZipFile(f)
        z.extractall()
        filename = z.filelist[0].filename.strip('/')

        self.job.set_filename(filename)

        csv_filename = [name for name in os.listdir(
            filename) if '.csv' in name][0]

        img_path = filename + '/images'

        for img_name in os.listdir(img_path):
            # TODO: Error handling if a file is not an image
            img = Image.open(os.path.join(img_path, img_name))
            imgs.append({'image': img, 'name': img_name})
            total_img_count += 1
            img_height, img_width = img.size
            cumalative_img_height += img_height
            cumalative_img_width += img_width
            img = img.resize((299, 299))
            img.save(img_path+'/'+img_name)

        self.input_size = (299, 299, 3)

        data_frame = pd.read_csv(filename+'/'+csv_filename)
        sample = data_frame.sample(n=1)
        filename_column = ""
        for col in sample:
            matches = re.findall(r'(\/.*?\.[\w:]+)', str(sample[col]))
            if len(matches) > 0:
                filename_column = col
                sample = matches[0]
                break

        if filename_column == "":
            # TODO: Handle the error if there arent any filenames in the dataset
            pass

        headers = list(data_frame.columns)
        # Add the filename path to the path in the filename column for all images
        splitter = "" if data_frame[filename_column][0] == '/' else '/'
        data_frame[filename_column] = self.job.filename + splitter + \
            data_frame[filename_column].astype(str)

        frame_type = 'single'

        for counts in data_frame[filename_column].value_counts():
            if counts > 1:
                frame_type = 'duplicate'
                break

        label_column = headers[1 if headers.index(
            filename_column) == 0 else 0]

        self.filename_col_header = filename_column
        self.label_col_header = label_column

        # If the data frame contains duplicate filenames for different classes, for eg.
        # __________________________
        # | filenames | data         |
        # |--------------------------|
        # | a.jpg     | desert       |
        # | a.jpg     | ocean        |
        # --------------------------

        if frame_type == 'duplicate':

            all_labels = data_frame.groupby(filename_column)[label_column].apply(
                list).reset_index(name=label_column)[label_column].tolist()
            possible_labels = set()
            for labels in all_labels:
                possible_labels = possible_labels.union(set(labels))

            self.output_classes = possible_labels = list(possible_labels)

            grouped_frame = data_frame.groupby(filename_column).agg(
                {label_column: lambda x: list(set(x))}).reset_index()

            self.data_frame = grouped_frame
            train_img_count = int(0.7*len(self.data_frame))
            self.train_img_count = train_img_count
            self.test_img_count = len(self.data_frame)-train_img_count
            self.train = self.data_frame[:train_img_count]
            self.test = self.data_frame[train_img_count:]
            self.train_batch_size = min(16, self.train_img_count)
            self.test_batch_size = min(16, self.test_img_count)

        # If the data frame contains single filenames for different classes, for eg.
        # __________________________
        # | filenames | data         |
        # |--------------------------|
        # | a.jpg     | desert, ocean|
        # | b.jpg     | ocean        |
        # --------------------------

        elif frame_type == 'single':
            self.data_frame = data_frame
            self.data_frame[self.label_col_header] = data_frame[self.label_col_header].apply(
                lambda x: x.split(','))

            self.output_classes = set()
            for labels in self.data_frame[self.label_col_header]:
                self.output_classes = self.output_classes.union(set(labels))

            self.output_classes = list(self.output_classes)
            train_img_count = int(0.7*len(self.data_frame))
            self.train_img_count = train_img_count
            self.test_img_count = len(self.data_frame)-train_img_count
            self.train = self.data_frame[:train_img_count]
            self.test = self.data_frame[train_img_count:]
            self.train_batch_size = min(16, self.train_img_count)
            self.test_batch_size = min(16, self.test_img_count)
