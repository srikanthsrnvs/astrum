from __future__ import print_function

import imageio
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python import keras
from tensorflow.python.keras.layers import (Activation, Average,
                                            AveragePooling2D, Concatenate,
                                            Conv2D, Dense, Dropout, Flatten,
                                            Input, MaxPooling2D, Reshape,
                                            ZeroPadding2D, GlobalAveragePooling2D)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.applications import InceptionV3
from lrn import LRN
from pool_helper import PoolHelper


class CustomLeNet:

    def __init__(self, output_classes, optimizer, output_activation, loss, weights_path=''):
        self.output_classes = output_classes
        self.weights_path = weights_path
        self.optimizer = optimizer
        self.loss = loss
        self.output_activation = output_activation
        self.model = self.build_lenet()

    def build_lenet(self):

        base_model = InceptionV3(weights='imagenet', include_top=False)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(self.output_classes, activation=self.output_activation)(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

        return model
