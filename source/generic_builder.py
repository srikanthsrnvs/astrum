import tensorflow as tf
from tensorflow.python import keras
import random
import numpy as np
from custom_lenet import CustomLeNet


class GenericBuilder:

    def __init__(self, type, input_size, output_classes):
        self.type = type
        self.input_size = input_size
        self.output_classes = output_classes

        if self.type == 'structured_classification':
            pass
        elif self.type == 'structured_prediction':
            pass
        elif self.type == 'image_classification':
            self.model = self.build_image_classifier()


    def build_image_classifier(self):
        model = CustomLeNet(self.input_shape, self.output_classes)
        return model

    def build_data_predictor(self):
        pass

