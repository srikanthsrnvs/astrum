import random

import numpy as np


class InvalidConvolutionTypeError(Exception):
    pass


class Preprocessor(object):

    def __init__(self, X_train, Y_train, X_test, Y_test, X_validation=None, Y_validation=None):
        super().__init__()

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.training_set = (X_train, Y_train)
        self.test_set = (X_test, Y_test)
        self.validation_set = (X_validation, Y_validation)

    def prepare_for_convolution(self, type, num_channels=1, input_shape):

        input_dimensions = len(input_shape)
        if type == 'conv_1d':
            self.X_train = self.X_train.reshape(
                (self.X_train.shape[0], self.X_train.shape[1]))
            self.training_set = (self.training_set[0].reshape(
                (self.training_set[0].shape[0]*self.training_set[0].shape[1], num_channels)))
        elif type == 'conv_2d':
            pass
        elif type == 'conv_3d':
            pass
        else:
            raise InvalidConvolutionTypeError

    def _prepare_conv_1d_shape(self, num_channels, input_dimension):
        if input_dimension == 1:
            in

    def _prepare_conv_2d_shape(self, num_channels):
        pass

    def _prepare_conv_3d_shape(self, num_channels):
        pass
