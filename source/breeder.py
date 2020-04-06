import numpy as np
import tensorflow as tf
from tensorflow.python import keras


class Breeder:

    def __init__(self):
        # Create the neural network
        # Add randomness to action step
        layers = [
            keras.layers.Input((3, 50)),
            keras.layers.Dense(25, activation=keras.activations.relu),
            keras.layers.Dense(75, activation=keras.activations.relu),
            keras.layers.Dense(10, activation=keras.activations.softmax)
        ]
        self.model = keras.Sequential(layers)

    def breed(self, arch1, arch2):
        # Predict action and take it
        pass
