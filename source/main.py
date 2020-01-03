from archbuilder import ArchBuilder
import tensorflow as tf
import numpy as np
from tensorflow.python import keras


if __name__ == "__main__":

    builder = ArchBuilder("v1", min_hidden_layers=2, max_hidden_layers=10)

    for arch in range(1, 10):

        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        neural_arch = builder.build_random_arch("dense{}".format(arch), (28, 28), 10, keras.activations.softmax)

        neural_arch.set_train_data(train_images, train_labels)
        neural_arch.set_test_data(test_images, test_labels)
        neural_arch.train()

        # arch.evaluate()
    