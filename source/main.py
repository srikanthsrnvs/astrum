from archbuilder import ArchBuilder
import tensorflow as tf
import numpy as np
from tensorflow.python import keras


if __name__ == "__main__":

    builder = ArchBuilder("v1", min_hidden_layers=1, max_hidden_layers=2)

    good_models = []
    bad_models = []

    for arch in range(1, 10):

        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        neural_arch = builder.build_random_arch("dense{}".format(arch), (28, 28), 10, keras.activations.softmax)

        neural_arch.set_train_data(train_images, train_labels)
        neural_arch.set_test_data(test_images, test_labels)

        log_dir = 'models/fit/{}'.format(arch)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        neural_arch.train(callbacks=[tensorboard_callback])

        neural_arch.evaluate()

        if neural_arch.accuracy > 0.5:
            good_models.append(neural_arch)
            neural_arch.model.save('{}{}.h5'.format(neural_arch.accuracy, neural_arch.identifier))
        else:
            bad_models.append(neural_arch)


    print("Final statistics;\n{} good models\n{} bad models".format(len(good_models), len(bad_models)))



    