from archbuilder import ArchBuilder
import tensorflow as tf
import numpy as np
from tensorflow.python import keras
from generic_builder import GenericBuilder
from neuralarch import NeuralArch
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import optparse
import requests, zipfile, io, re
import shutil


def build_generic_model(type, urls):

    os.makedirs("datasets")

    for url in urls:

        r = requests.get(url)
        f = io.BytesIO(r.content)
        z = zipfile.ZipFile(f)
        z.extractall()

        filename = z.filelist[0].filename.strip('/')

        os.rename(filename, 'datasets/{}'.format(filename))

   features = tfds.features.FeaturesDict({
            "image": tfds.features.Image(shape=(_TILES_SIZE,) * 2 + (3,)),
            "label": tfds.features.ClassLabel(
                names=_CLASS_NAMES),
            "filename": tfds.features.Text(),
        })


def get_dataset():
    filename = 'my_train_dataset.csv'
    generator = lambda: read_csv(filename)
    return tf.data.Dataset.from_generator(
        generator, (tf.float32, tf.int32), ((n_features,), ()))

    # generic_builder = GenericBuilder(type, input_size, output_classes)

    # neural_arch = NeuralArch(generic_builder.model, None, None, "GenericModel", 1)

    # neural_arch.set_train_data(train_data)
    

if __name__ == "__main__":

    parser = optparse.OptionParser()

    parser.add_option('--url',
        action="store", dest="url",
        help="The location of the dataset", default="")
    parser.add_option('--type',
        action="store", dest="type",
        help="The type of network to build", default="")

    options, args = parser.parse_args()

    url = options.url.split(',')
    job_type = options.type

    build_generic_model(job_type, url)

    

















    # builder = ArchBuilder("v1")

    # good_models = []
    # bad_models = []

    # for arch in range(1, 10):

    #     fashion_mnist = keras.datasets.fashion_mnist
    #     (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    #     neural_arch = builder.build_random_arch("dense{}".format(arch), (28, 28), 10, keras.activations.softmax)

    #     neural_arch.set_train_data(train_images, train_labels)
    #     neural_arch.set_test_data(test_images, test_labels)

    #     log_dir = 'models/fit/{}'.format(arch)
    #     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    #     neural_arch.train(callbacks=[tensorboard_callback])

    #     neural_arch.evaluate()

    #     if neural_arch.accuracy > 0.5:
    #         good_models.append(neural_arch)
    #         neural_arch.model.save('{}{}.h5'.format(neural_arch.accuracy, neural_arch.identifier))
    #     else:
    #         bad_models.append(neural_arch)


    # print("Final statistics;\n{} good models\n{} bad models".format(len(good_models), len(bad_models)))



    