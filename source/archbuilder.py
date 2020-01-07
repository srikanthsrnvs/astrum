import tensorflow as tf
from tensorflow.python import keras
import random
from genes import Genes
import numpy as np
from neuralarch import NeuralArch


class ArchBuilder:

    def __init__(self, name, seed=0):

        # Use a seed optionally to generate randomly
        if (seed != 0):
            assert(isinstance(seed, int))
            random.seed(seed)

        self.name = name
        self.possible_activations = [
            keras.activations.relu,
            keras.activations.softmax,
            keras.activations.elu,
            keras.activations.exponential,
            keras.activations.hard_sigmoid,
            keras.activations.linear,
            keras.activations.selu,
            keras.activations.sigmoid,
            keras.activations.tanh
        ]
        self.possible_main_layer_types = [
            'dense',
            'conv_1d',
            'conv_2d',
            'conv_3d',
        ]
        self.possible_conv_layer_types = [
            'transpose'
            'regular'
        ]
        self.possible_post_conv_layers = [
            'max_pool_1d',
            'max_pool_2d',
            'max_pool_3d',
            'avg_pool_1d',
            'avg_pool_2d',
            'avg_pool_3d'
        ]
        self.built_models = []
        self.initial_models = []

    def _build_conv_layer(self, type):
        

    def build_random_arch(self, identifier, input_shape, output_size, output_activation, min_layer_size=2, max_layer_size=1000, plot=False):

        num_hidden_layers = random.randint(1, 2)
        
        layer_dict = {}
        layer_list = [keras.layers.Flatten(input_shape=input_shape)]

        for layer in range(1, num_hidden_layers+1):
            layer_size = random.randint(min_layer_size, max_layer_size)
            layer_activation = random.choice(self.possible_activations)
            layer_type = None

            if layer_dict[-1]['type'] != 'dense':
                layer_type = random.choice(self.possible_post_conv_layers)
            else:
                layer_type = random.choice(self.possible_main_layer_types)

            if layer_type != 'dense':
                layer_subtype = random.choice(self.possible_conv_layer_types)
                layer_type = '{}_{}'.format(layer_type, layer_subtype)

            layer_dict[layer] = {'size': layer_size, 'activation': layer_activation, 'type': layer_type}
            layer_list.append(keras.layers.Dense(layer_size, activation=layer_activation))

        layer_dict[num_hidden_layers+1] = {'size': output_size, 'activation': output_activation, 'type': 'Output'}

        layer_list.append(keras.layers.Dense(output_size, activation=output_activation))

        model = keras.Sequential(layers=layer_list, name=identifier)

        model_genes = Genes(layer_dict, None)

        if plot:
            keras.utils.plot_model(model, to_file='{}.png'.format(identifier), show_shapes=True)

        neural_arch = NeuralArch(model, model_genes, None, identifier, 1)

        self.initial_models.append(neural_arch)
        self.built_models.append(neural_arch)

        return neural_arch


    def build_child_arch(self):
        pass

