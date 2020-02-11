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
            'conv_2d'
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

    def _build_conv_layer(self, type, activation, input_dims=None):

        layer = None
        filters = random.randint(1, 256)
        kernel = random.randint(2, 5)

        if type == 'conv_1d':
            if input_dims == None:
                layer = keras.layers.Conv1D(filters, kernel, activation=activation)
            else:
                layer = keras.layers.Conv1D(filters, kernel, activation=activation, input_shape=input_dims)
        elif type == 'conv_2d':
            if input_dims == None:
                layer = keras.layers.Conv2D(filters, kernel, activation=activation)
            else:
                layer = keras.layers.Conv2D(filters, kernel, activation=activation, input_shape=input_dims)
        else:
            if input_dims == None:
                layer = keras.layers.Conv3D(filters, kernel, activation=activation)
            else:
                layer = keras.layers.Conv3D(filters, kernel, activation=activation, input_shape=input_dims)

        return layer

    def _build_post_conv_layer(self, type, pool_size):

        layer = None

        if type == 'max_pool_1d':
            layer = keras.layers.MaxPool1D(pool_size=pool_size)
        elif type == 'max_pool_2d':
            layer = keras.layers.MaxPool2D(pool_size=pool_size)
        elif type == 'max_pool_3d':
            layer = keras.layers.MaxPool3D(pool_size=pool_size)
        elif type == 'avg_pool_1d':
            layer = keras.layers.AvgPool1D(pool_size=pool_size)
        elif type == 'avg_pool_2d':
            layer = keras.layers.AvgPool2D(pool_size=pool_size)
        else:
            layer = keras.layers.AvgPool3D(pool_size=pool_size)

        return layer

    def _decide_conv_input_dims(self, input_shape, type):
        if type == "conv_1d":
            return (input_shape[0]*input_shape[1], )
        elif type == "conv_3d":
            return (input_shape[0], input_shape[1], -1)
        
        return (input_shape[0], input_shape[1], 1)

    def build_random_arch(self, identifier, input_shape, output_size, output_activation, min_layer_size=2, max_layer_size=1000, plot=False):

        num_hidden_layers = random.randint(1, 2)
        
        layer_dict = {}
        layer_list = []

        for layer_num in range(1, num_hidden_layers+1):

            layer_size = random.randint(min_layer_size, max_layer_size)
            layer_activation = random.choice(self.possible_activations)
            layer_type = None
            layer = None

            # If no layers in the net, we are setting the first layer
            if len(layer_dict.keys()) == 0:

                layer_type = random.choice(self.possible_main_layer_types)
                # If we got a conv, then we build the conv layer, else we build a dense layer.
                if layer_type == 'conv_1d' or layer_type == 'conv_2d' or layer_type == 'conv_3d':
                    layer_list.append(keras.layers.Reshape((28, 28, 1), input_shape=input_shape))
                    layer = self._build_conv_layer(layer_type, layer_activation)
                    layer_type = '{}_regular'.format(layer_type)
                else:
                    layer_list.append(keras.layers.Flatten(input_shape=input_shape))
                    layer = keras.layers.Dense(layer_size, activation=layer_activation)

            # If there were layers, check if it was a conv
            else:
                # If it was a conv, choose a pooling layer
                if layer_dict[layer_num-1]["type"] != 'dense' and layer_dict[layer_num-1]["type"] in self.possible_main_layer_types:
                    layer_type = random.choice(self.possible_post_conv_layers)
                    pool_size = random.randint(2, 5)
                    layer = self._build_post_conv_layer(layer_type, pool_size=pool_size)

                #otherwise just choose another main layer
                else:
                    layer_type = random.choice(self.possible_main_layer_types)

                    if layer_type == 'dense':
                        layer = keras.layers.Dense(layer_size, activation=layer_activation)
                    else:
                        layer = self._build_conv_layer(layer_type, layer_activation)
                        layer_type = '{}_regular'.format(layer_type)

            layer_dict[layer_num] = {'size': layer_size, 'activation': layer_activation, 'type': layer_type}
            layer_list.append(layer)

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

