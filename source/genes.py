import tensorflow as tf
from tensorflow.python import keras
import random
import numpy as np


class Genes:

    def __init__(self, layer_dict=None, hyperparamters=None, vector=None):
        
        if layer_dict is None and hyperparamters is None:
            self.layer_dict = {}
            layer_sizes         =        vector[0:1,][0]
            layer_activations   =        vector[1:2,][0]
            layer_types         =        vector[2:3,][0]

            for index in range(0, len(layer_sizes)):
                self.layer_dict[index+1] = {'type': layer_types[index], 'activation': layer_activations[index], 'size': layer_sizes[index]}
                self.num_hidden_layers = len(self.layer_dict.keys())
                self.hyperparamters = hyperparamters

        else:
            self.layer_dict = layer_dict
            self.num_hidden_layers = len(self.layer_dict.keys())
            self.hyperparamters = hyperparamters

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
            self.possible_layer_types = [
                'dense',
                'conv_1d_transpose',
                'conv_2d_transpose',
                'conv_3d_transpose',
                'conv_1d_regular',
                'conv_2d_regular',
                'conv_3d_regular',
                'max_pool_1d',
                'max_pool_2d',
                'max_pool_3d',
                'avg_pool_1d',
                'avg_pool_2d',
                'avg_pool_3d'
            ]

    def __str__(self):
        print("No. hidden layers: {} \n Layer information: {}".format(self.num_hidden_layers, self.layer_dict))

    def get_activation_of_layer(self, layer):
        return self.layer_dict[layer]['activation']

    def get_size_of_layer(self, layer):
        return self.layer_dict[layer]['size']

    def get_type_of_layer(self, layer):
        return self.layer_dict[layer]['type']

    def get_encoded_type_of_layer(self, layer):
        return self.possible_layer_types.index(self.layer_dict[layer]['type'])

    def get_encoded_activation_at_layer(self, layer):
        return self.possible_activations.index(self.layer_dict[layer]['activation'])

    def encode_genes_as_vector(self):
        # Encode the genes based on the following vector representation
        # [
        #   [Hidden layer size]
        #   [neurons per layer]
        #   [activations per layer]
        #   [layer types]
        # ]
        vector = np.zeros((4,50))
        
        for layer in self.layer_dict.keys():
            vector[0,layer-1] = self.get_size_of_layer(layer)
            vector[1,layer-1] = self.get_encoded_activation_at_layer(layer)
            vector[2,layer-1] = self.get_encoded_type_of_layer(layer)

        return vector

        
