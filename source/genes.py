import tensorflow as tf
from tensorflow.python import keras
import random


class Genes:

    def __init__(self, layer_dict, hyperparamters):
        self.layer_dict = layer_dict
        self.num_hidden_layers = len(self.layer_dict.keys())
        self.hyperparamters = hyperparamters

    def __str__(self):
        print("No. hidden layers: {} \n Layer information: {}".format(self.num_hidden_layers, self.layer_dict))

    def get_activation_of_layer(self, layer):
        return self.layer_dict[layer]['activation']

    def get_size_of_layer(self, layer):
        return self.layer_dict[layer]['size']

    def get_type_of_layer(self, layer):
        return self.layer_dict[layer]['type']

    def encode_genes_as_vector(self):
        pass

    def decode_vector_as_genes(self):
        pass
