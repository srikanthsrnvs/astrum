import tensorflow as tf
from tensorflow.python import keras
from genes import Genes


class NeuralArch:

    def __init__(self, model, genes, parents, identifier, generation):
        self.parents = parents
        self.genes = genes
        self.identifier = identifier
        self.generation = generation
        self.model = model
        self.data_provided = False

    def set_train_data(self, train_data, train_labels):
        self._train_data = train_data
        self._train_labels = train_labels

    def set_test_data(self, test_data, test_labels):
        self._test_data = test_data
        self._test_labels = test_labels
        self._data_provided = True

    def train(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], epochs=10, callbacks=None):      
        if self._data_provided:
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            self.model.fit(self._train_data, self._train_labels, epochs=epochs, verbose=1, callbacks=callbacks)
        else:
            print("No training data set. Ensure that training data has been provided.") 
            return

    def evaluate(self):
        stats = self.model.evaluate(self._test_data, self._test_labels, verbose=2)
        self.accuracy = stats[1]
        self.loss = stats[0]
        return stats

    def predict(self, data):
        return self.model.predict(data)
        
    def set_mate(self, mate):
        self.mate = mate

    def set_child(self, child):
        self.child = child
