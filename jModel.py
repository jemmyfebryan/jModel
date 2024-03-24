import numpy as np
import pickle

class Converter_runtime:
    def __init__(self, layers, weights, hyperparameter_separator):
        self.layers = layers
        self.weights = weights
        self.hyperparameter_separator = hyperparameter_separator

class Model:
    def __init__(self):
        self.layers = []

    def load_model(self, file_path):
        if file_path[-7:] != ".jmodel": raise NameError("Model should ended with .jmodel")
        with open(file_path, 'rb') as file:
            converter = pickle.load(file)
        for ind, layer in enumerate(converter.layers):
            components = layer.split(sep=converter.hyperparameter_separator)
            exec(f"self.layers.append({components[0]}(weights=converter.weights[ind],{','.join(components[1:])}))")

    def predict(self, input_array):
        for layer in self.layers:
            input_array = layer.feed(input_array)
        return input_array

class Activation:
    @staticmethod
    def linear(input_array):
        return input_array
    
    @staticmethod
    def relu(input_array):
        return np.maximum(0, input_array)
    
    @staticmethod
    def sigmoid(input_array):
        return 1 / (1 + np.exp(-input_array))
    
    @staticmethod
    def tanh(input_array):
        return np.tanh(input_array)

class Dense:
    def __init__(self, weights, use_bias=True, activation='linear'):
        self.weights = weights
        self.use_bias = use_bias
        self.activation = getattr(Activation, activation)

    def feed(self, input_array):
        output = np.dot(input_array, self.weights[0])
        if self.use_bias:
            output += self.weights[1]
        return self.activation(output)