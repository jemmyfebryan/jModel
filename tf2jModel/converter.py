import tensorflow as tf
import pickle
from jmodel import Converter_runtime

class Converter:
    version = "0.0.1"

    def __init__(self):
        self.hyperparameter_separator = "--"
        self.layer_separator = "##"
        self.layers = []
        self.weights = []

    def layer_name_add(self, layer_name: str, added_string: str):
        if layer_name == "": return added_string
        return layer_name + self.hyperparameter_separator + added_string

    def adapt(self, tf_model: tf.keras.Sequential):
        weights = tf_model.get_weights()
        weight_index = 0
        for layer in tf_model.layers:
            layer_name = ""
            layer_weight = [weights[weight_index]]
            weight_index += 1
            if layer.__class__.__name__ == "Dense":
                layer_name = self.layer_name_add(layer_name, "Dense")
                # Check Bias
                if layer.use_bias == True:
                    layer_name = self.layer_name_add(layer_name, "use_bias=True")
                    layer_weight.append(weights[weight_index])
                    weight_index += 1
                else:
                    layer_name = self.layer_name_add(layer_name, "use_bias=False")
                # Check Activation
                layer_name = self.layer_name_add(layer_name, f"activation='{layer.activation.__name__}'")
            else:
                raise TypeError(f"Layer {layer.__class__.__name__} is not supported for jModel=={Converter.version}")
            self.layers.append(layer_name)
            self.weights.append(layer_weight)
        print("Model successfully adapted to jModel!")

    def convert(self, file_path):
        if file_path[-7:] != ".jmodel": raise NameError("File name must ended with .jmodel")
        with open(file_path, 'wb') as file:
            pickle.dump(Converter_runtime(self.layers, self.weights, self.hyperparameter_separator), file)
        print(f"Model successfully saved at {file_path}")


