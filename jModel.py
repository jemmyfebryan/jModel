import numpy as np
import pickle
import logging

def error_handler(message, e = None, type_e = None):
    if type_e: raise type_e(message)
    raise type(e)(message)

class Converter_runtime:
    def __init__(self, layers, weights, hyperparameter_separator, version):
        self.layers = layers
        self.weights = weights
        self.hyperparameter_separator = hyperparameter_separator
        self.version = version

class Model:
    version = "0.2.1"

    def __init__(self):
        self.layers = []

    def load_model(self, file_path):
        if file_path[-7:] != ".jmodel": error_handler("Model should ended with .jmodel", type_e=NameError)
        # Loading Model
        try:
            with open(file_path, 'rb') as file:
                converter = pickle.load(file)
        except Exception as e:
            error_message = f"Error when loading the model, {e}"
            error_handler(error_message, e)
        # Loading Layers
        try:
            for ind, layer in enumerate(converter.layers):
                components = layer.split(sep=converter.hyperparameter_separator)
                exec(f"self.layers.append({components[0]}(weights=converter.weights[ind],{','.join(components[1:])}))")
        except Exception as e:
            error_message = f"Error when loading the layers, {e}, make sure ModelVersion and RuntimeVersion is compatible, ModelVersion=={converter.version} & RuntimeVersion=={self.version}"
            error_handler(error_message, e)
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
    
    @staticmethod
    def softmax(input_array, axis=-1):
        exp_values = np.exp(input_array - np.max(input_array, axis=axis, keepdims=True))
        return exp_values / np.sum(exp_values, axis=axis, keepdims=True)
    
    @staticmethod
    def constrained_softmax(x, axis=-1, min_mask=None, max_mask=None, tol=1e-8):
        x = np.array(x, dtype=np.float64)
        shape = x.shape
        n = shape[axis]

        # Broadcast masks if not provided
        if max_mask is None:
            max_mask = np.ones_like(x)
        else:
            max_mask = np.broadcast_to(np.array(max_mask, dtype=np.float64), x.shape)

        if min_mask is None:
            min_mask = np.zeros_like(x)
        else:
            min_mask = np.broadcast_to(np.array(min_mask, dtype=np.float64), x.shape)
            
        if np.any(min_mask > max_mask):
            raise ValueError("min_mask cannot exceed max_mask")

        # Sum checks
        total_min = np.sum(min_mask, axis=axis)
        total_max = np.sum(max_mask, axis=axis)

        if np.any(total_min > 1.0 + tol):
            raise ValueError("min_mask total exceeds 1.0")
        if np.any(total_max < 1.0 - tol):
            raise ValueError("max_mask total less than 1.0")

        # Softmax
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        softmax_weights = e_x / np.sum(e_x, axis=axis, keepdims=True)

        # Clip to min/max masks
        clipped = np.clip(softmax_weights, min_mask, max_mask)

        # Mask of clipped elements
        clipped_mask = (softmax_weights <= min_mask) | (softmax_weights >= max_mask)
        unclipped_mask = ~clipped_mask

        # Get sums
        clipped_sum = np.sum(clipped * clipped_mask, axis=axis, keepdims=True)
        unclipped_sum = np.sum(softmax_weights * unclipped_mask, axis=axis, keepdims=True)

        # Reweight the unclipped values
        scaled_unclipped = np.zeros_like(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale_factor = (1 - clipped_sum) / (unclipped_sum + tol)
            scale_factor = np.where(unclipped_sum > 0, scale_factor, 0)
            scaled_unclipped = softmax_weights * unclipped_mask * scale_factor

        # Combine clipped and rescaled values
        final_result = clipped_mask * clipped + scaled_unclipped

        return final_result

class AveragePooling2D:
    def __init__(self, weights, pool_size, strides, padding='valid'):
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def feed(self, input_array):
        batch_size, input_height, input_width, num_channels = input_array.shape
        pool_height, pool_width = self.pool_size
        stride_y, stride_x = self.strides

        # Padding if necessary
        if self.padding == 'same':
            pad_along_height = max((input_height - 1) // stride_y, 0) * stride_y + pool_height - input_height
            pad_along_width = max((input_width - 1) // stride_x, 0) * stride_x + pool_width - input_width
            
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left
            
            input_array = np.pad(input_array, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=np.nan)
            
            # Update input dimensions after padding
            input_height, input_width, _ = input_array.shape[1:]

        # Calculate output dimensions
        output_height = (input_height - pool_height) // stride_y + 1
        output_width = (input_width - pool_width) // stride_x + 1

        # Initialize output array
        output_array = np.zeros((batch_size, output_height, output_width, num_channels))

        # Perform average pooling using numpy's array operations
        for i in range(output_height):
            for j in range(output_width):
                output_array[:, i, j, :] = np.nanmean(
                    input_array[:, i*stride_y:i*stride_y+pool_height, j*stride_x:j*stride_x+pool_width, :],
                    axis=(1, 2)
                )
        return output_array


class Reshape:
    def __init__(self, weights, target_shape):
        self.target_shape = target_shape

    def feed(self, input_array):
        batch_size = input_array.shape[0]
        reshaped_array = input_array.reshape((batch_size,) + self.target_shape)
        return reshaped_array
    
class Flatten:
    def __init__(self, weights):
        pass

    def feed(self, input_array):
        new_shape = (input_array.shape[0], -1)
        return input_array.reshape(new_shape)

class Dense:
    def __init__(self, weights, use_bias=True, activation='linear'):
        self.weights = weights
        self.activation = getattr(Activation, activation)
        self.use_bias = use_bias

    def feed(self, input_array):
        # Ensure that the input_array has at least 2 dimensions
        if len(input_array.shape) < 2:
            input_array = np.expand_dims(input_array, axis=0)

        # Calculate the output
        output = np.dot(input_array, self.weights[0])

        if self.use_bias:
            # Expand the bias to match the batch size
            bias = np.tile(self.weights[1], (input_array.shape[0], 1))
            output += bias

        return self.activation(output)
    
class SimpleRNN:
    def __init__(self, weights, units, activation='tanh', use_bias=True, return_sequences=False):
        self.weights = weights
        self.units = int(units)
        self.activation = getattr(Activation, activation)
        self.use_bias = use_bias
        self.return_sequences = return_sequences

    def feed(self, input_array):
        # Initialize weights and biases
        hidden_dim = self.units

        # Input-to-hidden weights
        w_ih = self.weights[0]
        # Hidden-to-hidden weights
        w_hh = self.weights[1]
        # Biases
        if not self.use_bias:
            b = np.zeros(hidden_dim)
        else:
            b = self.weights[2]

        seq_len = input_array.shape[1]

        hiddens = []
        h = np.zeros((input_array.shape[0], hidden_dim))  # Hidden state initialization

        for t in range(seq_len):
            x_t = input_array[:, t, :]

            # SimpleRNN cell operations
            h = self.activation(np.dot(x_t, w_ih) + np.dot(h, w_hh) + b)

            hiddens.append(h)

        if self.return_sequences:
            output = np.array(hiddens).transpose(1, 0, 2)
        else:
            output = h

        return output

class LSTM:
    def __init__(self, weights, units, activation='tanh', use_bias=True, return_sequences=False):
        self.weights = weights
        self.units = int(units)
        self.activation = getattr(Activation, activation)
        self.use_bias = use_bias
        self.return_sequences = return_sequences

    def feed(self, input_array):
        # Initialize weights and biases
        hidden_dim = self.units

        # Input-to-hidden weights
        w_ih = self.weights[0]
        # Hidden-to-hidden weights
        w_hh = self.weights[1]
        # Biases
        if not self.use_bias:
            b = np.zeros(4 * hidden_dim)
        else:
            b = self.weights[2]

        seq_len = input_array.shape[1]

        hiddens = []
        c = np.zeros((input_array.shape[0], hidden_dim))  # Cell state initialization
        h = np.zeros((input_array.shape[0], hidden_dim))  # Hidden state initialization

        for t in range(seq_len):
            x_t = input_array[:, t, :]

            # LSTM cell operations
            gates = np.dot(x_t, w_ih) + np.dot(h, w_hh) + b
            input_gate = Activation.sigmoid(gates[:, :hidden_dim])
            forget_gate = Activation.sigmoid(gates[:, hidden_dim:2*hidden_dim])
            candidate_cell = self.activation(gates[:, 2*hidden_dim:3*hidden_dim])
            output_gate = Activation.sigmoid(gates[:, 3*hidden_dim:])

            c = forget_gate * c + input_gate * candidate_cell
            h = output_gate * self.activation(c)

            hiddens.append(h)

        if self.return_sequences:
            output = np.array(hiddens).transpose(1, 0, 2)
        else:
            output = h

        return output