import numpy as np
from Layers.Base import BaseLayer

"""
    Changes the shape of multi dimensional array to 2D array 
    CNN gives multi dimensional array (batch, height, width, channels)
    FC layer receives 2D array (batch, features)
"""
class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.trainable = False

    def  forward(self, input_tensor):
        self.input_shape = input_tensor.shape

    # converts into 1D array of shape row (batch size) and determine the columns (features)
        new_input = np.ravel(input_tensor).reshape(self.input_shape[0], -1)
        return new_input

    def backward(self, error_tensor):
        return np.reshape(error_tensor, self.input_shape)
        
