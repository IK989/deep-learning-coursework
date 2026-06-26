import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.prediction = None

    def forward(self, input_tensor):
        shift_input = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_input = np.exp(shift_input)

        sum_exp = np.sum(exp_input, axis=1, keepdims=True)

        self.prediction = exp_input / sum_exp
        return self.prediction

    def backward(self, error_tensor):
        sum_term = np.sum(error_tensor * self.prediction, axis=1, keepdims=True)
        result = self.prediction * (error_tensor - sum_term)
        return result