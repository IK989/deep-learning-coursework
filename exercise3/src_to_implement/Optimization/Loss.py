import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        epsilon = np.finfo(float).eps 
        return np.sum(-np.log(prediction_tensor[label_tensor == 1] + epsilon))

    def backward(self, label_tensor):
        epsilon = np.finfo(float).eps
        #Negative gradient means: Increase this probability next time!
        #We want the probability of the correct class to go UP. So gradient is negative.
        return -label_tensor / (self.prediction_tensor + epsilon)