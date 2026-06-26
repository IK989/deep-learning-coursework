import numpy as np

class Constant:
    def __init__(self, value=0.1):
        self.value = value

    """
        Mostly for testing, Easy to predict behavior

        Bad for real training (symmetry problem)

        If all weights are the same, all neurons learn the same thing.
    """
    def initialize(self, weights_shape, fan_in, fan_out):
        output = np.full(weights_shape, self.value)
        return output

class UniformRandom:
    def __init__(self):
        pass

    """"
        it allows neurons to learn different things because of random weights
        Variance is not controlled, Causes vanishing/exploding gradients in deep networks
        Good as a baseline  but not for CNNs
    """
    def initialize(self, weights_shape, fan_in, fan_out):
        output = np.random.rand(weights_shape[0], weights_shape[1])
        return output

class Xavier:
    def __init__(self):
        pass

    # In xavier, Signals are preserved evenly in both directions. centered around 0
    # 2nd param tells us , how close of far from 0?
    def initialize(self, weights_shape, fan_in, fan_out):
        var = np.sqrt(2/(fan_out + fan_in))
        output = np.random.normal(0, var, size=weights_shape)
        return output

class He:
    def __init__(self):
        pass

    # Compensate for ReLU’s signal loss, because RELU kills 50% of neurons. it focuses on in,
    # means the kill after RELU , signal strength becomes stable.
    def initialize(self, weights_shape, fan_in, fan_out):
        var = np.sqrt(2/fan_in)
        output = np.random.normal(0, var, size=weights_shape)
        return output

