import numpy as np

class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weights

"""
    Maintains memory of previous gradients and minimize zigzag effect from SGD
"""
class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        #how fast it was going
        self.momentum_rate = momentum_rate
        self.new_gradient = 0
        #self.new_gradient = np.zeros((1,1))

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        # mutiply the old velocity with momentum rate that how much past gradient should matter now.
        self.new_gradient = (self.momentum_rate*(self.new_gradient)) - self.learning_rate*(gradient_tensor)
        #self.new_gradient = (self.momentum_rate*(self.new_gradient))[:,None] - self.learning_rate*(gradient_tensor)
        updated_weight = weight_tensor + self.new_gradient
        return updated_weight


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu # Direction - Smooth movement
        self.rho = rho # Size - Safe step sizes
        self.iter_k = 1
        self.v_k = 0 # mean of gradients
        self.r_k = 0 # mean of squared gradients

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        #first order and second order gradient descent
        self.v_k = (self.mu*(self.v_k)) + (1-self.mu)*gradient_tensor
        self.r_k = (self.rho*(self.r_k)) + (1-self.rho)*(gradient_tensor**2)

        #correct the averages because initially they were 0
        v = self.v_k/(1-(np.power(self.mu,self.iter_k)))
        r = self.r_k/(1-(np.power(self.rho,self.iter_k)))
        self.iter_k = self.iter_k + 1

        #weight update
        updated_weight = weight_tensor - self.learning_rate*v/(np.sqrt(r) + np.finfo(float).eps)

        return updated_weight
