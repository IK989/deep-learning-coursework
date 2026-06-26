import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    """
    Fully Connected Layer (Dense Layer) that performs linear transformation.
    Output = Input @ Weights + Bias
    """
    
    def __init__(self, input_size, output_size):
        """
        Constructor for FullyConnected layer.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
        """
        super().__init__()
        self.trainable = True
        
        # Initialize weights uniformly random in range [0, 1)
        # Shape: (input_size + 1, output_size) - extra row for bias
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        
        # Protected member for optimizer
        self._optimizer = None
        
        # Store input for backward pass
        self._input_tensor = None
        
        # Store gradient with respect to weights
        self._gradient_weights = None

        self.fan_in = input_size
        self.fan_out = output_size
    
    @property
    def optimizer(self):
        """Getter for optimizer property."""
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        """Setter for optimizer property."""
        self._optimizer = optimizer
    
    @property
    def gradient_weights(self):
        """Getter for gradient with respect to weights."""
        return self._gradient_weights
    
    def forward(self, input_tensor):
        """
        Forward pass of the fully connected layer.
        
        Args:
            input_tensor: Input data of shape (batch_size, input_size)
        
        Returns:
            output_tensor: Output data of shape (batch_size, output_size)
        """
        # Store input tensor for backward pass (with bias term)
        # Add column of ones for bias
        batch_size = input_tensor.shape[0]
        self._input_tensor = np.concatenate([input_tensor, np.ones((batch_size, 1))], axis=1)
        
        # Compute output: X @ W
        output_tensor = self._input_tensor @ self.weights
        
        return output_tensor
    
    def backward(self, error_tensor):
        """
        Backward pass of the fully connected layer.
        
        Args:
            error_tensor: Gradient from the next layer, shape (batch_size, output_size)
        
        Returns:
            error_tensor_previous: Gradient for previous layer, shape (batch_size, input_size)
        """
        # Calculate gradient with respect to weights
        # Gradient: X^T @ error_tensor
        self._gradient_weights = self._input_tensor.T @ error_tensor
        
        # Calculate error tensor for previous layer
        # error_previous = error_tensor @ W^T
        error_tensor_previous = error_tensor @ self.weights.T
        
        # Remove the bias column from error tensor (last column)
        error_tensor_previous = error_tensor_previous[:, :-1]
        
        # Update weights using optimizer if available
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        
        return error_tensor_previous

    def initialize(self, weights_initializer, bias_initializer):
        weights_shape = (self.fan_in + 1, self.fan_out)
        # Number of biases = number of output neurons
        bias_shape = self.fan_out
        self.weights = weights_initializer.initialize(weights_shape, self.fan_in, self.fan_out)
        self.weights[-1, :] = bias_initializer.initialize(bias_shape, self.fan_in, self.fan_out)