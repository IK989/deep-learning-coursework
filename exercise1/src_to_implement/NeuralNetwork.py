import copy


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []  # Liste für Loss-Werte pro Iteration
        self.layers = []  # Liste der Layer
        self.data_layer = None
        self.loss_layer = None

        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()

        tensor = self.input_tensor
        for layer in self.layers:
            tensor = layer.forward(tensor)

        loss_val = self.loss_layer.forward(tensor, self.label_tensor)
        return loss_val

    def backward(self):
        error = self.loss_layer.backward(self.label_tensor)

        # Rückwärts durch die Layer iterieren
        for layer in reversed(self.layers):
            error = layer.backward(error)

    def append_layer(self, layer):
        if layer.trainable:
            # Deep Copy des Optimizers für trainierbare Layer
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            loss_val = self.forward()
            self.loss.append(loss_val)
            self.backward()

    def test(self, input_tensor):
        tensor = input_tensor
        for layer in self.layers:
            tensor = layer.forward(tensor)
        return tensor