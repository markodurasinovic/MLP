import numpy as np

class MLP:
    def __init__(self, hidden_layers, learning_rate=0.0001, momentum=0.5):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.weights = list()

    def fit(self, data, target):
        self.init_weights(data)
        for w in self.weights:
            print(w.shape)
            print(w)

    def init_weights(self, data):
        hidden_len = len(self.hidden_layers)
        input_layer_size = data.shape[1]

        # Set of weights from input layer to first hidden layer
        w = np.ndarray((input_layer_size, self.hidden_layers[0]))
        w.fill(0.5)
        self.weights.append(w)

        # Weights between hidden layers
        i = 0
        while i < hidden_len - 1:
            w = np.ndarray((self.hidden_layers[i], self.hidden_layers[i + 1]))
            w.fill(0.5)
            self.weights.append(w)
            i += 1

        # Weights from last hidden layer to output layer
        w = np.ndarray((self.hidden_layers[hidden_len - 1], 1))
        w.fill(0.5)
        self.weights.append(w)
