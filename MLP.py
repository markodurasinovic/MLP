import numpy as np


class MLP:
    def __init__(self, hidden_layers, learning_rate=0.0001, momentum=0.5, verbose=False):
        self.verbose = verbose

        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.weights = list()
        self.bias = list()
        self.neurons = list()
        self.error = list()

    def fit(self, data, targets):
        self.init_weights(data.shape[1])
        self.init_neurons()

        err = self.forward_pass_batch(data, targets, self.sigmoid)
        print(err)

        # if self.verbose:
        #     self.log()

    def init_weights(self, input_layer_size):
        hidden_len = len(self.hidden_layers)

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

        self.bias = np.ndarray(len(self.weights))
        self.bias.fill(0.5)

    def init_neurons(self):
        for layer_size in self.hidden_layers:
            n = np.ndarray(layer_size)
            n.fill(0)
            self.neurons.append(n)

    def forward_pass_batch(self, data, targets, activation_func):
        if not self.error:
            self.error = np.ndarray(len(data))
        for i in range(len(data)):
            out = self.forward_pass(data[i], activation_func)
            self.error[i] = targets[i] - out

        mse = 1 / len(self.error) * sum(self.error ** 2)
        return mse

    def forward_pass(self, dp, activation_func):
        hidden_len = len(self.hidden_layers)

        n = dp.dot(self.weights[0]) + 1 * self.bias[0]
        self.neurons[0] = activation_func(n)
        for i in range(1, hidden_len):
            n = self.neurons[i - 1].dot(self.weights[i]) + 1 * self.bias[i]
            self.neurons[i] = activation_func(n)

        output = self.neurons[hidden_len - 1].dot(self.weights[hidden_len]) + 1 * self.bias[hidden_len]
        return activation_func(output)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def log(self):
        self.print_neurons()
        self.print_weights()
        self.print_error()

    def print_weights(self):
        print("===printing weights===")
        for w in self.weights:
            print(w.shape)
            print(w)
        print(f"bias: {self.bias}")

    def print_neurons(self):
        print("===printing neurons===")
        for i in range(len(self.neurons)):
            print(i)
            print(self.neurons[i])

    def print_error(self):
        print("===printing error===")
        print(self.error)
