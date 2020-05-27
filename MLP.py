import numpy as np


class MLP:
    def __init__(self, hidden_layers, learning_rate=0.00001, momentum=0.5, iterations=1000):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.iterations = iterations

        self.weights = list()
        self.neurons = list()
        self.deltas = list()
        self.bias = list()

    def fit(self, data, targets, batch=True):
        targets = np.array([targets]).T

        self.init_weights(data.shape[1])
        self.init_neurons()
        if batch:
            for i in range(self.iterations):
                self.epoch = i
                data, targets = self.shuffle(data, targets)
                self.train(data, targets)
        else:
            for i in range(self.iterations):
                self.epoch = i
                data, targets = self.shuffle(data, targets)
                for j in range(data.shape[0]):
                    self.train(np.array([data[j]]), targets[j])

    def init_weights(self, input_layer_size):
        hidden_len = len(self.hidden_layers)

        # Set of weights from input layer to first hidden layer
        w = np.random.randn(input_layer_size, self.hidden_layers[0])
        self.weights.append(w)
        self.bias.append(np.random.randn(1, self.hidden_layers[0]))

        # Weights between hidden layers
        i = 0
        while i < hidden_len - 1:
            w = np.random.randn(self.hidden_layers[i], self.hidden_layers[i + 1])
            self.weights.append(w)
            self.bias.append(np.random.randn(1, self.hidden_layers[i + 1]))
            i += 1

        # Weights from last hidden layer to output layer
        w = np.random.randn(self.hidden_layers[hidden_len - 1], 1)
        self.weights.append(w)
        self.bias.append(np.random.randn(1, 1))

    def init_neurons(self):
        for layer_size in self.hidden_layers:
            n = np.ndarray(layer_size)
            n.fill(0)
            self.neurons.append(n)

    def shuffle(self, data, targets):
        # Shuffle data and targets in unison
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)

        return data[indices], targets[indices]

    def train(self, data, targets):
        out = self.forward_pass(data)
        deltas = self.calculate_deltas(targets, out)
        self.update_weights(data, deltas)

    def predict(self, data):
        out = self.forward_pass(data)

        return out

    def forward_pass(self, data):
        hidden_len = len(self.hidden_layers)
        n = data.dot(self.weights[0]) + 1 * self.bias[0]
        self.neurons[0] = self.sigmoid(n)
        for i in range(1, hidden_len):
            n = self.neurons[i - 1].dot(self.weights[i]) + 1 * self.bias[i]
            self.neurons[i] = self.sigmoid(n)

        output = self.neurons[hidden_len - 1].dot(self.weights[hidden_len]) + 1 * self.bias[hidden_len]

        return self.relu(output)

    def calculate_deltas(self, target, output):
        deltas = list()

        mse = np.mean(np.square(target - output))
        if self.epoch == self.iterations - 1:
            res = [(target[i], output[i]) for i in range(target.shape[0])]
            print(res)
        #     # print(target - output)
        print(f"Epoch: {self.epoch}")
        print(f"Loss: {mse}")
        error = output - target

        delta = error * self.relu(output, derivative=True)
        deltas.append(delta)

        for i in reversed(range(1, len(self.weights))):
            error = delta.dot(self.weights[i].T)
            delta = error * self.sigmoid(self.neurons[i - 1], derivative=True)
            deltas.append(delta)

        return list(reversed(deltas))

    def update_weights(self, data, deltas):
        new_weights = self.weights[0] - (self.learning_rate * data.T.dot(deltas[0]))
        self.weights[0] = new_weights

        temp = np.ndarray((1, data.shape[0]))
        temp.fill(1)
        new_weights = self.bias[0] - (self.learning_rate * temp.dot(deltas[0]))
        self.bias[0] = new_weights

        for i in range(1, len(self.weights)):
            new_weights = self.weights[i] - (self.learning_rate * self.neurons[i - 1].T.dot(deltas[i]))
            self.weights[i] = new_weights

            new_weights = self.bias[i] - (self.learning_rate * temp.dot(deltas[i]))
            self.bias[i] = new_weights

    def sigmoid(self, x, derivative=False):
        x = np.clip(x, -500, 500)
        if derivative:
            return x * (1 - x)
        else:
            return 1 / (1 + np.exp(-x))

    def relu(self, x, derivative=False):
        if derivative:
            return np.where(x > 0, 1, 0.01)
        else:
            return np.maximum(0.01 * x, x)
