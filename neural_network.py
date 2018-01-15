"Math module"
import numpy as np

class NeuralNetwork(object):
    """Class for making and training a nerual network.
    Initialize with integers representing the number
    of neurons in each layer."""

    @staticmethod
    def relu(value, deriv=False):
        "Rectified linear unit"
        if deriv:
            value[value <= 0] = 0
            value[value > 0] = 1
            return value
        return np.maximum(value, 0, value)

    @staticmethod
    def sig(value, deriv=False):
        "Sigmoid"
        if deriv:
            return value * (1-value)
        return 1 / (1+np.exp(-value))

    @staticmethod
    def tanh(value, deriv=False):
        "Hyperbolic tangent"
        if deriv:
            return 1 - (value * value)
        return np.tanh(value)

    def __init__(self, *layout, **kwargs):
        self.n_of_synapses = len(layout) - 1
        # Make weights and biases
        np.random.seed(kwargs["seed"] if "seed" in kwargs else 1)
        self.weights = {}
        self.biases = {}
        for i in range(self.n_of_synapses):
            self.weights[i] = 2 * np.random.random((layout[i], layout[i+1])) - 1
            self.biases[i] = np.zeros(layout[i+1])
        # Remember activation functions
        if "activations" in kwargs:
            self.activations = [{
                "relu": self.relu,
                "sigmoid": self.sig,
                "tanh": self.tanh
            }[i] for i in kwargs["activations"]]
        else:
            self.activations = (self.relu,) * (self.n_of_synapses-1) + (self.sig,)

    def forwardprop(self, input_data, **kwargs):
        "Forwardpropagates and returns dict of prediction and layers"
        # Set keyword arguments
        dropout = (kwargs["dropout"] if "dropout" in kwargs else {})

        # Propagate through layers
        layers = {0: input_data}
        for i in range(self.n_of_synapses):
            # Dropout
            if i in dropout:
                layers[i] *= np.random.binomial(
                    [np.ones(layers[i].shape)],
                    dropout[i]
                )[0] * 1/dropout[i]
            # Set new layer
            layers[i+1] = self.activations[i]( \
                np.dot(layers[i], self.weights[i]) + self.biases[i])
        return layers

    def train(self, input_data, labels, iterations, **kwargs):
        """Train neural network using gradient descent.
        train(input, labels, iterations, alpha=learning_rate, beta=momentum)
        Returns list of error history."""
        # Set keyword-arguments
        alpha = (kwargs["alpha"] if "alpha" in kwargs else 0.01)
        beta = (kwargs["beta"] if "beta" in kwargs else 0.9)
        err_interval = (kwargs["err_interval"] if "err_interval" in kwargs else 20)
        dropout = (kwargs["dropout"] if "dropout" in kwargs else {})

        # Start training
        error_history = []
        w_momentums = {}
        b_momentums = {}
        reverse_synapse_indices = list(reversed(range(self.n_of_synapses)))
        for j in range(1, iterations+1):
        # Forwardpropagate
            layers = self.forwardprop(input_data, dropout=dropout)
            pred = layers[self.n_of_synapses]

        # Backpropagate
            d_layers = {self.n_of_synapses: (labels - pred) * \
                self.activations[-1](pred, deriv=True)}
            for i in reverse_synapse_indices:
                # Calulate Slope and Momentum Jump
                # For layer i
                if i > 0:
                    d_layers[i] = np.dot(d_layers[i+1], self.weights[i].T) * \
                        self.activations[i-1](layers[i], deriv=True)
                # For weight i
                slope = alpha * np.dot(layers[i].T, d_layers[i+1])
                w_momentums[i] = beta * (slope + w_momentums.get(i, 0))
                self.weights[i] += slope + w_momentums[i]
                # For bias i
                slope = alpha * np.sum(d_layers[i+1], axis=0)
                b_momentums[i] = beta * (slope + b_momentums.get(i, 0))
                self.biases[i] += slope + b_momentums[i]

            # Remember error
            if j % (iterations // err_interval) == 0:
                error_history.append(np.mean(np.abs(labels - pred)))
        return error_history
