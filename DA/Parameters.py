class Parameters:
    def __init__(self, in_out_size, learn_rate, weights=None, biases=None):
        if weights is None:
            weights = {}
        if biases is None:
            biases = {}

        self.hidden_size = in_out_size * 2
        self.in_out_size = in_out_size
        self.weights = weights
        self.biases = biases
        self.learn_rate = learn_rate

        self.params = None