class Parameters:
    def __init__(self, in_out_size, hidden_size, learn_rate, weights: dict = None, biases: dict = None):
        self.hidden_size = hidden_size
        self.in_out_size = in_out_size
        self.weights = weights
        self.biases = biases
        self.learn_rate = learn_rate
