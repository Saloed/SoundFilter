from collections import namedtuple

Batch = namedtuple('Batch', ['input_real', 'input_imag', 'target_real', 'target_imag'])
NUM_RETRY = 100
NUM_EPOCH = 1000
TRAIN_SET_SIZE = 8


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
