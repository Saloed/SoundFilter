import numpy as np
from theano import shared

from DA.Parameters import Parameters

randomizer = np.random.RandomState(314)


def initialize(params: Parameters):
    w_in = randomizer.uniform(-10.0, 10.0, params.in_out_size * params.hidden_size)
    w_h = randomizer.uniform(-10.0, 10.0, params.hidden_size * params.hidden_size)
    w_out = randomizer.uniform(-10.0, 10.0, params.in_out_size * params.hidden_size)

    b_in = randomizer.uniform(-10.0, 10.0, params.in_out_size)
    b_h = randomizer.uniform(-10.0, 10.0, params.hidden_size)
    b_out = randomizer.uniform(-10.0, 10.0, params.in_out_size)

    w_in = np.asarray(w_in.reshape((params.hidden_size, params.in_out_size)))
    w_h = np.asarray(w_h.reshape((params.hidden_size, params.hidden_size)))
    w_out = np.asarray(w_out.reshape((params.in_out_size, params.hidden_size)))

    b_in = np.asarray(b_in.reshape((params.in_out_size)))
    b_h = np.asarray(b_h.reshape((params.hidden_size)))
    b_out = np.asarray(b_out.reshape((params.in_out_size)))

    params.weights['w_in'] = shared(w_in, 'w_in')
    params.weights['w_h'] = shared(w_h, 'w_h')
    params.weights['w_out'] = shared(w_out, 'w_out')

    params.biases['b_in'] = shared(b_in, 'b_in')
    params.biases['b_h'] = shared(b_h, 'b_h')
    params.biases['b_out'] = shared(b_out, 'b_out')

    return params
