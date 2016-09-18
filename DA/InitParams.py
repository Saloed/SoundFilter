import numpy as np
from theano import shared

from DA.Parameters import Parameters
from Utils.Wrappers import timing

randomizer = np.random.RandomState(314)


def reshape_and_cast(param, shape, dtype='float32'):
    return np.asarray(param.reshape(shape),
                      dtype=dtype)


def rand_param(shape_0, shape_1=1):
    size = shape_0 * shape_1
    param = randomizer.uniform(-10.0, 10.0, size)
    if shape_1 != 1:
        param = reshape_and_cast(param, (shape_0, shape_1))
    else:
        param = reshape_and_cast(param, shape_0)
    return param


@timing
def initialize(params: Parameters):
    w_r_in = rand_param(params.in_out_size, params.in_out_size)
    w_i_in = rand_param(params.in_out_size, params.in_out_size)
    w_h = rand_param(params.hidden_size, params.hidden_size)
    w_r_out = rand_param(params.hidden_size, params.in_out_size)
    w_i_out = rand_param(params.hidden_size, params.in_out_size)

    b_r_in = rand_param(params.in_out_size)
    b_i_in = rand_param(params.in_out_size)
    b_h = rand_param(params.hidden_size)
    b_r_out = rand_param(params.in_out_size)
    b_i_out = rand_param(params.in_out_size)

    params.weights['w_r_in'] = shared(w_r_in, 'w_r_in')
    params.weights['w_i_in'] = shared(w_i_in, 'w_i_in')
    params.weights['w_h'] = shared(w_h, 'w_h')
    params.weights['w_r_out'] = shared(w_r_out, 'w_r_out')
    params.weights['w_i_out'] = shared(w_i_out, 'w_i_out')

    params.biases['b_r_in'] = shared(b_r_in, 'b_r_in')
    params.biases['b_i_in'] = shared(b_i_in, 'b_i_in')
    params.biases['b_h'] = shared(b_h, 'b_h')
    params.biases['b_r_out'] = shared(b_r_out, 'b_r_out')
    params.biases['b_i_out'] = shared(b_i_out, 'b_i_out')

    return params


def reinit_params(params: Parameters):
    w_r_in = rand_param(params.in_out_size, params.in_out_size)
    w_i_in = rand_param(params.in_out_size, params.in_out_size)
    w_h = rand_param(params.hidden_size, params.hidden_size)
    w_r_out = rand_param(params.hidden_size, params.in_out_size)
    w_i_out = rand_param(params.hidden_size, params.in_out_size)

    b_r_in = rand_param(params.in_out_size)
    b_i_in = rand_param(params.in_out_size)
    b_h = rand_param(params.hidden_size)
    b_r_out = rand_param(params.in_out_size)
    b_i_out = rand_param(params.in_out_size)

    params.weights['w_r_in'].set_value(w_r_in)
    params.weights['w_i_in'].set_value(w_i_in)
    params.weights['w_h'].set_value(w_h)
    params.weights['w_r_out'].set_value(w_r_out)
    params.weights['w_i_out'].set_value(w_i_out)

    params.biases['b_r_in'].set_value(b_r_in)
    params.biases['b_i_in'].set_value(b_i_in)
    params.biases['b_h'].set_value(b_h)
    params.biases['b_r_out'].set_value(b_r_out)
    params.biases['b_i_out'].set_value(b_i_out)

    return params
