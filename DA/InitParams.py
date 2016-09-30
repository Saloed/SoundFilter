import numpy as np
from theano import shared
from lasagne.init import GlorotUniform, Constant
from DA.Parameters import Parameters
from Utils.Wrappers import timing

randomizer = np.random.RandomState(314)


def reshape_and_cast(param, shape, dtype='float32'):
    return np.asarray(param.reshape(shape),
                      dtype=dtype)


def rand_param(shape_0, shape_1=1):
    size = shape_0 * shape_1
    param = randomizer.uniform(-1.0, 1.0, size)
    if shape_1 != 1:
        param = reshape_and_cast(param, (shape_0, shape_1))
    else:
        param = reshape_and_cast(param, shape_0)
    return shared(param)


@timing
def initialize(params: Parameters):

    w_h_in = rand_param(params.in_out_size, params.hidden_size)
    w_h_1 = rand_param(params.hidden_size, params.hidden_size)
    w_h_2 = rand_param(params.hidden_size, params.hidden_size)
    w_h_out = rand_param(params.hidden_size, params.hidden_size)
    w_out = rand_param(params.hidden_size, params.in_out_size)

    b_h_in = rand_param(params.hidden_size)
    b_h_1 = rand_param(params.hidden_size)
    b_h_2 = rand_param(params.hidden_size)
    b_h_out = rand_param(params.hidden_size)
    b_out = rand_param(params.in_out_size)

    params.weights['w_h_in'] = w_h_in
    params.weights['w_h_1'] = w_h_1
    params.weights['w_h_2'] = w_h_2
    params.weights['w_h_out'] = w_h_out
    params.weights['w_out'] = w_out

    params.biases['b_h_in'] = b_h_in
    params.biases['b_h_1'] = b_h_1
    params.biases['b_h_2'] = b_h_2
    params.biases['b_h_out'] = b_h_out
    params.biases['b_out'] = b_out

    return params


def reset_params(params: Parameters):
    w_h_in = rand_param(params.in_out_size, params.hidden_size)
    w_h_1 = rand_param(params.hidden_size, params.hidden_size)
    w_h_2 = rand_param(params.hidden_size, params.hidden_size)
    w_h_out = rand_param(params.hidden_size, params.hidden_size)
    w_out = rand_param(params.hidden_size, params.in_out_size)

    b_h_in = rand_param(params.hidden_size)
    b_h_1 = rand_param(params.hidden_size)
    b_h_2 = rand_param(params.hidden_size)
    b_h_out = rand_param(params.hidden_size)
    b_out = rand_param(params.in_out_size)

    params.weights['w_h_in'].set_value(w_h_in.get_value())
    params.weights['w_h_1'].set_value(w_h_1.get_value())
    params.weights['w_h_2'].set_value(w_h_2.get_value())
    params.weights['w_h_out'].set_value(w_h_out.get_value())
    params.weights['w_out'].set_value(w_out.get_value())

    params.biases['b_h_in'].set_value(b_h_in.get_value())
    params.biases['b_h_1'].set_value(b_h_1.get_value())
    params.biases['b_h_2'].set_value(b_h_2.get_value())
    params.biases['b_h_out'].set_value(b_h_out.get_value())
    params.biases['b_out'].set_value(b_out.get_value())

    return params
