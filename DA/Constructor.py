import theano.tensor as T
from lasagne.layers import *
from lasagne.nonlinearities import tanh, identity
from lasagne.updates import nesterov_momentum, adadelta
from theano import function
from theano.printing import pydotprint
from DA.Parameters import Parameters
from SoundPreprocess.Preprocessing import BATCH_SIZE
from Utils.Wrappers import timing


@timing
def construct(params: Parameters, is_learn, is_validation):
    input = T.fmatrix('IN')
    target = T.fmatrix('TAR')

    in_layer = InputLayer([BATCH_SIZE, params.in_out_size], input, 'in')

    h1_layer = DenseLayer(in_layer, params.hidden_size,
                          W=params.weights['w_h_in'], b=params.biases['b_h_in'],
                          nonlinearity=tanh,
                          name='in_hidden')
    hh_layer = DenseLayer(h1_layer, params.hidden_size,
                          W=params.weights['w_h_1'], b=params.biases['b_h_1'],
                          nonlinearity=tanh,
                          name='hidden')
    hh_layer = DenseLayer(hh_layer, params.hidden_size,
                          W=params.weights['w_h_2'], b=params.biases['b_h_2'],
                          nonlinearity=tanh,
                          name='hidden')
    h2_layer = DenseLayer(hh_layer, params.hidden_size,
                          W=params.weights['w_h_out'], b=params.biases['b_h_out'],
                          nonlinearity=tanh,
                          name='out_hidden')
    out_layer = DenseLayer(h2_layer, params.in_out_size,
                           W=params.weights['w_out'], b=params.biases['b_out'],
                           nonlinearity=tanh,
                           name='out')

    out = get_output(out_layer)
    used_params = get_all_params(out_layer)
    cost = T.std(T.sub(out, target))
    # pydotprint(cost, 'cost_graph.png')
    if is_learn:
        if not is_validation:
            updates = adadelta(cost, used_params)
            return function([input, target], outputs=cost, updates=updates)
        else:
            return function([input, target], outputs=cost)

    else:
        return function([input], outputs=out)
