import theano.tensor as T
from lasagne.layers import *
from lasagne.nonlinearities import tanh, identity
from lasagne.updates import nesterov_momentum, adadelta
from theano import function
from theano.printing import pydotprint
from DA.Parameters import Parameters
from Utils.Wrappers import timing


@timing
def construct(params: Parameters, is_learn, is_validation):
    input = T.fvector('IN')
    target = T.fvector('TAR')

    in_layer = InputLayer([params.in_out_size], input, 'in')

    h1_layer = DenseLayer(in_layer, params.hidden_size,
                          W=params.weights['w_h_in'], b=params.biases['b_h_in'],
                          nonlinearity=tanh,
                          name='in_hidden')
    h2_layer = DenseLayer(h1_layer, params.hidden_size,
                          W=params.weights['w_h_out'], b=params.biases['b_h_out'],
                          nonlinearity=tanh,
                          name='out_hidden')
    out_layer = DenseLayer(h2_layer, params.in_out_size,
                           W=params.weights['w_out'], b=params.biases['b_out'],
                           nonlinearity=tanh,
                           name='out')

    # T.nnet.sigmoid(T.dot(real_input, params.weights['w_r_in']) + params.biases['b_r_in'])
    # T.nnet.sigmoid(T.dot(imag_input, params.weights['w_i_in']) + params.biases['b_i_in'])
    #
    # T.nnet.sigmoid(
    #     T.dot(T.concatenate([r_in_layer, i_in_layer]), params.weights['w_h']) + params.biases['b_h'])
    #
    # T.nnet.sigmoid(T.dot(h_layer, params.weights['w_r_out']) + params.biases['b_r_out'])
    # T.nnet.sigmoid(T.dot(h_layer, params.weights['w_i_out']) + params.biases['b_i_out'])

    out = get_output(out_layer)
    used_params = get_all_params(out_layer)
    cost = T.std(out - target)
    # pydotprint(cost, 'cost_graph.png')
    if is_learn:
        if not is_validation:
            updates = adadelta(cost, used_params)
            # upd_params = []
            # upd_params.extend(params.weights.values())
            # upd_params.extend(params.biases.values())
            # grads = T.grad(cost, upd_params)
            # updates = [
            #     (param, param - params.learn_rate * gparam)
            #     for param, gparam in zip(upd_params, grads)
            #     ]
            return function([input, target], outputs=cost, updates=updates)
        else:
            return function([input, target], outputs=cost)

    else:
        return function([input], outputs=out)
