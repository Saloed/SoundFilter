import theano.tensor as T
from lasagne.updates import nesterov_momentum
from theano import function
from lasagne.layers import ConcatLayer, InputLayer, DenseLayer, get_output, get_all_params
from lasagne.nonlinearities import tanh
from DA.Parameters import Parameters
from Utils.Wrappers import timing


@timing
def construct(params: Parameters, is_learn, is_validation):
    real_input = T.fvector('r_in')
    imag_input = T.fvector('i_in')
    real_target = T.fvector('r_tar')
    imag_target = T.fvector('i_tar')

    r_in = InputLayer([params.in_out_size], real_input, 'r_in')
    i_in = InputLayer([params.in_out_size], imag_input, 'i_in')

    r_in_layer = DenseLayer(r_in, params.in_out_size, nonlinearity=tanh, name='r_in_layer')
    i_in_layer = DenseLayer(i_in, params.in_out_size, nonlinearity=tanh, name='i_in_layer')

    c_layer = ConcatLayer([r_in_layer, i_in_layer], axis=0)

    h_layer = DenseLayer(c_layer, params.hidden_size, nonlinearity=tanh, name='h_layer')

    r_out_layer = DenseLayer(h_layer, params.in_out_size, nonlinearity=tanh, name='i_out_layer')
    i_out_layer = DenseLayer(h_layer, params.in_out_size, nonlinearity=tanh, name='i_out_layer')

    # T.nnet.sigmoid(T.dot(real_input, params.weights['w_r_in']) + params.biases['b_r_in'])
    # T.nnet.sigmoid(T.dot(imag_input, params.weights['w_i_in']) + params.biases['b_i_in'])
    #
    # T.nnet.sigmoid(
    #     T.dot(T.concatenate([r_in_layer, i_in_layer]), params.weights['w_h']) + params.biases['b_h'])
    #
    # T.nnet.sigmoid(T.dot(h_layer, params.weights['w_r_out']) + params.biases['b_r_out'])
    # T.nnet.sigmoid(T.dot(h_layer, params.weights['w_i_out']) + params.biases['b_i_out'])

    r_out, i_out = get_output([r_out_layer, i_out_layer])
    used_params = get_all_params([r_out_layer, i_out_layer])

    cost_r = T.std(r_out - real_target)
    cost_i = T.std(i_out - imag_target)
    cost = T.sqrt(T.sqr(cost_r) + T.sqr(cost_i))

    if is_learn:
        if not is_validation:
            params.params = used_params

            updates = nesterov_momentum(cost, used_params, params.learn_rate)
            # upd_params = []
            # upd_params.extend(params.weights.values())
            # upd_params.extend(params.biases.values())
            # grads = T.grad(cost, upd_params)
            # updates = [
            #     (param, param - params.learn_rate * gparam)
            #     for param, gparam in zip(upd_params, grads)
            #     ]
            return function([real_input, imag_input, real_target, imag_target], outputs=cost, updates=updates)
        else:
            return function([real_input, imag_input, real_target, imag_target], outputs=cost)

    else:
        return function([real_input, imag_input], outputs=[r_out, i_out])
