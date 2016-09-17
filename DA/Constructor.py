import theano.tensor as T
from theano import function

from DA.Parameters import Parameters
from Utils.Wrappers import timing


@timing
def construct(params: Parameters, is_learn, is_validation):
    real_input = T.dvector('r_in')
    imag_input = T.dvector('i_in')
    real_target = T.dvector('r_tar')
    imag_target = T.dvector('i_tar')

    r_in_layer = T.nnet.sigmoid(T.dot(real_input, params.weights['w_r_in']) + params.biases['b_r_in'])
    i_in_layer = T.nnet.sigmoid(T.dot(imag_input, params.weights['w_i_in']) + params.biases['b_i_in'])

    h_layer = T.nnet.sigmoid(
        T.dot(T.concatenate([r_in_layer, i_in_layer]), params.weights['w_h']) + params.biases['b_h'])

    r_out_layer = T.nnet.sigmoid(T.dot(h_layer, params.weights['w_r_out']) + params.biases['b_r_out'])
    i_out_layer = T.nnet.sigmoid(T.dot(h_layer, params.weights['w_i_out']) + params.biases['b_i_out'])

    cost_r = T.std(r_out_layer - real_target)
    cost_i = T.std(i_out_layer - imag_target)
    cost = T.sqrt(T.sqr(cost_r) + T.sqr(cost_i))

    if is_learn:
        if not is_validation:
            upd_params = []
            upd_params.extend(params.weights.values())
            upd_params.extend(params.weights.values())
            grads = T.grad(cost, upd_params)
            updates = [
                (param, param - params.learn_rate * gparam)
                for param, gparam in zip(upd_params, grads)
                ]
            return function([real_input, imag_input, real_target, imag_target], outputs=cost, updates=updates)
        else:
            return function([real_input, imag_input, real_target, imag_target], outputs=cost)
    else:
        return function([real_input, imag_input], outputs=[r_out_layer, i_out_layer])
