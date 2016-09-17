import theano.tensor as T
from theano import function

from DA.Parameters import Parameters
from Utils.Wrappers import timing


@timing
def construct(params: Parameters, is_learn, is_validation):
    input = T.zvector('in')
    target = T.zvector('target')
    in_layer = T.nnet.sigmoid(T.dot(input, params.weights['w_in']) + params.biases['b_in'])
    h_layer = T.nnet.sigmoid(T.dot(in_layer, params.weights['w_h']) + params.biases['b_h'])
    out_layer = T.nnet.sigmoid(T.dot(h_layer, params.weights['w_out']) + params.biases['b_out'])

    cost = T.nnet.categorical_crossentropy(out_layer, target)

    if is_learn:
        if not is_validation:
            upd_params = params.weights.values() + params.weights.values()
            grads = T.grad(cost, upd_params)
            updates = [
                (param, param - params.learn_rate * gparam)
                for param, gparam in zip(upd_params, grads)
                ]
            return function([input, target], outputs=cost, updates=updates)
        else:
            return function([input, target], outputs=cost)
    else:
        return function([input], outputs=out_layer)
