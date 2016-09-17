from collections import namedtuple
from copy import deepcopy
from random import shuffle

from DA.Constructor import construct
from DA.InitParams import initialize
from DA.Parameters import Parameters
from SoundPreprocess.Preprocessing import ParsedSound
from Utils.Wrappers import timing

Batch = namedtuple('Batch', ['input_real', 'input_imag', 'target_real', 'target_imag'])
NUM_RETRY = 100
NUM_EPOCH = 1000
TRAIN_SET_SIZE = 8


@timing
def create_batches(learn_1: ParsedSound, learn_2: ParsedSound):
    size = len(learn_1.sound)
    return [Batch(learn_1.sound[i].real, learn_1.sound[i].imagine,
                  learn_2.sound[i].real, learn_2.sound[i].imagine)
            for i in range(size)]


@timing
def epoch(batches, train_set_size, train_net, valid_net):
    shuffle(batches)
    train_set = batches[:train_set_size]
    valid_set = batches[train_set_size + 1:]
    valid_error = 0
    train_error = 0
    train_size = len(train_set)
    valid_size = len(valid_set)

    for batch in train_set:
        train_error += train_net(batch)
    train_error /= train_size

    for batch in valid_set:
        valid_error += valid_net(batch)
    valid_error /= valid_size

    return train_error, valid_error


@timing
def retry(batches, params, train_set_size, train_net, valid_net):
    params = initialize(params)
    for ep in range(NUM_EPOCH):
        err = epoch(batches, train_set_size, train_net, valid_net)
        print(err)


@timing
def train(params: Parameters, learn_1: ParsedSound, learn_2: ParsedSound):
    batches = create_batches(learn_1, learn_2)
    train_set_size = len(batches) // 10 * TRAIN_SET_SIZE
    params = initialize(params)
    train_net = construct(params, True, False)
    valid_net = construct(params, True, True)
    for ret in range(NUM_RETRY):
        retry(batches, params, train_set_size, train_net, valid_net)
