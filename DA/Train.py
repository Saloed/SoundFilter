import _pickle as c_pickle
from random import shuffle
import numpy as np
from DA.Constructor import construct
from DA.InitParams import reset_params
from DA.Parameters import Parameters, Batch, TRAIN_SET_SIZE, NUM_RETRY, NUM_EPOCH
from SoundPreprocess.Preprocessing import ParsedSound, BATCH_SIZE
from Utils.Visualizer import new_figure, update_figure
from Utils.Wrappers import timing


@timing
def create_batches(learn_1: ParsedSound, learn_2: ParsedSound):
    size = len(learn_1.sound) // BATCH_SIZE
    batches = [Batch] * size
    for i in range(size):
        s = i * BATCH_SIZE
        e = (i + 1) * BATCH_SIZE
        inp = np.array(learn_1.sound[s:e])
        tar = np.array(learn_2.sound[s:e])
        batches[i] = Batch(inp, tar)
    return batches


def debug_print(i, err):
    if i % 100 == 0:
        print("Batch {0}\terr: {1}".format(i, err))


@timing
def epoch(batches, train_set_size, train_net, valid_net):
    shuffle(batches)
    train_set = batches[:train_set_size]
    valid_set = batches[train_set_size + 1:]
    valid_error = 0
    train_error = 0
    train_size = len(train_set)
    valid_size = len(valid_set)
    for i, batch in enumerate(train_set):
        terr = train_net(batch.input, batch.target)
        train_error += terr
        debug_print(i, terr)
    train_error /= train_size
    print(train_error)
    for i, batch in enumerate(valid_set):
        verr = valid_net(batch.input, batch.target)
        valid_error += verr
        debug_print(i, verr)

    valid_error /= valid_size
    print(valid_error)
    return train_error, valid_error


@timing
def retry(batches, params, retry, train_set_size, train_net, valid_net):
    # params = reset_params(params)
    axis, plot = new_figure(retry)

    for ep in range(NUM_EPOCH):
        terr, verr = epoch(batches, train_set_size, train_net, valid_net)
        update_figure(plot, axis, ep, verr)
        if ep % 100 == 0:
            with open('new_params_r{0}_e{1}'.format(retry, ep), 'wb') as fout:
                c_pickle.dump(params, fout)


@timing
def train(params: Parameters, learn_1: ParsedSound, learn_2: ParsedSound):
    batches = create_batches(learn_1, learn_2)
    train_set_size = len(batches) // 10 * TRAIN_SET_SIZE
    params = reset_params(params)
    train_net = construct(params, True, False)
    valid_net = construct(params, True, True)
    for ret in range(NUM_RETRY):
        retry(batches, params, ret, train_set_size, train_net, valid_net)
