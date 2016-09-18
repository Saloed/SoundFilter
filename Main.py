import gc

import theano

from DA.Parameters import Parameters
from DA.Train import train
from SoundPreprocess.Preprocessing import process_wav_file, write_wav_file

theano.config.mode = 'FAST_RUN'
theano.config.floatX = 'float32'
# theano.config.exception_verbosity = 'high'


def main():
    # garbage collection for lowing memory usage
    gc.enable()

    learn_1 = "Glockenspiel.wav"
    learn_2 = "Xylophone.wav"
    learn_1_ps = process_wav_file(learn_1)
    learn_2_ps = process_wav_file(learn_2)

    if learn_1_ps.part_size != learn_2_ps.part_size:
        raise Exception("Learn sounds frequencies are different")
    if len(learn_1_ps.sound) != len(learn_2_ps.sound):
        raise Exception("Learn sounds lengths are different")

    part_size = learn_1_ps.part_size
    parameters = Parameters(in_out_size=part_size, learn_rate=0.03)
    train(parameters, learn_1_ps, learn_2_ps)

    # write_wav_file(learn_1_ps)


if __name__ == '__main__':
    main()
