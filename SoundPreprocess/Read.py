from pylab import *
from scipy.io import wavfile
from pydub import AudioSegment as AS
import numpy as np
from numpy.fft.fftpack import fft as fourier_transform, ifft as inv_fourier_transform
import time


def change_to_wav(filename):
    out_file = filename + '.wav'
    AS.from_mp3(filename).export(format='wav', out_f=open(out_file, 'wb'))
    return out_file


def timing(f):
    def wrap(*args):
        print('%s function start' % (f.__name__,))
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        elapse = (time2 - time1) * 1000.0
        seconds = elapse / 1000
        millis = elapse % 1000
        print('%s function elapse %i sec %i ms' % (f.__name__, seconds, millis))
        return ret

    return wrap


@timing
def read_wav(file):
    return wavfile.read(file)


@timing
def fourier(sound):
    return fourier_transform(sound)


@timing
def inv_fourier(array):
    return inv_fourier_transform(array)


def process_wav_file(file):
    freq, sound = read_wav(file)

    p = fourier(sound)

    new_sound = inv_fourier(p)

    np.set_printoptions(edgeitems=10000)

    print(sound)
    print(p)
    print(new_sound)


def main():
    file = "C:/Users/admin/PycharmProjects/SoundFilter/SoundPreprocess/TestFiles/guitar_original.mp3.wav"
    if file.endswith('.mp3'):
        file = change_to_wav(file)
    process_wav_file(file)


if __name__ == '__main__':
    main()
