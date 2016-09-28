from collections import namedtuple

import numpy as np
import theano
from numpy.fft import rfft as fourier_transform, irfft as inv_fourier_transform
from pydub import AudioSegment as AS
from scipy.io import wavfile

from Utils.Wrappers import timing

ParsedSound = namedtuple('ParsedSound', ['sound_name', 'sound', 'freq', 'part_size'])
SOUND_FILES_DIR = 'TestFiles/'
SOUND_PART_LEN = 50  # 0.02 sec = 20 ms


def change_to_wav(filename):
    out_file = filename + '.wav'
    in_filename = SOUND_FILES_DIR + filename
    with open(SOUND_FILES_DIR + out_file, 'wb') as fout:
        AS.from_mp3(in_filename).export(format='wav', out_f=fout)
    return out_file


@timing
def make_partition(sound, freq):
    sound_len = len(sound)
    part_size = freq // SOUND_PART_LEN
    num_parts = sound_len // part_size
    new_len = num_parts * part_size
    new_sound = sound[:new_len]
    # todo check this
    new_sound /= 2 ** 15
    return new_sound, part_size


@timing
def collapse(sound_parts: list):
    return np.concatenate(sound_parts)


def process_wav_file(sound_name):
    freq, sound = wavfile.read(SOUND_FILES_DIR + sound_name)
    sound = sound.T
    new_sound, part_size = make_partition(sound, freq)
    return ParsedSound(sound_name, new_sound, freq, part_size)


def write_wav_file(ps: ParsedSound):
    new_sound = collapse(ps.sound)
    wavfile.write(SOUND_FILES_DIR + 'test_' + ps.sound_name, ps.freq, new_sound)
