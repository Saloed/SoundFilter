from collections import namedtuple

import numpy as np
from numpy.fft import rfft as fourier_transform, irfft as inv_fourier_transform
from pydub import AudioSegment as AS
from scipy.io import wavfile

from Utils.Wrappers import timing


class ComplexArray:
    def __init__(self, complex_number):
        self.real = np.real(complex_number)
        self.imagine = np.imag(complex_number)

    def to_complex(self):
        return self.real + 1j * self.imagine

    def __str__(self):
        return "{0} + {1}j".format(self.real, self.imagine)


ParsedSound = namedtuple('ParsedSound', ['sound_name', 'sound', 'freq', 'part_size'])
SOUND_FILES_DIR = 'TestFiles/'
SOUND_PART_LEN = 10  # 0.004 sec = 4 ms


def change_to_wav(filename):
    out_file = filename + '.wav'
    in_filename = SOUND_FILES_DIR + filename
    with open(SOUND_FILES_DIR + out_file, 'wb') as fout:
        AS.from_mp3(in_filename).export(format='wav', out_f=fout)
    return out_file


@timing
def make_partition(sound, freq):
    num_parts = freq // (2 * SOUND_PART_LEN)
    sound_len = len(sound)
    part_size = sound_len // num_parts
    new_len = sound_len - sound_len % num_parts
    new_sound = sound[:new_len]
    new_sound = fourier_transform(new_sound, axis=0)
    partition = [ComplexArray(new_sound[i * part_size:(i + 1) * part_size]) for i in range(num_parts)]
    return partition, part_size


@timing
def collapse(sound_parts: list):
    for i, part in enumerate(sound_parts):
        sound_parts[i] = part.to_complex()
    new_sound = np.concatenate(sound_parts)
    new_sound = inv_fourier_transform(new_sound, axis=0)
    return new_sound


def process_wav_file(sound_name):
    freq, sound = wavfile.read(SOUND_FILES_DIR + sound_name)
    sound = sound.T
    new_sound, part_size = make_partition(sound, freq)
    return ParsedSound(sound_name, new_sound, freq, part_size)


def write_wav_file(ps: ParsedSound):
    new_sound = collapse(ps.sound)
    wavfile.write(SOUND_FILES_DIR + 'test_' + ps.sound_name, ps.freq, new_sound)
