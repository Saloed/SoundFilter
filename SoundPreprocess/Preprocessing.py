from collections import namedtuple

import numpy as np
from pydub import AudioSegment as AS
from scipy.io import wavfile

from Utils.Wrappers import timing

ParsedSound = namedtuple('ParsedSound', ['sound_name', 'sound', 'freq', 'part_size'])
SOUND_PART_LEN = 50  # 0.02 sec = 20 ms
SOUND_FILES_DIR = 'TestFiles/'
BATCH_SIZE = 10

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
    partition = [sound[i * part_size: (i + 1) * part_size] for i in range(num_parts)]
    return partition, part_size


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


def sub_main():
    learn_1 = "Glockenspiel.wav"
    learn_1_ps = process_wav_file(learn_1)
    write_wav_file(learn_1_ps)


if __name__ == '__main__':
    sub_main()
