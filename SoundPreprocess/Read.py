import numpy as np
from numpy.fft import rfft as fourier_transform, irfft as inv_fourier_transform
from pydub import AudioSegment as AS
from scipy.io import wavfile
from collections import namedtuple
from SoundPreprocess.Utils.Wrappers import timing

ParsedSound = namedtuple('ParsedSound', ['sound_name', 'sound', 'freq'])
SOUND_FILES_DIR = 'SoundPreprocess/TestFiles/'
SOUND_PART_LEN = 50  # 0.02 sec


def change_to_wav(filename):
    out_file = filename + '.wav'
    in_filename = SOUND_FILES_DIR + filename
    with open(SOUND_FILES_DIR + out_file, 'wb') as fout:
        AS.from_mp3(in_filename).export(format='wav', out_f=fout)
    return out_file


@timing
def make_partition(sound, freq):
    num_parts = freq // 2 // SOUND_PART_LEN
    sound_len = len(sound)
    part_size = sound_len // num_parts
    new_len = sound_len - sound_len % num_parts
    new_sound = sound[:new_len]
    new_sound = fourier_transform(new_sound, axis=0)
    partition = [new_sound[i * part_size:(i + 1) * part_size] for i in range(num_parts)]
    return partition


@timing
def collapse(sound_parts):
    new_sound = np.concatenate(sound_parts)
    new_sound = inv_fourier_transform(new_sound, axis=0)
    return new_sound


def process_wav_file(sound_name):
    freq, sound = wavfile.read(SOUND_FILES_DIR + sound_name)
    sound = sound.T
    new_sound = make_partition(sound, freq)
    return ParsedSound(sound_name, new_sound, freq)


def write_wav_file(ps: ParsedSound):
    new_sound = collapse(ps.sound)
    wavfile.write(SOUND_FILES_DIR + 'test_' + ps.sound_name, ps.freq, new_sound)


def main():
    file = "Glockenspiel.wav"
    if file.endswith('.mp3'):
        file = change_to_wav(file)
    ps = process_wav_file(file)
    write_wav_file(ps)


if __name__ == '__main__':
    main()
