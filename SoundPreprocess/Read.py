from pylab import *
from scipy.io import wavfile
from pydub import AudioSegment as AS
import subprocess
import os


def change_to_wav(filename, out_filename):
    AS.from_mp3(filename).export(format='wav', out_f=open(out_filename, 'wb'))


def process_wav_file(file):
    freq, sound = wavfile.read(file)
    print(sound.shape)
    print(sound.dtype)
    timeArray = arange(0, 100000, 1)
    timeArray = timeArray / freq
    timeArray = timeArray / 1000  # scale to milliseconds

    plot(timeArray, sound[0:100000], color='k')
    ylabel('Amplitude')
    xlabel('Time (ms)')

    s1 = sound
    n = len(s1)
    p = fft(s1)  # take the fourier transform

    nUniquePts = int(ceil((n + 1) / 2.0))
    p = p[0:nUniquePts]
    p = abs(p)

    p = p / float(n)  # scale by the number of points so that
    # the magnitude does not depend on the length
    # of the signal or on its sampling frequency
    p = p ** 2  # square it to get the power

    # multiply by two (see technical document for details)
    # odd nfft excludes Nyquist point
    if n % 2 > 0:  # we've got odd number of points fft
        p[1:len(p)] = p[1:len(p)] * 2
    else:
        p[1:len(p) - 1] = p[1:len(p) - 1] * 2  # we've got even number of points fft

    freqArray = arange(0, nUniquePts, 1.0) * (freq / n)
    plot(freqArray / 1000, 10 * log10(p), color='k')
    xlabel('Frequency (kHz)')
    ylabel('Power (dB)')


def main():
    file = "C:/Users/admin/PycharmProjects/SoundFilter/SoundPreprocess/TestFiles/guitar_orig.mp3"
    out_file = file + '.wav'
    change_to_wav(file, out_file)
    process_wav_file(out_file)


if __name__ == '__main__':
    main()
