"""
This module implements pitch-tracking algorithms, i.e. algorithms that transform
a sound wave or a frequency spectrum to a fundamental frequency, a pitch, as might
be perceived by the human ear.
"""
import numpy as np
import scipy.signal


def make_sine_wave(f0, sampling_frequency, frame_size):
    """Generates a sine wave of frequency f0.

    :param f0: float, fundamental frequency
    :param sampling_frequency: int, number of samples per second
    :param frame_size: int, number of samples in frame
    :return:
        - waveform - ndarray of waveform
    """
    t = np.arange(frame_size) / sampling_frequency
    return np.sin(2 * np.pi * f0 * t)


def time_domain_f0_autocorrelation(waveform, sampling_frequency):
    """
    Computes f0 using time-domain autocorrelation.

    :param waveform: ndarray containing time-domain signal
    :param sampling_frequency: float
    :return:
        - f0 - float, f0 frequency, in Hz
    """
    bins = np.arange(waveform.size) - waveform.size // 2
    corr = np.correlate(waveform, waveform, mode='same')
    candidates = bins[scipy.signal.argrelmax(corr)]
    candidate = candidates[candidates > 0][0]
    return 1 / candidate * sampling_frequency


if __name__ == '__main__':
    SAMPLE_FREQ = 44100
    for freq in [82.1, 164.2, 440, 880]:
        waveform = make_sine_wave(freq, SAMPLE_FREQ, 2048)
        f0 = time_domain_f0_autocorrelation(waveform, SAMPLE_FREQ)
        print(f'expected f0: {freq} - computed: {f0}')
