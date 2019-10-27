"""
This module implements pitch-tracking algorithms, i.e. algorithms that transform
a sound wave or a frequency spectrum to a fundamental frequency, a pitch, as might
be perceived by the human ear.
"""
import numpy as np
import scipy.signal

TEST_SAMPLE_FREQ = 22050
TEST_FREQS = [82.1, 164.2, 233.5, 335.2, 440.5, 550.8]
FMIN, FMAX = 75, 770


def make_sine_wave(f0, sampling_frequency, frame_size, phase=0):
    """Generates a sine wave of frequency f0.

    :param f0: float, fundamental frequency
    :param sampling_frequency: int, number of samples per second
    :param frame_size: int, number of samples in frame
    :return:
        - waveform - ndarray of waveform
    """
    t = np.arange(frame_size) / sampling_frequency
    return np.sin(2 * np.pi * f0 * t + phase)


def make_harmonic_wave(f0, sampling_frequency, frame_size, n_harmonics=10):
    """Generates a harmonic (multiples of f0) wave of frequency f0.

    :param f0: float, fundamental frequency
    :param sampling_frequency: int, number of samples per second
    :param frame_size: int, number of samples in frame
    :param n_harmonics: int, number of harmonics to add
    :return:
        - waveform - ndarray of waveform
    """
    waveform = np.zeros((frame_size,), dtype=float)
    for f in [f0 * i for i in range(1, n_harmonics + 1)]:
        waveform += f0 / f * make_sine_wave(f, sampling_frequency, frame_size, phase=f)
    return waveform


def time_domain_f0_autocorrelation(waveform, sampling_frequency, fmin=FMIN, fmax=FMAX):
    """
    Computes f0 using time-domain autocorrelation.

    :param waveform: ndarray containing time-domain signal
    :param sampling_frequency: float
    :return:
        - f0 - float, f0 frequency, in Hz
    """
    bins = np.arange(waveform.size) - waveform.size // 2
    corr = np.correlate(waveform, waveform, mode='same')
    # choose valid maxima only from those in the frequency range (fmin-fmax)
    valid = ((1 / bins * sampling_frequency) >= fmin) & ((1 / bins * sampling_frequency) <= fmax)
    valid_bins = bins[valid]
    valid_corr = corr[valid]
    peaks = scipy.signal.argrelmax(valid_corr)
    candidates = valid_bins[peaks]
    # take highest correlation peak candidate
    amps = valid_corr[peaks]
    candidate = candidates[np.argmax(amps)]
    return 1 / candidate * sampling_frequency


def frequency_domain_f0_cepstrum(waveform, sampling_frequency, fmin=FMIN, fmax=FMAX):
    """
    Cepstrum based f0 identification.

    For more information see https://en.wikipedia.org/wiki/Cepstrum.

    :param waveform: ndarray containing time-domain signal
    :param sampling_frequency: float
    :return:
        - f0 - float, f0 frequency, in Hz
    """
    # compute cepstrum
    frame_size = waveform.size
    windowed_signal = np.hamming(frame_size) * waveform
    dt = 1 / sampling_frequency
    freq_vector = np.fft.rfftfreq(frame_size, d=dt)
    X = np.fft.rfft(windowed_signal)
    log_X = np.log(np.abs(X))
    cepstrum = np.fft.rfft(log_X)
    df = freq_vector[1] - freq_vector[0]
    quefrency_vector = np.fft.rfftfreq(log_X.size, df)
    # extract max-peak in cepstrum in valid region
    valid = (quefrency_vector > 1 / fmax) & (quefrency_vector <= 1 / fmin)
    max_quefrency_index = np.argmax(np.abs(cepstrum)[valid])
    f0 = 1 / quefrency_vector[valid][max_quefrency_index]
    return f0


def test_autocorrelation():
    print('testing time_domain_f0_autocorrelation')
    for freq in TEST_FREQS:
        waveform = make_harmonic_wave(freq, TEST_SAMPLE_FREQ, 2048, 40)
        f0 = time_domain_f0_autocorrelation(waveform, TEST_SAMPLE_FREQ)
        print(f'expected f0: {freq} - computed: {f0:.2f}')


def test_cepstrum():
    print('testing frequency_domain_f0_cepstrum')
    for freq in TEST_FREQS:
        waveform = make_harmonic_wave(freq, TEST_SAMPLE_FREQ, 2048, 40)
        f0 = frequency_domain_f0_cepstrum(waveform, TEST_SAMPLE_FREQ)
        print(f'expected f0: {freq} - computed: {f0:.2f}')


if __name__ == '__main__':
    test_autocorrelation()
    test_cepstrum()
