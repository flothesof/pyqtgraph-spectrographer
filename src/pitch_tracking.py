"""
This module implements pitch-tracking algorithms, i.e. algorithms that transform
a sound wave or a frequency spectrum to a fundamental frequency, a pitch, as might
be perceived by the human ear.
"""
import numpy as np
import scipy.signal

TEST_SAMPLE_FREQ = 22050
TEST_FREQS = [82.1, 164.2, 440.5, 880.8]


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


def frequency_domain_f0_cepstrum(spectrum, sampling_frequency):
    """
    Cepstrum based f0 identification.

    For more information see https://en.wikipedia.org/wiki/Cepstrum.

    :param spectrum: complex ndarray holding Fourier spectrum
    :param sampling_frequency: float
    :return:
        - f0 - float, f0 frequency, in Hz
    """
    log_spectrum = np.log(np.abs(spectrum) + 1e-12)
    cepstrum = np.fft.ifft(log_spectrum)
    abs_cepstrum = np.abs(cepstrum)
    maxi = scipy.signal.argrelmax(abs_cepstrum)
    f0_index = maxi[0][np.argmax(abs_cepstrum[maxi])]
    # assuming the spectrum was computed using rfftfreq
    df = sampling_frequency / 2 / spectrum.size
    quefrency_vector = np.fft.fftfreq(n=log_spectrum.size, d=df)
    f0 = 1 / quefrency_vector[f0_index]
    return f0


def test_autocorrelation():
    print('testing time_domain_f0_autocorrelation')
    for freq in TEST_FREQS:
        waveform = make_sine_wave(freq, TEST_SAMPLE_FREQ, 2048)
        f0 = time_domain_f0_autocorrelation(waveform, TEST_SAMPLE_FREQ)
        print(f'expected f0: {freq} - computed: {f0:.2f}')


def test_cepstrum():
    print('testing frequency_domain_f0_cepstrum')
    for freq in TEST_FREQS:
        waveform = make_harmonic_wave(freq, TEST_SAMPLE_FREQ, 2048, 3)
        spectrum = np.fft.rfft((waveform - waveform.mean()) * np.hamming(waveform.size))
        f0 = frequency_domain_f0_cepstrum(spectrum, TEST_SAMPLE_FREQ)
        print(f'expected f0: {freq} - computed: {f0:.2f}')


if __name__ == '__main__':
    test_autocorrelation()
    test_cepstrum()

    freq = TEST_FREQS[2]
    waveform = make_harmonic_wave(freq, TEST_SAMPLE_FREQ, 2048, 30)
    waveform_before_spectrum = (waveform - waveform.mean()) * np.hamming(waveform.size)
    spectrum = np.fft.rfft(waveform_before_spectrum)
    dt = 1 / TEST_SAMPLE_FREQ
    time_vector = np.arange(waveform.size) * dt
    freq_vector = np.fft.rfftfreq(waveform_before_spectrum.size, d=dt)
    log_spectrum = np.log(np.abs(spectrum) + 1e-12)
    df = freq_vector[1] - freq_vector[0]
    quefrency_vector = np.fft.fftfreq(n=log_spectrum.size, d=df)
    cepstrum = np.fft.ifft(log_spectrum * np.hamming(log_spectrum.size))
    maxi = scipy.signal.argrelmax(np.abs(cepstrum))
    abs_cepstrum = np.abs(cepstrum)
    f0_index = maxi[0][np.argmax(abs_cepstrum[maxi])]
    f0 = f0_index / abs_cepstrum.size * TEST_SAMPLE_FREQ
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=4)
    ax[0].plot(time_vector, waveform)
    ax[0].set_title(f'f0 = {freq}')
    ax[0].set_xlabel('time (s)')
    ax[1].plot(freq_vector, np.abs(spectrum))
    ax[1].set_xlabel('frequency (Hz)')
    ax[2].plot(freq_vector, log_spectrum)
    ax[2].set_xlabel('frequency (Hz)')
    ax[3].plot(quefrency_vector, np.abs(cepstrum))
    ax[3].plot(quefrency_vector[maxi[0]], np.abs(cepstrum)[maxi], 'o')
    plt.tight_layout()
