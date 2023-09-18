# -*- coding: utf-8 -*-
"""
This is the main file to launch pyqtgraph-spectrographer.
"""

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from src.microphone import MicrophoneRecorder


def generatePgColormap(cm_name):
    """Converts a matplotlib colormap to a pyqtgraph colormap.

    See https://github.com/pyqtgraph/pyqtgraph/issues/561 for source."""
    colormap = plt.get_cmap(cm_name)
    colormap._init()
    lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
    return lut


CHUNKSIZE = 2048
SAMPLE_RATE = 44100
TIME_VECTOR = np.arange(CHUNKSIZE) / SAMPLE_RATE
N_FFT = 4096
FREQ_VECTOR = np.fft.rfftfreq(N_FFT, d=TIME_VECTOR[1] - TIME_VECTOR[0])
WATERFALL_FRAMES = int(1000 * 2048 // N_FFT)
TIMEOUT = int(TIME_VECTOR.max())
EPS = 1e-8


app = pg.mkQApp()

recorder = MicrophoneRecorder(sample_rate=SAMPLE_RATE, chunksize=CHUNKSIZE)
recorder.start()

win = QtWidgets.QMainWindow()
cw = pg.GraphicsLayoutWidget()
win.show()
win.resize(1000, 600)
win.setCentralWidget(cw)
win.setWindowTitle('pyqtgraph spectrographer')


waveform_plot = cw.addPlot(title="Waveform")
waveform_plot.showGrid(x=True, y=True)
waveform_plot.enableAutoRange('xy', False)
waveform_plot.setXRange(TIME_VECTOR.min(), TIME_VECTOR.max())
waveform_plot.setYRange(-2 ** 15, 2 ** 15 - 1)
waveform_plot.setLabel('left', "Microphone signal", units='A.U.')
waveform_plot.setLabel('bottom', "Time", units='s')
curve = waveform_plot.plot(pen='y')


def update_waveform():
    global curve, data, ptr, waveform_plot, recorder
    frames = recorder.get_frames()
    if len(frames) == 0:
        data = np.zeros((recorder.chunksize,), dtype=np.int)
    else:
        data = frames[-1]
        curve.setData(x=TIME_VECTOR, y=data)


timer = QtCore.QTimer()
timer.timeout.connect(update_waveform)
timer.start(TIMEOUT)

fft_plot = cw.addPlot(title='FFT plot')
fft_curve = fft_plot.plot(pen='y')
fft_plot.enableAutoRange('xy', False)
fft_plot.showGrid(x=True, y=True)
fft_plot.setXRange(FREQ_VECTOR.min(), FREQ_VECTOR.max())
fft_plot.setYRange(20 * np.log10(2 ** 11 * CHUNKSIZE) - 100, 20 * np.log10(2 ** 11 * CHUNKSIZE))
fft_plot.setLabel('left', "Amplitude", units='A.U.')
fft_plot.setLabel('bottom', "Frequency", units='Hz')

waterfall_data = deque(maxlen=WATERFALL_FRAMES)


def update_fft():
    global data, fft_curve, fft_plot
    if data.max() > 1:
        X = np.abs(np.fft.rfft(np.hanning(data.size) * data, n=N_FFT))
        magn = 20 * np.log10(X + EPS)
        fft_curve.setData(x=FREQ_VECTOR, y=magn)
        waterfall_data.append(np.log10(X + 1e-12))


timer_fft = QtCore.QTimer()
timer_fft.timeout.connect(update_fft)
timer_fft.start(TIMEOUT)

cw.nextRow()

image_data = np.random.rand(20, 20)
waterfall_plot = cw.addPlot(title='Waterfall plot', colspan=2)
waterfall_plot.setLabel('left', "Frequency", units='Hz')
waterfall_plot.setLabel('bottom', "Time", units='s')
waterfall_plot.setXRange(0, WATERFALL_FRAMES * TIME_VECTOR.max())
waterfall_image = pg.ImageItem()
waterfall_plot.addItem(waterfall_image)
waterfall_image.setImage(image_data)
lut = generatePgColormap('viridis')
waterfall_image.setLookupTable(lut)
# set scale: x in seconds, y in Hz
scale_factor = CHUNKSIZE / SAMPLE_RATE, int(FREQ_VECTOR.max()) * 2. / N_FFT
tr = QtGui.QTransform()
tr.scale(*scale_factor)
waterfall_image.setTransform(tr)
#waterfall_image.scale(scale_factor)


def update_waterfall():
    global waterfall_data, waterfall_image
    arr = np.c_[waterfall_data]
    if arr.size > 0:
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        max = arr.max()
        min = max / 10
        waterfall_image.setImage(arr, levels=(min, max))


timer_waterfall = QtCore.QTimer()
timer_waterfall.timeout.connect(update_waterfall)
timer_waterfall.start(2 * TIMEOUT)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.exec()
