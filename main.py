# -*- coding: utf-8 -*-
"""
This is the main file to launch pyqtgraph-spectrographer.
"""

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from src.microphone import MicrophoneRecorder

CHUNKSIZE = 1024
SAMPLE_RATE = 44100

recorder = MicrophoneRecorder(sample_rate=SAMPLE_RATE, chunksize=CHUNKSIZE)
recorder.start()

win = pg.GraphicsWindow()
win.resize(1000, 600)
win.setWindowTitle('pyqtgraph spectrographer')

waveform_plot = win.addPlot(title="Waveform")
waveform_plot.showGrid(x=True, y=True)
curve = waveform_plot.plot(pen='y')

ptr = 0
TIME_VECTOR = np.arange(CHUNKSIZE) / SAMPLE_RATE


def update_waveform():
    global curve, data, ptr, waveform_plot, recorder
    frames = recorder.get_frames()
    if len(frames) == 0:
        data = np.zeros((recorder.chunksize), dtype=np.int)
    else:
        data = frames[-1]
        curve.setData(x=TIME_VECTOR, y=data)
        if ptr % 100 == 0:
            waveform_plot.enableAutoRange('xy', True)
        else:
            waveform_plot.enableAutoRange('xy', False)
        ptr += 1


timer = QtCore.QTimer()
timer.timeout.connect(update_waveform)
timer.start(50)

win.nextRow()

fft_plot = win.addPlot(title='FFT plot')
fft_curve = fft_plot.plot(pen='y')

N_FFT = 2048
FREQ_VECTOR = np.fft.rfftfreq(N_FFT, d=TIME_VECTOR[1] - TIME_VECTOR[0])

waterfall_data = deque(maxlen=1000)


def update_fft():
    global data, fft_curve, ptr, fft_plot
    if data.max() > 1:
        X = np.abs(np.fft.rfft(data, n=N_FFT))
        fft_curve.setData(x=FREQ_VECTOR, y=X)
        waterfall_data.append(np.log10(X))
        if ptr % 100 == 0:
            fft_plot.enableAutoRange('xy', True)
        else:
            fft_plot.enableAutoRange('xy', False)


timer_fft = QtCore.QTimer()
timer_fft.timeout.connect(update_fft)
timer_fft.start(50)

win.nextRow()

image_data = np.random.rand(20, 20)
waterfall_plot = win.addPlot(title='Waterfall plot')
waterfall_image = pg.ImageItem()
waterfall_plot.addItem(waterfall_image)
waterfall_image.setImage(image_data)

cmap = plt.get_cmap('rainbow')
LUT = pg.makeARGB(np.array([cmap(val) for val in np.linspace(0, 1, num=256)]), levels=(0, 1))


def update_waterfall():
    global waterfall_data, waterfall_image, LUT
    arr = np.c_[waterfall_data]
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    waterfall_image.setImage(arr, levels=(0, np.log10(N_FFT)))


timer_waterfall = QtCore.QTimer()
timer_waterfall.timeout.connect(update_waterfall)
timer_waterfall.start(100)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
