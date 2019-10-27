# -*- coding: utf-8 -*-
"""
Simple waveform recorder with f0 detection.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from src.microphone import MicrophoneRecorder
from src.pitch_tracking import time_domain_f0_autocorrelation, frequency_domain_f0_cepstrum

CHUNKSIZE = 2048
SAMPLE_RATE = 44100
TIME_VECTOR = np.arange(CHUNKSIZE) / SAMPLE_RATE
N_FFT = 4096
FREQ_VECTOR = np.fft.rfftfreq(N_FFT, d=TIME_VECTOR[1] - TIME_VECTOR[0])
WATERFALL_FRAMES = int(1000 * 2048 // N_FFT)
TIMEOUT = TIME_VECTOR.max()

recorder = MicrophoneRecorder(sample_rate=SAMPLE_RATE, chunksize=CHUNKSIZE)
recorder.start()

app = QtGui.QApplication([])
w = QtGui.QWidget()
hbox = QtGui.QHBoxLayout()
label = QtGui.QLabel('Pitch-tracking algorithm:')
algorithm_choice = QtGui.QComboBox()
algorithm_choice.addItems(['autocorrelation', 'cepstrum'])
hbox.addWidget(label)
hbox.addWidget(algorithm_choice)
hbox.addStretch()


plots_layout = pg.GraphicsLayoutWidget()
waveform_plot = plots_layout.addPlot(title="Waveform")
waveform_plot.showGrid(x=True, y=True)
waveform_plot.enableAutoRange('xy', False)
waveform_plot.setXRange(TIME_VECTOR.min(), TIME_VECTOR.max())
waveform_plot.setYRange(-2 ** 15 + 1, 2 ** 15)
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

plots_layout.nextRow()

fft_plot = plots_layout.addPlot(title='FFT plot')
fft_curve = fft_plot.plot(pen='y')
fft_plot.enableAutoRange('xy', False)
fft_plot.showGrid(x=True, y=True)
fft_plot.setXRange(FREQ_VECTOR.min(), FREQ_VECTOR.max())
fft_plot.setYRange(0, 20 * np.log10(2 ** 14 * CHUNKSIZE))
fft_plot.setLabel('left', "Amplitude", units='A.U.')
fft_plot.setLabel('bottom', "Frequency", units='Hz')
vLine = pg.InfiniteLine(angle=90, movable=False)
fft_plot.addItem(vLine, ignoreBounds=True)
label = pg.TextItem(anchor=(0, 1))
fft_plot.addItem(label)


def update_fft():
    global data, fft_curve, fft_plot, vLine
    if data.max() > 1:
        windowed_data = np.hanning(data.size) * (data - data.mean())
        X = np.abs(np.fft.rfft(windowed_data, n=N_FFT))
        Xlog = 20 * np.log10(X + 1e-12)
        fft_curve.setData(x=FREQ_VECTOR, y=Xlog)
        if algorithm_choice.currentText() == 'autocorrelation':
            f0 = time_domain_f0_autocorrelation(windowed_data, SAMPLE_RATE)
        elif algorithm_choice.currentText() == 'cepstrum':
            f0 = frequency_domain_f0_cepstrum(data, SAMPLE_RATE, 60, 700)
        else:
            print('Something is wrong. Could not perform f0 estimation.')
            f0 = 0.
        vLine.setPos(f0)
        label.setText(f"estimated f0: {f0:.2f} Hz")


timer_fft = QtCore.QTimer()
timer_fft.timeout.connect(update_fft)
timer_fft.start(TIMEOUT)

layout = QtGui.QVBoxLayout()
w.setLayout(layout)
layout.addLayout(hbox)
layout.addWidget(plots_layout)
w.show()

# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
