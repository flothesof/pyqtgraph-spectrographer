"""
This file contains the utilities related to the microphone input.
"""
import pyaudio
import numpy as np
import threading
import atexit


class MicrophoneRecorder(object):
    """
    A recorder class that starts listening to the user microphone.

    Class inspired by the SciPy 2015 Vispy talk opening example, see https://github.com/vispy/vispy/pull/928.
    """

    def __init__(self, sample_rate=44100, chunksize=1024):
        self.sample_rate = sample_rate
        self.chunksize = chunksize
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.sample_rate,
                                  input=True,
                                  frames_per_buffer=self.chunksize,
                                  stream_callback=self.new_frame)
        self.lock = threading.Lock()
        self.stop = False
        self.frames = []
        atexit.register(self.close)

    def new_frame(self, data, frame_count, time_info, status):
        data = np.frombuffer(data, 'int16')
        with self.lock:
            self.frames.append(data)
            if self.stop:
                return None, pyaudio.paComplete
        return None, pyaudio.paContinue

    def get_frames(self):
        with self.lock:
            frames = self.frames
            self.frames = []
            return frames

    def start(self):
        self.stream.start_stream()

    def close(self):
        with self.lock:
            self.stop = True
        self.stream.close()
        self.p.terminate()


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    recorder = MicrophoneRecorder()
    recorder.start()
    print('recording')
    time.sleep(1)
    frames = recorder.get_frames()
    recorder.close()
    fig, ax = plt.subplots()
    ax.plot(frames[-1])
    plt.show()

