import numpy as np
from scipy import signal

class FFT:
    def __init__(self, method, data, fs):
        self.fs = fs
        if method == 'CPU':
            self.__CPU(data)

    def __CPU(self,data):
        print data
        f, Pxx_den = signal.periodogram(data, self.fs)
