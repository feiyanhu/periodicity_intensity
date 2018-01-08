import numpy as np
from scipy import signal
import matplotlib.pylab as plt

def find_nearest(array,value,error):
    tmp = np.abs(array-value)
    if np.min(tmp) <= error:
        idx = tmp.argmin()
        return idx, array[idx]
    else:
        print 'No value within error range'
        return None, None

class FFT:
    def __init__(self, method, data, fs, target_fs):
        self.fs = fs
        self.ts = 1/fs
        self.target_fs = target_fs # in seconds
        if method == 'CPU':
            self.__CPU(data)

    def __FFT_remove(self,data):
        tmp_fft = np.fft.rfft(data,norm=True)
        f = np.fft.rfftfreq(data.shape[0], 1/self.fs)
        idx, nnb = find_nearest(f, self.target_fs, 0.00001)
        a = np.arange(0, idx, 1)
        a = np.hstack((a , np.arange(idx+1, tmp_fft.shape[0], 1)))
        np.put(tmp_fft, a, np.asarray([0+0j]*a.shape[0]))
        tmp_o = np.fft.irfft(tmp_fft)
        msa = np.mean(np.abs(data - tmp_o))
        mse = np.mean(np.square(data - tmp_o))
        return msa, mse

    def __CPU_periodogram(self,data):
        tmp_fft = np.fft.rfft(data,norm='ortho')
        f = np.fft.rfftfreq(data.shape[0], 1/self.fs)
        idx, nnb = find_nearest(f, self.target_fs, 0.00001)

        if idx != None:
            p_den = (np.abs(tmp_fft[idx])**2)*2*self.ts # (a^2 + b^2)/N
            amp1 = np.abs(tmp_fft[idx]) # sqrt((a^2 + b^2)/N)
            #amp2 = np.abs(tmp_fft[idx])/np.sqrt(data.size) # sqrt(a^2 + b^2)/N

            f2, Pxx_den = signal.periodogram(data, self.fs,detrend='linear')

            a = np.arange(0, idx, 1)
            a = np.hstack((a , np.arange(idx+1, tmp_fft.shape[0], 1)))
            np.put(tmp_fft, a, np.asarray([0+0j]*a.shape[0]))
            tmp_o = np.fft.irfft(tmp_fft,norm='ortho')
            self.circadian_pattern = tmp_o
            msa = np.mean(np.abs(data - tmp_o))
            mse = np.mean(np.square(data - tmp_o))
            cc = np.corrcoef(tmp_o,data)[0,1]
            c = np.sum(tmp_o*data)
            print cc,c,p_den,amp1,p_den/c
            #i_a = np.argsort(Pxx_den)[::-1]
            #print Pxx_den[i_a[0]], 1/f[i_a[0]]/3600.0
            #idx, nnb = find_nearest(f,self.target_fs,0.0001)
            #print idx, 1/nnb/3600.0, Pxx_den[idx], Pxx_den[idx]**0.5
            plt.plot(data)
            plt.plot(tmp_o)
            plt.show()
            #return Pxx_den[idx]
            return p_den,amp1,Pxx_den[idx],msa, mse
        else: return np.nan,np.nan,np.nan,np.nan,np.nan




    def __CPU(self,data):
        self.psd, self.asd, self.psd_dt, self.mae, self.mse = self.__CPU_periodogram(data)
        #self.msa, self.mse = self.__FFT_remove(data)

    def get_results(self):
        return self.psd, self.asd, self.psd_dt, self.mae, self.mse



