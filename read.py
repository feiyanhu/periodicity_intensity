import numpy as np
from datetime import datetime,timedelta
import FFT

import matplotlib.pyplot as plt


def processdate(start_time, end_time, data):
    fs = 5.0 #second
    n_perday = 24*3600/fs
    #delta_time = end_time - start_time
    #print start_time,end_time,delta_time
    #print 'check data alignment'
    #print delta_time.total_seconds() / fs
    #print len(data)

    start_time_end = str(start_time.date()+timedelta(days=1))+' 00:00:00'
    start_time_end = datetime.strptime(start_time_end, '%Y-%m-%d %H:%M:%S')
    end_time_start = str(end_time.date())+' 00:00:00'
    end_time_start = datetime.strptime(end_time_start, '%Y-%m-%d %H:%M:%S')

    delta_time = start_time_end - timedelta(seconds=5) - start_time
    #print delta_time.total_seconds()/fs,'cccc'
    start_remove = int(np.around(delta_time.total_seconds()/fs)+1)
    first_day_data = data[0:start_remove]
    first_day_start = start_time
    del data[0:start_remove]
    #print 'now data length',len(data), start_remove,'less'

    delta_time = end_time - end_time_start
    end_remove = int(np.around(delta_time.total_seconds()/fs)+1)
    #print delta_time.total_seconds()/fs,'cccc'
    last_day_data = data[len(data) - end_remove:]
    last_day_start = end_time_start
    del data[len(data) - end_remove:]
    #print 'now data length',len(data), end_remove,'less'

    #tt = end_time_start - timedelta(seconds=5) - start_time_end
    #print 'left length', tt.total_seconds()/fs+1

    data = np.asarray(data)
    data = data.reshape((-1,int(n_perday)))
    date_list = [str(start_time.date()+timedelta(days=(1+i)))+' 00:00:00' for i in range(data.shape[0])]
    date_list = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in date_list]

    if len(first_day_data) == int(n_perday):
        data = np.concatenate((np.asarray([first_day_data]),data),axis = 0)
        date_list = [first_day_start] + date_list
        first_day_data = None
        first_day_start = None
    if len(last_day_data) == int(n_perday):
        data = np.concatenate((data, np.asarray([last_day_data])),axis = 0)
        date_list = date_list + [last_day_start]
        last_day_data = None
        last_day_start = None
    #print data.shape, date_list
    #print len(first_day_data), first_day_start
    #print len(last_day_data), last_day_start
    return first_day_start,np.asarray([first_day_data]),date_list,data,last_day_start,np.asarray([last_day_data])

def readone(path):
    f = open(path,'rb')
    count = 0
    all_d = []
    imputation_all = []
    for line in f.readlines():
        line = line.replace('\r\n','')
        if count == 0:
            line = line.split(' - ')
            start_time = line[1]
            start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
            #print start_time
            end_time = line[2]
            end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
        else:
            tmp = line.split(',')
            try:
                all_d.append(float(tmp[0]))
                imputation_all.append(int(tmp[1]))
            except:
                #print line,path
                #print line
                all_d.append(np.NAN)
                imputation_all.append(np.NAN)
        count = count + 1
    #all_d = np.asarray(all_d)
    d1,d2,d3,d4,d5,d6 = processdate(start_time, end_time, all_d)
    return d1,d2,d3,d4,d5,d6,np.asarray(imputation_all,dtype=bool)
    #return all_d

def joint_all_data(first_day_data, data, last_day_data):
    data = np.append(first_day_data, data)
    data = np.append(data, last_day_data)
    return data

if __name__ == "__main__":
    first_day_date,first_day_data,date_list,data,last_day_date,last_day_data, imputation_array = \
        readone('/media/demcare/1.4T_Linux/UKBiobank/sample.csv')
    #print first_day_date,first_day_data.shape
    #print date_list,data.shape
    #print last_day_date,last_day_data.shape
    data = joint_all_data(first_day_data, data, last_day_data)
    #print data
    #d= np.asarray([d],dtype='float64')
    fs = 1/5.0
    FFT.FFT('CPU',data,fs)
    '''x = T.matrix('x', dtype='float64')
    rfft = fft.rfft(x, norm='ortho')
    #rfft = fft.rfft(x)
    f_rfft = theano.function([x], rfft)

    import time
    start = time.time()
    out = f_rfft(d)
    print time.time() - start

    start = time.time()
    f, Pxx_den = signal.periodogram(d, fs)
    print time.time() - start

    start = time.time()
    ot = np.fft.rfft(d)
    print time.time() - start

    start = time.time()
    out = f_rfft(d)
    print time.time() - start

    start = time.time()
    out = f_rfft(d)
    print time.time() - start

    start = time.time()
    out = f_rfft(d)
    print time.time() - start

    c_out = np.asarray(out[0, :, 0] + 1j*out[0, :, 1])
    abs_out = abs(c_out)
    print c_out
    print abs_out

    print ot
    print abs(ot)'''
    #index = np.argsort(Pxx_den)[::-1]
    #print Pxx_den[index[0]],1/f[index[0]]/60.0/60.0
    #print Pxx_den[index[1]],1/f[index[1]]/60.0/60.0
