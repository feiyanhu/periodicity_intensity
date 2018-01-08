import numpy as np
import test_read as tr
import csv
import Periodicity_Intensity as period

import matplotlib.pyplot as plt
from scipy.stats import pareto

def compute_autocorrelation(data, lag, imputation=None):
    #print data.shape
    #print imputation.shape
    d_original = data[0:-lag]
    d_lagged = data[lag:]
    im_original = imputation[0:-lag].astype(np.int8)
    im_lagged = imputation[lag:].astype(np.int8)
    a = d_original*d_lagged
    b = im_original* im_lagged
    c = a*(1-b)
    return a.shape[0],np.sum(b),np.sum(a), np.sum(c), \
        np.sum(a)/(a.shape[0]+0.0), np.sum(c)/(a.shape[0]-np.sum(b))

def autocorr_non_stationary(data):
    lags = np.arange(0,data.shape[0],1)
    re = []
    for lag in lags:
        d_original = data[0:lags.shape[0]-lag]
        d_lagged = data[lag:]
        re.append(np.corrcoef(d_original,d_lagged)[0,1])
    re = np.asarray(re)
    return re

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    nn_biased = np.ones(x.shape[0])*x.shape[0]
    nn_unbiased = np.arange(x.shape[0],0,-1)
    return result[result.size/2:]/nn_biased, result[result.size/2:]/nn_unbiased

def set_nan_to_zeros(data):
    return np.nan_to_num(data), np.sum(np.isnan(data))

def normalize(data):
    #print np.std(data),np.mean(data)
    if np.std(data) == 0:
        return data
    else:
        data = (data - np.mean(data))/np.std(data)
        return data

def normalize2(data):
    data = (1 - np.exp(data/np.mean(data)))
    return data

def visualize(data,norm_data):
    import matplotlib.pyplot as plt

    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0,data.shape[0],1), data, 'o-')
    plt.title('raw')
    plt.ylabel('acceleration')

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(0,norm_data.shape[0],1), norm_data, '.-')
    plt.xlabel('normalized')
    plt.ylabel('acceleration')

    plt.show()

def remove_last_nan_recursive(data):
    if np.isnan(data[-1]):
        print 'remove last one which is nan'
        #tmp_data = data[0:-2]
        data = remove_last_nan_recursive(data[0:-2])
        return data
    else:
        return data

def remove_last_nans(data):
    while data.shape[0] > 0:
        if np.isnan(data[-1]):
            print 'remove last one which is nan'
            #tmp_data = data[0:-2]
            data = data[0:-2]
        else:
            break
    return data

def is_valid(data, h):
    nan_count = np.sum(np.isnan(data))
    return nan_count, nan_count<720*h
    #return tmp_data

def preprocess(data, h):
    nan_count, isValid = is_valid(data, h)
    if isValid:
        data = np.nan_to_num(data)
        return normalize(data), nan_count, isValid
    if not isValid:
        return data, nan_count, isValid
    #return data

def process(fp_data, fp_imputation):
    fp_data,nan_count = set_nan_to_zeros(fp_data)
    if fp_data.shape[0]%2 != 0:
        fp_data = fp_data[0:fp_data.shape[0]-1]
        fp_imputation = fp_imputation[0:fp_imputation.shape[0]-1]
        print '!!!!!!!!!!!!!!!!!!!!!!!!',fp_data.shape

    #tmp_obj = period.FFT('CPU',fp_data,1/5.0, 1/(24*60*60.0))
    #a0,b0,c0 = tmp_obj.get_results()
    #print a0, np.sqrt(a0), b0

    fp_data = normalize(fp_data)

    d_biased, d_unbiased = autocorr(fp_data)

    tmp_obj = period.FFT('CPU',d_biased,1/5.0, 1/(24*60*60.0))
    b1,b2,b3,b4,b5 = tmp_obj.get_results()
    #print a, np.sqrt(a), b,'!!test', a0/(np.sqrt(a))/(fp_data.shape[0])

    tmp_obj = period.FFT('CPU',d_unbiased,1/5.0, 1/(24*60*60.0))
    ub1,ub2,ub3,ub4,ub5 = tmp_obj.get_results()
    #print a, np.sqrt(a), b

    '''for lag in [720*24,720*12,720*6,720*3]:
        results = compute_autocorrelation(fp_data, lag, fp_imputation)
        print results'''
    return b1,b2,b3,b4,b5,ub1,ub2,ub3,ub4,ub5, nan_count, np.sum(fp_imputation)

def init_csv_writers(namelist):
    file_p = {}
    csv_p = {}
    for name in namelist:
        csv_p[name] = open('/media/demcare/1.4T_Linux/UKBiobank/auto_correlation/lag_'+name+'.csv', 'wb')
        csv_writer = csv.writer(csv_p[name], delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        '''csv_writer.writerow(['name','length','num. of nan','num. of overlapping','num. of imputation data',\
                             'correlation','correlation without imputation data',\
                             'normalized correlation','normalized correlation without imputation data'])'''
        csv_writer.writerow(['ID','length','num. of nan','num. of imputation data',\
                             'PSD of biased','ASD of biased','PSD of detrended biased','MAE of biased','MSE of biased',\
                             'PSD of unbiased','ASD of unbiased','PSD of detrended unbiased','MAE of unbiased','MSE of unbiased'])
        file_p[name] = csv_writer
    return file_p, csv_p

def write(filename, data, imp):
    print filename
    data,_ = set_nan_to_zeros(data)
    path = '/media/demcare/Seagate Expansion Drive/normalized/'
    #path_1 = '/media/demcare/1.4T_Linux/UKBiobank/biased/'
    #path_2 = '/media/demcare/1.4T_Linux/UKBiobank/unbiased/'
    f = open(path+filename+'.csv', 'wb')
    csv_writer = csv.writer(f, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Acceletation. Mean is '+str(np.mean(data))+'. STD is '+str(np.std(data)),'imputed'])
    data = normalize(data)
    data = np.vstack((data,imp))
    data = data.T
    np.savetxt(f, data, fmt=['%.3f','%1d'], delimiter=',')
    #csv_writer.writerows(data)
    f.close()
    #aa = [0.0,1.0,0.0,1.0]
    #np.savetxt('test.csv', aa, fmt='%1d', delimiter=',')



if __name__ == "__main__":
    #file_list = ['circadian_periodiciy']
    #file_p,csv_p = init_csv_writers(file_list)

    for file_name, fp_data, fp_imputation,data_shape,first_day_date,date_list,last_day_date in tr.read_memmap():
        #file_name = file_name.split('_')[0]
        #print file_name, fp_data.shape, np.sum(fp_imputation)
        #write(file_name, fp_data, fp_imputation)
        b1,b2,b3,b4,b5,ub1,ub2,ub3,ub4,ub5,nan_count,imp = process(fp_data, fp_imputation)

        #print p1,p2,e1,e2,p3,p4,e3,e4

        '''for x in file_p:
            tmp = [file_name,fp_data.shape[0],nan_count,imp, b1,b2,b3,b4,b5,ub1,ub2,ub3,ub4,ub5]
            file_p[x].writerow(tmp)

    for name in file_list:
        csv_p[name].close()'''
