import read
from os import listdir
from os.path import isfile, join
import numpy as np
import cPickle


def write_data(data, meta_data, path, name):
    data = data.astype(np.float32)
    fp = np.memmap(path+'np_memmap/'+name+'.dat', dtype='float32', mode='w+', shape=data.shape)
    fp[:] = data[:]
    del fp
    cPickle.dump(meta_data,open(path+'memmap_meta/'+name+'.p',mode='wb'))
def write_imputation_data(data, path, name):
    fp = np.memmap(path+'np_imputation/'+name+'.dat', dtype='bool', mode='w+', shape=data.shape)
    fp[:] = data[:]
    del fp

def start_convert():
    mypath = '/media/demcare/1.4T_Linux/UKBiobank/downloads/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and '.csv' in f]
    #onlyfiles = [onlyfiles[0],'1635148_90004_0_0.csv']
    #print len(onlyfiles)
    '''while len(onlyfiles) != 0:
        if onlyfiles[0] != '1635148_90004_0_0.csv':
            del onlyfiles[0]
        else:
            break'''
    for file in onlyfiles:
        print file
        first_day_date,first_day_data,date_list,data,last_day_date,last_day_data, imputation_array = \
        read.readone(mypath+file)
        #data = read.joint_all_data(first_day_data, data, last_day_data)
        #meta_data = [data.shape,first_day_date,date_list,last_day_date]
        #write_data(data,meta_data,'/media/demcare/1.4T_Linux/UKBiobank/',file.replace('.csv',''))
        write_imputation_data(imputation_array,'/media/demcare/1.4T_Linux/UKBiobank/',file.replace('.csv',''))

def read_memmap(onlyfiles=None):
    mypath = '/media/demcare/1.4T_Linux/UKBiobank/downloads/'
    path = '/media/demcare/1.4T_Linux/UKBiobank/'
    if onlyfiles == None:
        onlyfiles = [f.replace('.csv', '') for f in listdir(mypath) if isfile(join(mypath, f)) and '.csv' in f]
    #print onlyfiles
    #onlyfiles = ['3685073_90004_0_0']
    for file_name in onlyfiles:
        #print file_name
        [data_shape,first_day_date,date_list,last_day_date] = \
            cPickle.load(open(path+'memmap_meta/'+file_name+'.p',mode='rb'))
        fp_data = np.memmap(path+'np_memmap/'+file_name+'.dat', dtype='float32', mode='r', shape=data_shape)
        fp_imputation = np.memmap(path+'np_imputation/'+file_name+'.dat', dtype='bool', mode='r', shape=data_shape)
        yield file_name,fp_data, fp_imputation,data_shape,first_day_date,date_list,last_day_date
        #data, nan_count, isValid = preprocess(fp_data, 5) #5 hours
        #if isValid:
            #print fp_imputation
            #visualize(fp_data,data)





if __name__ == "__main__":
    #start_convert()
    read_memmap()
    '''print is_valid(data)
        if is_valid(data):
            norm_data = remove_last_nans(data)
            norm_data = normalize(norm_data)
            norm_data = set_nan_to_zeros(norm_data)
            #visualize(data,norm_data)
            print last_day_date
            print data
            print data.shape'''

    #print first_day_data.shape
