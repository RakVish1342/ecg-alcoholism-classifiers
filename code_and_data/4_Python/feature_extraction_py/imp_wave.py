import pywt
import numpy as np
from pywt import wavedec
from scipy.signal import welch
from scipy import signal
import calc_power


def wavelet_imp(sig):
    coeffs=wavedec(sig,'db5',level=8)   ##Performing Wavelet Transform
    cA8,cD8,cD7,cD6,cD5,cD4,cD3,cD2,cD1=coeffs  ##Reassigning the coefficients
    n=len(sig)
    renc=pywt.upcoef('a',cA8,'db5',level=8,take=n)    ##Recreating a particular signal of signal
    renc1=(sig)-(renc)      ##Removing the component that causes base line wandering
    
##    csvfile=open('py_value.csv','wb')       
##    writer=csv.writer(csvfile,delimiter=',')
##    writer.writerow(renc1)
    return renc1

##    freq_fea=calc_power.power_calc(sig)     ##Calculating PSD
##    print freq_fea

