import numpy as np
from scipy import signal
##import my_modules
import matplotlib.pyplot as plt
from scipy.signal import welch


def sig_flt(sig):
    Fs = 1000.0
    t_start =1    # DO NOT make the t_start and t_stop indecies double type, since they are always integers and need to be in that form for the range() function 
    #t_start = 188000
    t_stop  = len(sig) #195500;
    #t_stop  = 195500
    f_stop = 5.0
    f_pass = 7.0
    filt_type = 'high'
    fp = 1.0                             
    fs = 10.0                           
    rp = 0.2                           
    rs = 1.0                             
    wp = 2*fp/Fs
    ws = 2*fs/Fs
    n, wn = signal.buttord(wp, ws, rp, rs, False)
    nr_coeff, dr_coeff = np.array(signal.butter(n, wn, filt_type, False))
    #print "\n Filter characteristics... "
    #print n, nr_coeff, dr_coeff
    #freqz(nr_coeff, dr_coeff, t_stop-t_start, Fs)  #https://docs.scipy.org/doc/scipy-0.18.1/reference/tutorial/signal.html#iir-filter
    sig_fil = np.array(signal.lfilter(nr_coeff, dr_coeff, sig))
    ##---------------------------------------------------------------
    #X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
##    plt.plot(sig_fil[1:10000], color="blue", linewidth=1.0, linestyle="-")
##    plt.show()
##    #-----------------------------------------------

    return sig_fil

            
