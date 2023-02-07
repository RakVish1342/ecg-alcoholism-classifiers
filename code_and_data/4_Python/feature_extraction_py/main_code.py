import numpy as np
import scipy
import time_domain_feat as tdf
import sig_rd
import sig_flt_iir
import csv
import imp_wave
import calc_power


sig=sig_rd.signal_read()            ##Reading the signal
sig_fil=sig_flt_iir.sig_flt(sig)    ##Filtering the signal using iir filter
print "Signal loaded "

##Time domain Features##
t_start=1
t_stop=len(sig_fil)

t_rr=tdf.find_peaks(sig_fil,t_start,t_stop) ##Finding RR interval
time_fea=tdf.calculate_features(t_rr)
[t_rr_diff, t_rr_diff_diff, f_mean, f_std, f_mean_HR, f_std_HR, f_rms, f_nn50, f_pnn50]=time_fea
time_fea_1=np.append(f_mean, f_std, f_mean_HR)
time_fea_1=np.append(time_fea_1, f_std_HR, f_rms)
time_fea_1=np.append(time_fea_1, f_nn50, f_pnn50)
[points,sd1,sd2]=tdf.poincare(t_rr_diff)    ##Poincare plot details

##need to include entropy code##
print "Time Domain Features extracted successfully"

##Frequency Domain Features##
sig_wav=imp_wave.wavelet_imp(sig)       ##performing Wavelet Transform on the signal

freq_fea=calc_power.power_calc(sig_wav) ##Calculating Frequency Domain Features

time_fea_1=time_fea_1.tolist()
sd1=sd1.tolist()
sd2=sd2.tolist()
sd1=[sd1]
sd2=[sd2]

print "Frequency Domain Features extracted successfully"

features=time_fea_1+sd1+sd2+freq_fea

print "All features appended successfully"

csvfile=open('features.csv','wb')           ##Writing to a CSV file
writer=csv.writer(csvfile,delimiter=',')
writer.writerow(features)
csvfile.close()
