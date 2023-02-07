import numpy as np
import biosppy
import numpy
import scipy
from scipy.signal import welch


def power_calc(sig):
    #fxx,pxx=biosppy.signals.tools.power_spectrum(signal=sig,sampling_rate=1000,pad=None,pow2=False,decibel=False)
    fxx,pxx=welch(sig,fs=1000,window='hamming',nperseg=512,noverlap=50,nfft=1024)
    pxx=pxx.tolist()
    fxx=fxx.tolist()
    pxx1=pxx
    
    ###VLF###
    ind0_1=[i for i,v in enumerate(fxx) if v<10]
    X=fxx[0:ind0_1[len(ind0_1)-1]]
    val=pxx[0:len(X)-1]
    abs_pw_vlf=max(val)
    ind=pxx.index(abs_pw_vlf)
    pk_freq_vlf=fxx[ind]
    #val=val[0:len(val)-2]
    #pxx=np.array(pxx)
    #fxx=np.reshape(fxx,149498)
    X=X.extend([0])
    ab_pow_vlf=numpy.trapz(val,x=X)


    ###LF###
    ind1_1=[i for i,v in enumerate(fxx) if ((v>10)&(v<15))]
    X1=fxx[ind0_1[len(ind0_1)-1]:ind1_1[len(ind1_1)-1]]
    val1=pxx[ind0_1[len(ind0_1)-1]:(ind0_1[len(ind0_1)-1]+(len(X1)-1))]
    abs_pw_lf=max(val1)
    ind1=pxx.index(abs_pw_lf)
    pk_freq_lf=fxx[ind1]
    X1=X1.extend([0])
    ab_pow_lf=scipy.integrate.trapz(val1,x=X1)


    ###HF###
    ind2_1=[i for i,v in enumerate(fxx) if ((v>15)&(v<40))]
    #print ind2_1
    X2=fxx[ind1_1[len(ind1_1)-1]:ind2_1[len(ind2_1)-1]]
    val2=pxx[ind1_1[len(ind1_1)-1]:(ind1_1[len(ind1_1)-1]+(len(X2)-1))]
    abs_pw_hf=max(val2)
    ind2=pxx.index(abs_pw_hf)
    pk_freq_hf=fxx[ind2]
    X2=X2.extend([0])
    ab_pow_hf=scipy.integrate.trapz(val2,x=X2)

    pw_ttl=ab_pow_vlf+ab_pow_lf+ab_pow_hf
    rp_vlf=ab_pow_vlf/pw_ttl
    rp_lf=ab_pow_lf/pw_ttl
    rp_hf=ab_pow_hf/pw_ttl

    norm_lf=ab_pow_lf/(pw_ttl-ab_pow_vlf)
    norm_hf=ab_pow_hf/(pw_ttl-ab_pow_vlf)

    ratio=ab_pow_lf/ab_pow_hf

    freq_fea=[pk_freq_vlf,pk_freq_lf,pk_freq_hf,ab_pow_vlf,ab_pow_lf,ab_pow_hf,pw_ttl,rp_vlf,rp_lf,rp_hf,norm_lf,norm_hf,ratio] 

    return freq_fea    

    

    
    
    
    
