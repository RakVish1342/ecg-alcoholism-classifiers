import numpy as np
def signal_read():
    sig = np.array(np.genfromtxt('C:/Users/DELL/Desktop/alc1.txt', delimiter=','))
    for i in range(len(sig)):
	sig[i] = sig[i] + 0.0 # typecast all values to double ... np.array dooesn't make it floating point by default for Py27
	
    ##print "Signal loaded ... "

    return sig
    
