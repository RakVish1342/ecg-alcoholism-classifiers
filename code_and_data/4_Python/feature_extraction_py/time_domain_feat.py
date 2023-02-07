def find_peaks(sig_fil, t_start, t_stop):
    import numpy as np
    t_rr = np.array([])
    temp_signal = np.array([])
    temp_index = np.array([])
    thresh = min(sig_fil) + 0.8*(max(sig_fil) - min(sig_fil))
    cross_up = 0
    cross_up_prev = 0
    for i in range(t_start, t_stop):
        #i
        if(sig_fil[i] > thresh):
            #disp('high')
            #disp(sig_fil(i))
            temp_signal = np.append(temp_signal, sig_fil[i])
            temp_index = np.append(temp_index, i)
            cross_up = 1
        else:
            #disp('low')
            #disp(sig_fil(i))  
            cross_up = 0
        
        if(cross_up_prev == 1 and cross_up == 0):
            max_val = max(temp_signal)
            index_arr = np.where(temp_signal==max_val)
            max_index = index_arr[0][0]   # np.arrays give all occurrences as a list of arrays...here we need first element of list(there is only one element in the list, but we still need to call it as the first element from that list) (viz and array of all occurrences), and we need first element of that array 
            #but such a line is not needed for lists, since unlike matlab, python gives index of first occurrence only

            t_rr = np.append(t_rr, temp_index[index_arr]) #max_index]) 

            temp_signal = np.array([])
            temp_index = np.array([])
            max_val = 0
            index_arr = np.array([])
            max_index = 0
        cross_up_prev = cross_up
    return np.array(t_rr)


def calculate_features(t_rr):
    import numpy as np
    
    #print "length trr: ", len(t_rr)
    #print "trr: ", t_rr[0:5], t_rr[-5:]
    t_rr1 = np.insert(t_rr, 0, 0)
    #print "trr1: ", t_rr1[0:5], t_rr1[-5:]
    
    t_rr2 = np.append(t_rr, 0)
    #print "trr: ", t_rr[0:5], t_rr[-5:]
    #print "trr2: ", t_rr2[0:5], t_rr2[-5:]
    
    t_rr_diff = t_rr2 - t_rr1
    #print "length temp", len(t_rr_diff)
    #print "trr_diff_temp", t_rr_diff[0:5], t_rr_diff[-5:]
    t_rr_diff = t_rr_diff[1:-1]   # HOLDS THE RR INTERVALS #1 to end-1 =nt [0:-1] since this means "upto and not including the last element, last element being -1"
    #print "length is", len(t_rr_diff)
    #print "trr_diff", t_rr_diff[0:5], t_rr_diff[-5:]
    
    f_mean = np.mean(t_rr_diff)
    f_mean_min = np.mean(t_rr_diff/(1000*60)) # SAME AS: t_mean/(1000*60)
    
    f_std = np.sqrt(np.var(t_rr_diff))
    f_std_min = np.sqrt(np.var(t_rr_diff/(1000*60))) # SAME AS: f_std/(1000*60)      
    
    f_mean_HR = 1.0/f_mean_min
    f_std_HR = 1.0/f_std_min
    f_rms = np.sqrt( (1.0/len(t_rr_diff))*sum(np.power(t_rr_diff, 2)) )

    t_rr_diff1 = np.insert(t_rr_diff, 0, 0)
    t_rr_diff2 = np.append(t_rr_diff, 0)
    t_rr_diff_diff = t_rr_diff2 - t_rr_diff1
    t_rr_diff_diff = t_rr_diff_diff[1: -1] # HOLDS THE DIFFERENCES IN THE RR INTERVALS

    f_nn50 = len( np.where(t_rr_diff_diff >= 50.0/1000.0)[0] ) # [0] is to select the first "dimension"/first array in list of arrays
    #print f_nn50
    f_pnn50 = 100.0*(f_nn50/float(len(t_rr_diff)) )
    #print f_pnn50


    # CHECK UNITS

    #
    # ENSURE THAT MEAN HEART RATE IS IN MINUTES, WHILE MEAN IS IN MILLISECONS
    #


    #print [f_mean, f_std, f_mean_HR, f_std_HR, f_rms, f_nn50, f_pnn50]
    return [t_rr_diff, t_rr_diff_diff, f_mean, f_std, f_mean_HR, f_std_HR, f_rms, f_nn50, f_pnn50]



def poincare(t_rr_diff):
    import numpy as np
    # Verify if sd1^2 = (1/2)*SDSN^2 and sd2^2 = 2*SDNN - (1/2)*SDSN^2

    
    points = []  # Use REGULAR arrays of python...when and if needed to treat as a matrix for multiplication, convert on spot to matrix via np.matrix(points), store this to some variable and do some opperations on it
    for i in range(len(t_rr_diff)-1): # since ith and (i+1)th samples are used
        pair = [ t_rr_diff[i], t_rr_diff[i+1] ]
        points.append(pair)    
    
    #print points 
    #print (len(t_rr_diff)-1)
    #print len(points)
    
    #line x = y
    a = 1
    b = -1
    dist = np.array([])
    len_pts = len(t_rr_diff)-1 # since this is the number of times pairs are added to points
    for i in range(len_pts):
        d = np.abs( float(a*points[i][0] + b*points[i][1]) / float(np.sqrt(a*a + b*b)) )
        dist = np.append(dist, d)
    sd1 = np.sqrt(np.var(dist))

    #line x = -y
    a = 1
    b = 1
    dist = np.array([])
    for i in range(len_pts):
        d = np.abs( float(a*points[i][0] + b*points[i][1]) / float(np.sqrt(a*a + b*b)) )
        dist = np.append(dist, d)
    sd2 = np.sqrt(np.var(dist))
    
    return points, sd1, sd2
    
    
def apen_phi(m, r, N, t_rr_diff):
    import numpy as np
    
    u = np.zeros(m)
    vec = [] # Use REGULAR arrays of python...when and if needed to treat as a matrix for multiplication, convert on spot to matrix via np.matrix(points), store this to some variable and do some opperations on it
    for i in range(N-m+1):
        u = [ t_rr_diff[i], t_rr_diff[i+1] ]
        vec.append(u)


    # Initialize 'dist' and 'cj' to be INF vector or ZEROS vector? ... For ApEn it may not matter, for SampEn it would I guess

    vec_1 = len(vec) # No need of using np.shape, since the we are using a list of arrays and not a matrix

    dist = np.zeros( [vec_1, vec_1] )
    #print dist
    #print np.shape(dist)
    for j in range(vec_1): # cycle through uj
        v1 = vec[i][:]
        for k in range(vec_1): # check each uk
            v2 = vec[j][:]
            temp = np.zeros( len(range(m-1)) + 1 ) # OR len(range(m))...# 0:m-1 means 0 to m-1 both inclusive. range() excludes the last element 'm-1'
            for n in range(m): # calculate the distance for the pair NOTE: it's 0:m-1 only and NOT 1:m, since j will have a non zero value, so for n=0 also, index will be non-zero 
                temp[n] = np.abs( t_rr_diff[j+n] - t_rr_diff[k+n] )
            dist[j][k] = max(temp)    
    #print dist

    # relative index  
    cj = np.zeros(vec_1)
    for j in range(vec_1): # cycle through uj
        row = dist[j][:]
        count = 0
        for k in range(vec_1): # check each uk
            if (row[k] <= r): # distance between two vectors can be zero if the t_rr_diff is the same for multiple duratoins, or dist is being calculated with self (which itself is why cj can never be 0)
                count = count + 1    
        #count
        cj[j] = float(count)/float((N-m+1))
    #cj

    phi = float(sum(np.log(cj)))/float((N-m+1))
    return phi
