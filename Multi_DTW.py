import numpy as np
import os
import librosa

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
import librosa.display
from glob import glob
from scipy.spatial.distance import cdist

def dtw_distance(distances):
    DTW = np.empty_like(distances)
    DTW[0, 0] = 0
    for i in range(0, DTW.shape[0]):
        for j in range(0, DTW.shape[1]):
            if i ==0 and j==0:
                DTW[i, j] = distances[i, j]
            elif i == 0:
                DTW[i, j] = distances[i, j] + DTW[i, j-1]
            elif j == 0:
                DTW[i, j] = distances[i, j] + DTW[i-1, j]
            else:
                DTW[i, j] = distances[i, j] + min(DTW[i-1, j],  
                                              DTW[i, j-1],  
                                              DTW[i-1, j-1] 
                                             )    
    return DTW

def backtrack(DTW):
    i, j = DTW.shape[0] - 1, DTW.shape[1] - 1
    output_x = []
    output_y = []
    # append last scalar value in spectrogram(reference, compare)
    output_x.append(i) # last scalar in reference spectrogram
    output_y.append(j) # last scalar in compare spectrogram
    while i > 0 and j > 0:
        local_min = np.argmin((DTW[i - 1, j - 1], DTW[i, j - 1], DTW[i - 1, j]))
        if local_min == 0:
            i -= 1
            j -= 1
        elif local_min == 1:
            j -= 1
        else:
            i -= 1
        output_x.append(i)
        output_y.append(j)

    #append start of spectrogram scalar
    output_x.append(0)
    output_y.append(0)
    
    #Since it was calculated from the back, in this part, return to original sequence 
    output_x.reverse()
    output_y.reverse()
    
    return np.array(output_x), np.array(output_y) # this output is dtw path. Longer than reference

def multi_DTW(a,b,len_ref,len_tar): #a is reference dtw path, b is comparison target dtw path. a, b lengths are same.
    cnt = []
    for x in range(len(a)):
        if a[x-1] == a[x]:
            cnt.append(x)
    target = np.delete(b, cnt) # Match the length of the target to the reference 
    if len(target) < len_ref: # Sometimes the length of the comparison target is smaller than the standard, in which case the last value of the comparison target is entered.
        differ =  len_ref - len(target) 
        target = np.pad(target,(0,differ), 'constant', constant_values=(1, len_tar-1))
    return target

def dtw(ref, compare, len_ref, len_tar, distance_metric='euclidean'):
    distance = cdist(ref, compare, distance_metric) # use with euclidean
    cum_min_dist = dtw_distance(distance) # calculate using dtw algorithm
    x, y = backtrack(cum_min_dist) # back track in dtw
    final_y = multi_DTW(x, y, len_ref, len_tar)
    return final_y
