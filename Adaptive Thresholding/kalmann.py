# Kalman filter example demo in Python

# A Python implementation of the example given in pages 11-15 of "An
# Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
# University of North Carolina at Chapel Hill, Department of Computer
# Science, TR 95-041,
# http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html

# by Andrew D. Straw

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

import pandas as pd
import glob, os  

std_dev = 0


def get_pick_drop(directory,item):
    path = directory # './data'
    #drop_Files = glob.glob(path + "/*"+item+"_shelf_wood_drop*.csv")
    #pick_Files = glob.glob(path + "/*"+item+"_shelf_wood_pick*.csv")
    #noise_Files = glob.glob(path + "/*"+item+"_shelf_wood_noise*.csv")
    
    drop_Files = glob.glob(path + "/*_shelf_wood_drop*.csv")
    pick_Files = glob.glob(path + "/*_shelf_wood_pick*.csv")
    noise_Files = glob.glob(path + "/*_shelf_wood_noise*.csv")
    
    drop = []
    pick = []
    noise = []

    for file_ in drop_Files:
        # remove the first line, starting with "shopping"
        df = pd.read_csv(file_,index_col=None, header=None).iloc[1:]
        drop.append(df)
        
    for file_ in pick_Files:
        df = pd.read_csv(file_,index_col=None, header=None).iloc[1:] 
        pick.append(df)
        
    for file_ in noise_Files:
        df = pd.read_csv(file_,index_col=None, header=None).iloc[1:] 
        noise.append(df)

    return pick, drop, noise

def trunc_data(data,data_mean):
    global std_dev
    ind=0
    while(ind<len(data)):
        #if(data[ind]>data_mean):
            #data[ind] = data[ind]
        #else:
            #data[ind] = data_mean
        temp_val = data[ind]-data_mean
        data[ind] = data_mean+abs(temp_val)
        std_dev += abs(temp_val)
        ind+=1
    std_dev /= len(data)

root_path = "./data/"
item = "butter"
action = "drop"
index = "51"
path = root_path+"data_"+item+"_shelf_wood_"+action+index+".csv"

def window_data(chosen_data, window_size):
    # To be executed after get_pick_drop
    # Arguments : Type of data (Pick,Drop,Noise) and Window size
    # Returns : List of lists, each sublist being a data window of specified size
    ind = 0
    data = []
    while (ind<len(chosen_data)-1):
        data_temp = chosen_data[ind]
        data_temp = data_temp.values.T.tolist()
        data_temp = data_temp[0]
        for i in data_temp:
            data.append(int(i))
        ind+=1
    data = [ int(x) for x in data ]
    windows = []
    index = 0
    while ((index+window_size)<len(data)):
        windows.append(data[index:(index + window_size)])
        index += window_size
    return windows

def window_dfdata(chosen_data, window_size):
    # To be executed after get_pick_drop
    # Arguments : Type of data (Pick,Drop,Noise) and Window size
    # Returns : List of lists, each sublist being a data window of specified size
    ind = 0
    data = []
    data = pd.concat(chosen_data)
    windows = []
    index = 0
    while ((index+window_size)<len(data)):
        windows.append(data[index:(index + window_size)])
        index += window_size
    return windows
    

pick_data,drop_data,noise_data = get_pick_drop("../../data","butter")
chosen_data = drop_data
data = window_dfdata(chosen_data,2000)

data = data[5]
data_range = range(0,len(data),1)
data_max = max(data)
data_min = min(data)
data_mean = data_min+(data_max-data_min)/2

trunc_data(data,data_mean)

plt.rcParams['figure.figsize'] = (10, 8)

# intial parameters
n_iter = len(data)
sz = (n_iter,) # size of array
#x = 1 # truth value (typo in example at top of p. 13 calls this z)
z = data # observations (normal about x, sigma=0.1)

Q = 1e-5 # process variance

# allocate space for arrays
xhat=np.zeros(sz)      # a posteri estimate of x
P=np.zeros(sz)         # a posteri error estimate
xhatminus=np.zeros(sz) # a priori estimate of x
Pminus=np.zeros(sz)    # a priori error estimate
K=np.zeros(sz)         # gain or blending factor

R = std_dev**2 # estimate of measurement variance, change to see effect
print("Standard Deviation : ",std_dev)

# intial guesses
xhat[0] = data_mean
P[0] = 1.0

for k in range(1,n_iter):
    # time update
    xhatminus[k] = xhat[k-1]
    Pminus[k] = P[k-1]+Q

    # measurement update
    K[k] = Pminus[k]/( Pminus[k]+R )
    xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
    P[k] = (1-K[k])*Pminus[k]

plt.figure()
plt.plot(data,label='noisy measurements')
plt.plot(xhat,'b-',label='a posteri estimate')
#plt.axhline(x,color='g',label='truth value')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('Threshold Estimate')

print("final threshold estimate is : ",xhat[len(xhat)-1])