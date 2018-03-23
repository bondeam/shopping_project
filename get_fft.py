# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 21:19:38 2018

@author: satya
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

def get_data(pathto):
    return str(pathto)

n=1
limit = 5
value_axis = np.zeros((102400))
while(n<limit):
    #CHANGE
    root_path = 'drop'+str(n)
    data_path = root_path+'.csv'
    img_path = './fft_data_sliced/'+root_path+'.png'
    #CHANGE
    sliced_path = "./labelled_data/cookies/data_wood_shelf_cookies_"+root_path+".csv"
    
    vib_data = np.genfromtxt(get_data(sliced_path), delimiter=',')
    time_axis = vib_data[:,0]
    value_axis = value_axis+vib_data[:,1]
    
    length = len(value_axis)
    #print(length)
    fft_values = np.fft.fft(value_axis, n=None, axis=-1, norm=None)/length
    spectrum = len(fft_values)/50
    
    to_write = np.ndarray.tolist(abs(fft_values[range(0,spectrum,1)]))
    
    write_path = str('./fft_data_sliced/'+data_path)
    n+=1
    #myFile = open(write_path, 'w')
    #with myFile:
        #writer = csv.writer(myFile)
        #writer.writerow(to_write)

value_axis = value_axis/(limit-1)
plt.plot(range(0,len(value_axis)),value_axis,label="Drop")

n=1
limit = 5
value_axis = np.zeros((102400))
while(n<limit):
    #CHANGE
    root_path = 'pickup'+str(n)
    data_path = root_path+'.csv'
    img_path = './fft_data_sliced/'+root_path+'.png'
    #CHANGE
    sliced_path = "./labelled_data/cookies/data_wood_shelf_cookies_"+root_path+".csv"
    
    vib_data = np.genfromtxt(get_data(sliced_path), delimiter=',')
    time_axis = vib_data[:,0]
    value_axis = value_axis+vib_data[:,1]
    
    length = len(value_axis)
    #print(length)
    fft_values = np.fft.fft(value_axis, n=None, axis=-1, norm=None)/length
    spectrum = len(fft_values)/50
    
    to_write = np.ndarray.tolist(abs(fft_values[range(0,spectrum,1)]))
    
    write_path = str('./fft_data_sliced/'+data_path)
    n+=1
    #myFile = open(write_path, 'w')
    #with myFile:
        #writer = csv.writer(myFile)
        #writer.writerow(to_write)

value_axis = value_axis/(limit-1)
plt.plot(range(0,len(value_axis)),value_axis,label="Pickup")

n=1
limit = 5
value_axis = np.zeros((102400))
while(n<limit):
    #CHANGE
    root_path = 'noise'+str(n)
    data_path = root_path+'.csv'
    img_path = './fft_data_sliced/'+root_path+'.png'
    #CHANGE
    sliced_path = "./labelled_data/cookies/data_wood_shelf_cookies_"+root_path+".csv"
    
    vib_data = np.genfromtxt(get_data(sliced_path), delimiter=',')
    time_axis = vib_data[:,0]
    value_axis = value_axis+vib_data[:,1]
    
    length = len(value_axis)
    #print(length)
    fft_values = np.fft.fft(value_axis, n=None, axis=-1, norm=None)/length
    spectrum = len(fft_values)/50
    
    to_write = np.ndarray.tolist(abs(fft_values[range(0,spectrum,1)]))
    
    write_path = str('./fft_data_sliced/'+data_path)
    n+=1
    #myFile = open(write_path, 'w')
    #with myFile:
        #writer = csv.writer(myFile)
        #writer.writerow(to_write)

value_axis = value_axis/(limit-1)
plt.plot(range(0,len(value_axis)),value_axis,label="No Action")
plt.title("Vibration Magnitude Data")
plt.legend(loc='upper right')
plt.show()
#plt.plot(range(0,spectrum,1),abs(fft_values[range(0,spectrum,1)]))
#plt.savefig(img_path)

