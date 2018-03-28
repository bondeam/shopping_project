import numpy as np
import matplotlib.pyplot as plt
import csv

numbins = 80
trunc_value = 0.0025

def get_data(pathto):
    return str(pathto)

drop_path = "./fft_data_sliced/cookies/wood_shelf/average/drop_average.csv"
pickup_path = "./fft_data_sliced/cookies/wood_shelf/average/pickup_average.csv"
noise_path = "./fft_data_sliced/cookies/wood_shelf/average/noise_average.csv"

drop_data = np.genfromtxt(get_data(drop_path), delimiter=',')
drop_axis = drop_data[:]

noise_data = np.genfromtxt(get_data(noise_path), delimiter=',')
noise_axis = noise_data[:]

pickup_data = np.genfromtxt(get_data(pickup_path), delimiter=',')
pickup_axis = pickup_data[:]

plt.xlim(0,trunc_value)
plt.subplot(311)
plt.hist(drop_axis,label='Drop',bins=numbins,range=(0,trunc_value))
plt.title("Truncated Histogram for Frequency Distribution Values")
plt.legend(loc='upper right')
plt.subplot(312)
plt.hist(pickup_axis,label='Pickup',bins=numbins,range=(0,trunc_value))
plt.legend(loc='upper right')
plt.subplot(313)
plt.hist(noise_axis,label='No Action',bins=numbins,range=(0,trunc_value))
plt.legend(loc='upper right')
plt.show()
    
