import numpy as np
import matplotlib.pyplot as plt
import csv

n=1

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

print(max(drop_axis))
print(max(pickup_axis))
print(max(noise_axis))

plt.ylim(0,0.020)
plt.plot(range(0,len(drop_axis)),drop_axis,label='Drop')
plt.plot(range(0,len(pickup_axis)),pickup_axis,label='Pickup')
plt.plot(range(0,len(noise_axis)),noise_axis,label='Noise')
plt.title("FFT Distribution for Mean Vibration Values")
plt.legend(loc='upper right')
plt.show()
    
