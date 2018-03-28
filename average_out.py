import numpy as np
import matplotlib.pyplot as plt
import csv

def get_data(pathto):
    return str(pathto)

n=1
limit = 10
value_axis = np.zeros((2048))
while(n<limit):
    #CHANGE
    action = "noise"
    sample = str(n)
    data_path =  './fft_data_sliced/'+action+'_average.csv'
    img_path = './fft_data_sliced/'+action+'_average.png'
    #CHANGE
    sliced_path = "./fft_data_sliced/cookies/wood_shelf/"+action+"/"+action+sample+".csv"
    
    vib_data = np.genfromtxt(get_data(sliced_path), delimiter=',')
    value_axis = value_axis+vib_data[:]
    #print(value_axis)
    n=n+1

value_axis = value_axis/(limit-1)
to_write = np.ndarray.tolist(value_axis)
    
write_path = str('./fft_data_sliced/'+action+"_average.csv")
    
#myFile = open(write_path, 'w')
#with myFile:
   # writer = csv.writer(myFile)
    #writer.writerow(to_write)
plt.ylim(0,0.02)
plt.plot(range(0,len(value_axis)),value_axis)
plt.show()
plt.savefig(img_path)

