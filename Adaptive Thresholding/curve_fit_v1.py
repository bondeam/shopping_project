import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

root_path = "./data/"
item = "butter"
action = "drop"
index = "81"
path = root_path+"data_"+item+"_shelf_wood_"+action+index+".csv"

data = []
cnt = 0
with open(path, 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        if cnt != 0:
            curr_value = row[0]
            data.append(int(curr_value))
        cnt+=1

data_range = range(0,len(data),1)
data_max = max(data)
data_min = min(data)
data_mean = (data_max-data_min)/2

for i in data_range:
    if data[i]<data_mean:
        data[i] = data_mean
    else:
        data[i] = data[i]#-data_mean

co_eff,covar = np.polyfit(data_range,data,15,cov=True)
curve_coeff = np.poly1d(co_eff)
plt.plot(data_range,data,data_range,curve_coeff(data))
plt.show()
        
print(co_eff)
#print(covar)