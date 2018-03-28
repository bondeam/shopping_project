import serial
import time
import csv
import matplotlib.pyplot as plt
import pandas

max_time = 10 # maximum nomber of seconds to record data, change according to experiment
start_time = time.time()  # remember when we started

ser = serial.Serial('COM3', 9600, timeout=None)

final_arr = []
while (time.time() - start_time) < max_time:
  s = ser.readline().decode("utf-8").strip().split(",") 
  final_arr.append(s)

with open("output.csv", "w") as csv_file:
  writer = csv.writer(csv_file, delimiter=',')
  for line in final_arr:
    writer.writerow(line)

data_df = pandas.read_csv('output.csv')
plt.plot(data_df.index.values,data_df.iloc[:,0].values)
plt.title('accelX')
plt.show()
plt.plot(data_df.index.values,data_df.iloc[:,1].values)
plt.title('accelY')
plt.show()
plt.plot(data_df.index.values,data_df.iloc[:,2].values)
plt.title('accelZ')
plt.show()