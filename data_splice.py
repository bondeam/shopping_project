import csv
import pandas as pd 
import matplotlib.pyplot as plt

file = open(r"D:\CMU\Spring 2018\MPC\metal_cabinet\Detergent\data_metal_cabinet_detergent.csv", "r")
print(file)

values = csv.reader(file, delimiter=' ')
time = []
vib = []


for row in values:
    time.append(float(row[0].split(',')[0]))
    vib.append(float(row[0].split(',')[1]))

num_win = 10 #change this to adjust number of windows
size = int(len(time)/num_win)

lst_time = [time[i:i+size] for i  in range(0, len(time), size)]
lst_vib = [vib[i:i+size] for i  in range(0, len(vib), size)]

csv_array = []

thing_str = "mc_detergent_"
dotCsv = ".csv"
dotpng = ".png"
for j in range(len(lst_time)):
	for i in range(len(lst_time[j])):
		csv_array.append([lst_time[j][i], lst_vib[j][i]])
	filename = thing_str + str(j) + dotCsv
	print(filename)
	with open(filename, 'w') as resultFile:
		wr = csv.writer(resultFile, dialect='excel')
		wr.writerows(csv_array)

for i in range(len(lst_time)):
	plt.plot(lst_time[i], lst_vib[i])
	image = thing_str + str(i) + dotpng
	plt.savefig(image)
	plt.close()