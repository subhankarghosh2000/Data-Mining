import pandas as pd
import matplotlib.pyplot as plt
import csv
from collections import Counter
from sklearn.decomposition import PCA, KernelPCA

matrix=[]
input_file=open("Covid_cases.csv","r")
l=[]
for row in input_file.readlines():
	list_row=list(map(float,row.split(",")))
	l.extend(list_row[:])
	matrix.append(list_row[:])
df = pd.DataFrame(matrix).transpose()
from sklearn import preprocessing 
min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1))
x_after_min_max_scaler = min_max_scaler.fit_transform(df).transpose()
plt.plot(x_after_min_max_scaler,'o')
out_file=open("normalization.csv","w")
out_row="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16}".format("Length","01_Sham-cJ-1","02_Sham-cJ-2","03_Sham-cJ-3","04_TAC-cJ-4","05_TAC-cJ-5","06_TAC-cJ-6","07_Sham-cByJ-1","08_Sham-cByJ-2","09_Sham-cByJ-3","10_TAC-cByJ-4","11_TAC-cByJ-5","12_TAC-cByJ-6","shrunkenLFC TAC-cJ/Sham-cJ","shrunkenLFC TAC-cByJ/Sham-cByJ","padj TAC-cJ/Sham-cJ","padj TAC-cByJ/Sham-cByJ")+"\n"
out_file.write(out_row)
for row in x_after_min_max_scaler:
	out_row=",".join(map(str,row))+"\n"
	out_file.write(out_row)
out_file.close()





def create_bins(lower_bound, width, quantity):
	bins = []
	for low in range(lower_bound, lower_bound + quantity*width + 1, width):
		bins.append((low, low+width))
	return bins
def find_bin(value, bins):
	for i in range(0, len(bins)):
		if bins[i][0] <= value and bins[i][1]>=value:
			return i+2
	return 1
out_file1=open("Descritization_binning.csv","w")
out_row1="{0},{1},{2},{3}".format("Data Set Value","Bin Index","Bin Start Range","Bin End Range")+"\n"
out_file1.write(out_row1)
bins = create_bins(lower_bound=-100,width=100,quantity=3000)
binned_weights = []
for value in l:
	bin_index = find_bin(value, bins)
	out_row1="{0},{1},{2},{3}".format(str(value),str(bin_index),str(bins[bin_index-2][0]),str(bins[bin_index-2][1]))+"\n"
	out_file1.write(out_row1)
	binned_weights.append(bin_index)
out_file1.close()	
frequencies = Counter(binned_weights)
out_file2=open("Descritization_frequency.csv","w")
out_row2="{0},{1}".format("Bin Index","Frequency")+"\n"
out_file2.write(out_row2)
for i in range(len(frequencies)):
	out_row2="{0},{1}".format(str(list(frequencies.keys())[i]),str(list(frequencies.values())[i]))+"\n"
	out_file2.write(out_row2)
out_file2.close()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_yscale("log")
ax.bar(list(frequencies.keys()),list(frequencies.values()))
plt.show()




pca = PCA()
x_pca = pca.fit_transform(x_after_min_max_scaler)
x_pca_dataframe = pd.DataFrame(data=x_pca)

# kpca = KernelPCA()
# X_kpca = kpca.fit_transform(x_after_min_max_scaler)
# pca = PCA()
# X_pca = pca.fit_transform(x_after_min_max_scaler)


out_file3=open("Dimension_reduction.csv","w")
out_row3="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16}".format("Length","01_Sham-cJ-1","02_Sham-cJ-2","03_Sham-cJ-3","04_TAC-cJ-4","05_TAC-cJ-5","06_TAC-cJ-6","07_Sham-cByJ-1","08_Sham-cByJ-2","09_Sham-cByJ-3","10_TAC-cByJ-4","11_TAC-cByJ-5","12_TAC-cByJ-6","shrunkenLFC TAC-cJ/Sham-cJ","shrunkenLFC TAC-cByJ/Sham-cByJ","padj TAC-cJ/Sham-cJ","padj TAC-cByJ/Sham-cByJ")+"\n"
out_file3.write(out_row3)
for row in x_pca:
	out_row3=",".join(map(str,row))+"\n"
	out_file3.write(out_row3)
out_file3.close()
plt.plot(x_pca,'o')
#plt.plot(X_kpca,'o')
plt.show()

















