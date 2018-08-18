from matplotlib.image import imread
from numpy import array,full
from os import listdir as ls
from regex import search
X_dataset = list()
Y_dataset = list()
for img in ls("/home/cnlab/hubba/coil-20-proc/"):
	im = imread("./coil-20-proc/"+img)
	im = im.reshape(16384)	
	X_dataset.append(im)
	result = search('obj(.*?)_', img)
	s=full(20,-1)
	s[int(result[1])-1]=1
	Y_dataset.append(s)
print(len(X_dataset))
print(len(Y_dataset))
