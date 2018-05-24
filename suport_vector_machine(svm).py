import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm # to find groups
"""
support vector machine find the vectors to know if we are in one group or another.
"""



import time

digits = datasets.load_digits() 

clf = svm.SVC(gamma=0.001, C=100) #gamma = anchura de la linea separadora C=margen de error

x, y = digits.data[:-20], digits.target[:-20]
clf.fit(x, y) # find groups


for i in range(1, 21):
	print('Prediction: ', clf.predict(digits.data[[-i]]))
	plt.imshow(digits.images[-i], cmap=plt.cm.gray_r, interpolation="nearest")
	plt.show()
	
	