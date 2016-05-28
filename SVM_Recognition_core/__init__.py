import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.loading_digits()

print(digits.data)
print (digits.target)

clf = svm.SVC()
clf = svm.SVC(gamma=0.001, C=100)   #This loads in all but the last 10 data points, so we can use all of these for training. Then, we can use the last 10 data points for testing

X, y = digits.data[:-10], digits.target[:-10] # X contains all of the "coordinates" and y is simply the "target" or "classification" of the data. Each bit of data pertains to a number.

clf.fit(X,y)
print(clf.predict(digits.data[-5]))

plt.imshow(digits.image[-5],cmap=plt.cm.gray.r,interpolation='nearest'
plt.show()

