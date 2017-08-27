import kNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

# group, labels=kNN.createDataSet()
# datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# plt.show()

# kNN.datingClassTest()
# kNN.classifyPerson()
kNN.handwritingClassTest()

# testVector = kNN.img2vector('testDigits/0_13.txt')
# print testVector[0, 0:31]
# print testVector[0, 32:63]
