import numpy as np
import pandas as pd
import operator
import importlib

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classfy0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount={}
    for i in range(k):
        votelLabel = labels[sortedDistIndices[i]]
        classCount[votelLabel] = classCount.get(votelLabel, 0)+1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(fileName):
    fr = open(fileName)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMatrix = np.zeros(numberOfLines, 3)
    labelVector = []
    index = 0
    for line in arrayOLines:
        lineList = line.split('\t')
        returnMatrix[index:]=lineList[0:3]
        labelVector.append(int(lineList[-1]))
        index = index+1
    return returnMatrix, labelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataset = np.zeros(dataSet.shape)
    m = dataSet.shape[0]
    normDataset = dataSet-np.tile(minVals, (m, 1))
    normDataset = normDataset/np.tile(ranges, (m, 1))
    return normDataset, ranges, 
    
def dattingClassTest():
    rate = 0.1
    dataMatrix, dataLabels = file2matrix('datTest.txt')
    normDataSet, ranges = autoNorm(dataMatrix)
    m = normDataSet.shape[0]
    testVecs = int(m*rate)
    errors = 0.0
    for i in range(testVecs):
        classfyResult = classfy0(normDataSet[i,:], normDataSet[testVecs:m,], dataLabels[testVecs:m], 3)
        print("classfy result is %d, real answer is %d " % (classfyResult, dataLabels[i]))
        if(classfyResult!=dataLabels[i]):
            errors+=1
    print("error rate is %d " % errors/float(testVecs))