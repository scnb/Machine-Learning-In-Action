# -*- coding: utf-8 -*-
from math import *
from numpy import *

#   载入数据


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt', 'r', encoding='utf-8')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) ##分别为系数w并且初始化为1，x和y
        labelMat.append(int(lineArr[2]))        ##将样本标签添加到labelMat中
    return dataMat, labelMat

#   定义sigmoid函数


def sigmoid(inX):
    return 1.0/(1+exp(-inX))

##Logistic回归梯度上升优化算法
def gradAscent(dataMatIn,classlabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classlabels).transpose()     ##transpose是转置，从行向量转成列向量
    m, n = shape(dataMatrix)                    ##得到数据矩阵的行数和列数
    alpha = 0.001                               ##定义步长，即每次沿梯度方向上升多少
    maxCycles = 500                             ##最多迭代500轮
    weights = ones((n, 1))                      ##样本中每个属性的权重或者说是回归系数
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

'''
##画出数据集和logistic回归最佳拟合直线的函数
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]        ##得到有多少个坐标
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])         ##第i行第1列（从0列开始）的数作为横坐标
            ycord1.append(dataArr[i, 2])        ##第i行第2列（从0列开始）的数作为竖坐标
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]   ##因为0是sigmoid函数的分解处，所以令 0=w0x0+w1x1+w2x2（其中x0恒为1）
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
'''

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    from numpy import arange, shape, array
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = 100
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()



#随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):                               #每次仅使用一个样本更新回归系数
        h = sigmoid((sum(dataMatrix[i]) * weights))    #dataMatrix是输入矩阵,每一行表示一个样本，每个样本中有3个元素
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


#改进的随机梯度上升算法
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))          #在python3中，range()返回的是range对象，而不是数组对象
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights


#Logistc回归分类函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX *weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open("horseColicTraining.txt")
    frTest = open("horseColicTest.txt")
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))      #每一个样本向量中包含20个特征值和一个标签
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')     #将每一行中的数据以tab字符为分隔隔开
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():            #进行多次测试，并取平均值
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

