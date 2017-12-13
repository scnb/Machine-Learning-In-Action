#-*- coding:utf-8 -*-

import random
from numpy import *
from matplotlib import *



#加载数据
def loadDataSet(fileName):
    dataMat = []        #数据集合
    labelMat = []       #标签集合
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')    #以制表符为界分割开
        dataMat.append([float(lineArr[0]), float(lineArr[1])])      #lineArr中第1和第2个数据存在dataMat中
        labelMat.append(float(lineArr[2]))                          #lineArr中第3个数据存在标签集合中
    return dataMat, labelMat

#在某个区间范围内随机选择一个整数
def selectJrand(i,m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m)) #随机选一个不等于i的数
    return j

#调整α为大小合适的值
def clipAlpha(aj,H,L):
    if aj > H:  #如果aj比上限还大
        aj = H
    if L > aj:  #如果aj比下限还小
        aj = L

#简化版SMO算法
def smoSimple(dataMatIn, classLabels, C, toler, maxIter): # 五个参数分别为数据集、类别标签、常数C、容错率和退出前最大的循环数
    dataMatrix = mat(dataMatIn)              # 将数据集转换成矩阵的形式
    labelMat = mat(classLabels).transpose()  # 得到类别标签的列向量，该向量的每一行都与数据矩阵的每一行对应
    b = 0
    m, n = shape(dataMatrix)                 # 得到数据矩阵的行数和列数
    alphas = mat(zeros((m, 1)))              # 生成m行1列的矩阵，即将α转成列向量
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0               # 用来记录alpha是否已经进行优化
        for i in range(m):                  # 对于数据集中的每一个数据向量进行操作
            # 计算预测标签
            fxi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i,:].T)) + b
            Ei = fxi - float(labelMat[i])   # 计算误差
            # 如果错误率大于或小于正负容错率，则进行优化
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)    # 随机选出另一个α值
                # 针对上面选出的α值进行预测
                fxj = float(multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fxj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):         # 如果这两个α值对应的不是同一类的
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print ("L == H")
                else:
                    continue   #若L和H不相等，则跳过这次for循环
                eta = 2.0 * dataMatrix[i,:] *dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T - dataMatrix[j,:] * dataMatrix[j,:].T
                if eta >= 0:
                    print ("eta>=0")
                else:
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print ("j not moving enough")
                else:
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j,:] * dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print ("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):  # 如果本次循环没有进行优化，则迭代次数加1
            iter += 1
        else:
            iter = 0                  # 否则就将迭代次数清0，只有在连续maxIter迭代次都没有进行优化，while循环才会退出
        print ("iteration number: %d" % iter)
        return b, alphas
