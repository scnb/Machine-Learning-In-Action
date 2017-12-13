import random
import numpy
import matplotlib

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
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))      #生成m行1列的矩阵，即将α转成列向量
    iter = 0
    while (iter <maxIter):
        alphaParisChanged = 0
        for i in range(m):  #对于数据集中的每一个数据向量进行操作
            #计算预测标签
            fxi = float(multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[i:].T)) + b
            Ei = fxi - float(labelMat[i])       #计算误差
            #如果错误率大于或下于正负容错率，则进行优化
            if ((labelMat[i]*Ei < -toler) and (alphas[i] <  C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)    #随机选出另一个α值
                #针对上面选出的α值进行预测
                fxj = float(multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fxj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]): #如果这两个α值对应的不是同一类的
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print ("L == H")
                else:
                    continue;   #若L和H不相等，则跳过这次for循环
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
                alphas[i] += labelMat[j] * labelMat[i] *(alphaJold - alphas[j])

