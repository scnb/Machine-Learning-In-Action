#-*- coding:utf-8 -*-

from numpy import *
from matplotlib import *
from time import sleep


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
    return aj


#简化版SMO算法 SMO即序列最小优化算法
'''
创建一个alpha向量并初始化为0向量
当迭代次数小于最大迭代次数时进行外循环
    对数据集中的每个数据向量进行内循环：
        如果该数据向量可以被优化：
            随机选择另一个数据向量
            同时优化这两个数据向量
            如果两个数据向量都不能被优化，退出内循环
    如果所有向量都没有被优化，增加已迭代次数变量，继续下一次循环
'''
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
                    print ("L == H");continue
                eta = 2.0 * dataMatrix[i,:] *dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T - dataMatrix[j,:] * dataMatrix[j,:].T
                if eta >= 0:
                    print ("eta>=0");continue
                alphas[j] -= labelMat[j] * (Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print ("j not moving enough");continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
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

# 完整版Platt SMO的支持函数
'''
下面的一个对象和三个函数是作为辅助用到，当和优化过程及外循环组合在一起时，组成强大的SMO算法
'''

class optStrcut:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.x = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m= shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))        # m*1的矩阵
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))        # 误差缓存 m*2的矩阵，第一列是是否有效的标志位，第二列是E值

def calcEk(oS,k):                                   # 该函数计算对于给定的α值对应的E值
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i,oS,Ei):                               # 内循环中的启发式方法 用于选择第二个α值
    maxK = -1                                       # 选择合适的α值以保证在每次优化中采用最大步长
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]                           # 首先将输入值Ei在缓存中设置成有效的
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]  # nonzero函数返回一个以输入列表为目录的列表，这里的值非零
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS,k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek                             # 选择具有最大步长的j
        return maxK,Ej
    else:
        j = selecJrand(i, oS.m)
        Ej = clacEk(oS,j)
    return j, Ej

def updateEk(oS,k):                                 # 计算误差值并存入缓存当中
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]

# 完整版Platt SMO算法中的优化例子

def innerL(i, oS):
    Ei = calcEk(oS,i)
    if ((oS.labelMat[i] < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labellmat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print ("L==H")
            return 0
        eta = 2.0 *oS.X[i,:] *oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:] * oS.X[j,:].T
        if eta >= 0:
            print ("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print ("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        updateEk(oS,i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphasIold) * oS.X[i,:] * oS.X[i,:].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i,:] * oS.X[j,:].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphasIold) * oS.X[i,:] * oS.X[j,:].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j,:] * oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0

# 完整版Platt SMO外循环代码

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
    oS = optStrcut(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaParisChanged = 0
    while (iter < maxIter) and ((alphaParisChanged > 0) or (entireSet)):
        alphaParisChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaParisChanged += innerL(i,oS)
                print ("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphasParisChanged))
                iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) *(oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaParisChanged += innerL(i,oS)
                print ("non-bound, iter: %d i: %d , pairs changed %d:"%(iter, i, alphaParisChanged))
                iter += 1
        if entireSet:
            entireSet = False
        elif (alphaParisChanged == 0):
            entireSet = True
        print ("iteration number : %d" % iter)
    return oS.b, oS.alphas

