#-*- coding:utf-8 -*-

from numpy import *

# K均值聚类算法

# K-均值聚类支持函数

def loadDataSet(fileName):
    '''
    :param fileName:输入文件的名字
    :return:反回数据列表
    在python3下，map函数是一个坑，map函数的作用是将一个函数作用于一个list中的所有元素，在python2下它返回的
    是一个list，而在python3下它返回的是list的地址，要想得到该list，需要用list(map())强制类型转换
    '''
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))    # 映射所有的元素为 float（浮点数）类型
        dataMat.append(fltLine)
    return dataMat


# 计算两个向量的欧式距离

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

# 为给定数据集构建一个包含K个随机质心的的集合

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))      # 创建一个k行n列的零矩阵（每一列是一个特征，或者说是一个维度）
    for j in range(n):                  # 在本例中，n等于2
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)       # rangeJ是一个list
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)  # random.rand(k,1)生成一个k行的在0到1之间的矩阵
    return centroids


# K-均值聚类算法

def kMeans(dataSet, k, distMeas = distEclud, createCent = randCent):
    '''
    :param dataSet:输入数据集
    :param k:要分类的簇的个数
    :param distMeas:距离度量函数
    :param createCent:簇创建函数
    :return:返回质心和簇分配情况
    该函数的过程如下：
    创建k个点作为起始质心（随机选择）
    当任意一个点的簇分配结果发生变化时：
        对数据集中的每个数据点：
            对每个质心：
                计算质心与数据点之间的距离：
            将数据点分配到距离最近的簇
        对每一个簇，计算簇中所有点的均值，并将均值作为质心
    '''
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))     # 创建二维矩阵记录簇分配情况，分别记录离哪个簇最近，及距离是多少
    centroids = createCent(dataSet, k)      # 创建k个点作为起始质心
    clusterChanged= True                    # 簇分配情况是否发生变化的标记
    while clusterChanged:                   # 当某个点的簇分配发生变化时
        clusterChanged = False
        for i in range(m):                  # 对每个数据点，分别计算其与所有质心的距离
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])       # 计算数据点与质心的距离
                if distJI <minDist:         # 如果当前算的距离小于之前的最小距离
                    minDist = distJI
                    minIndex = j            # 更新该数据点与第j个质心最近
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2 # 将数据点分配到最近的簇
        #print (centroids)
    for cent in range(k):                   # 下面遍历所有质心并更新它们的取值
        ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]    # 通过数组过滤来得到哪些点变化了
        centroids[cent, :] = mean(ptsInClust, axis = 0)     # 将质心更新为簇中所有点的均值，axis=0表示沿矩阵的列方向计算均值
    return centroids, clusterAssment


# 二分K-均值算法

def biKmeans(dataSet, k, distMeas = distEclud):
    '''
    :param dataSet:输入数据集
    :param k:簇的个数
    :param distMeas:距离度量函数
    :return:返回质心列表以及簇分配结果
    该函数的过程为：
    将所有点看做一个簇
    当簇数目小于k时：
        对于每一个簇：
            计算总误差
            在给定的簇上面进行K-均值聚类（k=2）
            计算将该簇一分为二之后的总误差
        选择使得误差最小的那个簇进行划分
    '''
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))     # 创建一个二维矩阵来记录簇的分配情况
    centroid0 = mean(dataSet, axis = 0).tolist()[0]     # 创建一个初始簇（一开始，所有的点属于同一个簇，所以求所有数据的均值）
    centList = [centroid0]      # 创建一个list来保留所有的质心
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2   # 计算误差的平方并存在第二列中
    while (len(centList) < k):  # 当簇的个数没要达到数量要求时不断循环
        lowestSSE = inf         # 将最小SSE（误差平方和）设为无穷大
        for i in range(len(centList)):      # 对于每一个簇
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]  # 选出当前簇i下的所有数据点用来分割
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)  # 在上面选出的簇上进行以2为k的聚类
            sseSplit = sum(splitClustAss[:, i])     # 计算分割出去的数据集误差是多少
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], i])   # 计算剩余数据集的误差是多少
            print ("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:        # 如果上面的分割操作得到的误差小于之前的最小误差，则进行分割
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClutAss = splitClustAss.copy()
                lowestSSE = sseNotSplit + sseSplit
        bestClutAss[nonzero(bestClutAss[:, 0].A == 1)[0], 0] = len(centList)    # 通过数组过滤，来修改簇的编号
        bestClutAss[nonzero(bestClutAss[:, 0].A == 0)[0], 0] = bestCentToSplit  # 被分割的数据集应该被分到哪一个簇里
        #print (bestClutAss)
        print ("the bestClusToSplit is: ", bestCentToSplit)
        print ("the len of bestClustAss is: ", len(bestClutAss))
        centList[bestCentToSplit] = bestNewCents[0, :]  # 更新当前分割的数据集的最好质心
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClutAss  # 更新最好簇下的质心
    return mat(centList), clusterAssment



