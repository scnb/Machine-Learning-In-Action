#-*- coding:utf-8 -*-

from numpy import *


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) -1     # 计算一行中有几组数据（除去标签）
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):                # 因为numFeat是除去标签一行中有几组数据，所以这个for循环将数据取出来，存在dataMat中
            lineArr.append(float(curLine[i]))   # 文本中数据都是浮点型的，再读取的时候也要保持浮点型
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))     # 文本中数据都是浮点型的，再读取的时候也要保持浮点型
    return dataMat, labelMat


def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat           # 求x转置乘以x
    if linalg.det(xTx) == 0.0:    # 判断上面求得的矩阵是否存在逆（通过计算行列式是否为0）
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * xMat.T * yMat
    return ws

# 局部加权线性回归（Locally Weighted Linear Regression）

# 单个点的预测函数

def lwlr(testPoint, xArr, yArr, k = 1.0):   # k用来控制权重的衰减速度，testPoint是单个测试点
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m))           # 创建权重矩阵（对角矩阵）
    for j in range(m):              # 一行一行地计算
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k**2))        # 对角矩阵只有对角线上有非零元素
    xTx = xMat.T * (weights * xMat) # 新的计算回归系数的公式
    if linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

# 对数据集中的每个点都进行预测

def lwlrTest(testArr, xArr, yArr, k = 1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


# 预测误差计算函数

def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()


# 岭回归

# 该函数用于计算回归系数

def ridgeRegres(xMat, yMat, lam = 0.2):  # lam即λ
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:                    # 虽然岭回归一定可逆，但是λ被设置为0时，就是个例外情况
        print ("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * xMat.T * yMat
    return ws

#

def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # 下面进行数据标准化
    yMean = mean(yMat, 0)
    yMat = yMat - yMean         # 进行均值归一化
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar   # 将数据减去均值后，再除以方差
    numTesPts = 30                  # 计算30组λ值
    wMat = zeros((numTesPts, shape(xMat)[1]))
    for i in range(numTesPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))       # λ以指数级递增
        wMat[i, :] = ws.T
    return wMat


# 数据标准化处理函数

def regularize(xMat):# 按列进行规范化
    '''
    :param xMat:输入数据
    :return: 返回处理后均值为0，方差为1的标准化数据
    '''
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)   # 计算平均值然后减去它
    inVar = var(inMat, 0)      # 计算除以Xi的方差
    inMat = (inMat - inMeans)/inVar
    return inMat

# 前向逐步回归算法

'''
    该算法的过程如下：
    在每轮迭代中：
        设置当前最小误差lowestError为正无穷
        对每个特征：
            增大或缩小：
                改变一个系数得到一个新的W
                计算新W下的误差
                如果误差Error下于当前最小误差lowestError：设置Wbest等于当前的W
            将W设置为新的Wbest
'''

def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    '''
    :param xArr:输入数据
    :param yArr:预测变量
    :param eps:每次迭代需要调整的步长
    :param numIt:迭代次数
    :return:返回最优的权重矩阵
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1))
    wsTest = ws.copy()                  # 权重矩阵
    wsMax = ws.copy()
    for i in range(numIt):              # 迭代numIt次
        print (ws.T)
        lowestError = inf;
        for j in range(n):              # 对每个特征
            for sign in [-1, 1]:        # 对每种符号，即对权重增大还是缩小
                wsTest = ws.copy()
                wsTest[j] += eps * sign # 改变权重
                yTest = xMat * wsTest   # 计算改变权重之后的新的预测值
                rssE = rssError(yMat.A, yTest.A)        # 计算预测值与实际值的误差
                if rssE < lowestError:                  # 若本次的误差比最小误差还小，则进行更新
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat
