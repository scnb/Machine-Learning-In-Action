#-*- coding:utf-8 -*-

from numpy import *
from time import sleep
import json
import urllib.request


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


# 使用Google购物的API收集数据


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = ('get from code.google.com')
    searchURL = ('https://www.googleapis.com/shopping/search/v1/public/products? key = %s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum))
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())         # 获取产品信息组成的字典
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':   # 判断该产品是否是新的
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']  # 抽取货物清单
            for item in listOfInv:
                sellingPrice = item['prices']
                if sellingPrice > origPrc * 0.5:            # 判断该乐高套装是否完整（通过简单的判断当前售价是否大于原始售价的一半）
                    print ("%d\t%d\t%d\t%f\t%f" %(yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])     # 将信息分别保存在list对象retX和retY中
                retY.append(sellingPrice)
        except: print ('problem with item %d' % i)


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


# 交叉验证测试岭回归

def crossValidation(xArr, yArr, numVal = 10):
    '''
    :param xArr:输入数据
    :param yArr:输入数据
    :param numVal:交叉验证次数
    :return:无
    '''
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal, 30))
    for i in range(numVal):
        trainX = []         # 创建训练集
        trainY = []
        testX = []          # 创建测试集
        testY = []
        random.shuffle(indexList)       # 将全体数据随机打乱，以便于后面随机抽取
        for j in range(m):
            if j < m * 0.9:             # 90%的数据用于训练
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:                       # 10%的数据用于测试
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
    wMat = ridgeTest(trainX, trainY)    # 使用岭回归算法求回归系数
    for k in range(30):                 # 在测试集上用30组回归系数来循环测试回归效果
        matTestX = mat(testX)           # 下面五行先用训练时的参数将测试数据标准化
        matTrainX = mat(trainX)
        meanTrain = mean(matTrainX, 0)
        varTrain = var(matTrainX, 0)
        matTestX = (matTestX - meanTrain) / varTrain
        yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)
        errorMat[i, k] = rssError(yEst.T.A, array(testY))   # 用rssError函数来计算预测值和实际值的误差
    meanErrors = mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]
    xMat = mat(xArr)
    yMat = mat(yArr).T
    meanX = mean(xMat, 0)
    varX = var(xMat, 0)
    unReg = bestWeights / varX
    print ("the best model from Ridge Regression is:\n", unReg)
    print ("with constant term: ", -1 * sum(multiply(meanX, unReg)) + mean(yMat))
