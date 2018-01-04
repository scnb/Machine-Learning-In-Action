#-*- coding:utf-8 -*-
from numpy import *

# 导入一个简单数据集

def loadSimpData():
    dataMat = matrix([[ 1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2, 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


# 自适应数据加载函数

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))    # 得到一行中有几组数据（以制表符为分隔符）
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')      # 将当前行以制表符为分隔符分开
        for i in range(numFeat - 1):            # 将当前行中的几组数据（除了最后一组）都存到lineArr向量中
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)                 # 将数据存到数据列表中
        labelMat.append(float(curLine[-1]))     # 最后一组数据是标签，存到标签列表中
    return dataMat, labelMat

# 单层决策树（decision stump，决策树桩）生成函数

# 通过将样本的特征值与阈值进行比较，来对样本进行预测
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':                                  # 测试是否有某个值小于或者大于我们正在测试的阈值
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0  # threshVal是阈值，通过数组过滤的方式，将数据集中小于阈值的位置设置为-1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = 1.0   # 将数据集中大于阈值的位置设为1
    return retArray


def buildStump(dataArr, classLabels, D):
    '''
    该程序执行过程：
    将最小错误率minError设为正无穷：
    对数据集中的每一个特征：
        对每个步长：
            对每个不等号：
                建立一棵单层决策树并利用加权数据集对他进行测试
                如果错误率低于minError，则将当前决策树设置为最佳单层决策树
    返回最佳单层决策树
    '''
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)        # m个样本，每个样本有n个特征
    numSteps = 10.0
    bestStump = {}                  # 用来记录最好的树桩，是一个字典
    bestClasEst = mat(zeros((m, 1)))
    minError = inf                  # 将最小错误率设置为无穷大
    for i in range(n):              # 在数据的所有特征上遍历
        rangeMin = dataMatrix[:, i].min()   # 找到所有样本的第i个特征的最小值
        rangeMax = dataMatrix[:, i].max()   # 找到所有样本的第i个特征的最大值
        stepSize = (rangeMax - rangeMin) / numSteps # 计算在该特征组成的数轴上，以多大的步长来取阈值
        for j in range(-1, int(numSteps)+1):        # 对于可能取到的每一种步长，进行循环
            for inequal in ['lt', 'gt']:            # 每一个特征值与阈值存在两种不等关系：小于或等于，分别进行循环
                threshVal = (rangeMin + float(j) * stepSize) # 计算阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal) # 利用单层决策树预测标签
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0   # 通过数组过滤，如果预测的标签与真实标签不同，则在errArr相应位置1
                weightedError = D.T * errArr            # 计算加权错误率
                print ("split:dim %d, thresh %.2f, thresh ineqal: %s the weigthed error is %.3f" %(i, threshVal, inequal, weightedError))
                if weightedError <minError:             # 如果当前加权错误率比最小错误率小，则更新最佳单层决策树的相关信息
                    minError = weightedError
                    bestClasEst = predictedVals.copy()  # 将预测值复制，最后返回
                    bestStump['dim'] = i                # 用字典用记录最好的树桩
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


# 完整AdaBoost算法的实现——基于单层决策树的AdaBoost训练过程

def adaBoostTrainDS(dataArr, classLabels, numIt = 9):      # DS表示Decision Stump
    '''
    该函数的过程为：
    对每次迭代：
        利用buildStump()函数找到最佳的单层决策树
        将最佳单层决策树加入到单层决策树组（构建多个分类器）
        计算alpha（alpha是每个分类器的权重值）
        计算新的权重向量D
        更新累计类别估计值
        如果错误率等于0.0，则退出循环
    '''
    weakClassArr = []                      # 弱分类器数组
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)              # 一开始，数据集的权重D被设置成相等
    aggClassESt = mat(zeros((m, 1)))       # 列向量，记录每个数据点的类别估计累计值
    for i in range(numIt):                 # numIt是由用户指定的迭代次数
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)       # 每次循环中建立一个新的单层决策树，输入数据集权重D，输出具有最小错误率的单层决策树及其错误率
        print ("D:", D.T)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))            # 计算新的分类器权值α，max(error,1e-16)用来防止error太小而发生除零溢出
        bestStump['alpha'] = alpha         # 将alpha加入到字典中
        weakClassArr.append(bestStump)     # 将本次循环训练的单层决策树加入到弱分类器数组中
        print ("classEst: ", classEst.T)   # classEst是本次循环中单层决策树的预测标签值
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)     # 这里使用了一个小技巧：正确分类的，指数应该是负数，那么classLabels==classEst，得到1，剩下-1，使指数称为负数
        D = multiply(D, exp(expon))
        D = D / D.sum()                    # 更新样本的权重D
        aggClassESt += alpha * classEst    # 为什么乘以alpha呢：正因为是AdaBoost算法， 就是靠所有分类器的预测结果的加权和来进行判断是否分类正确
        print ("aggClassEst: ", aggClassESt.T)
        aggErrors = multiply(sign(aggClassESt) != mat(classLabels).T, ones((m, 1))) # 通过sign函数，取得总体分类情况的正负，然后与实际标签比较计算错误率
        errorRate = aggErrors.sum() / m     # 计算分类错误率,aggErrors是一个列向量，对应着每个样本是否被分类正确
        print ("total error: ", errorRate, "\n")
        if errorRate == 0.0:                # 如果训练错误率为0，则退出循环
            break
    return weakClassArr, aggClassESt


# 实现测试算法：基于AdaBoost的分类

def adaClassify(dataToClass, classifierArr):        # dataToClass即需要分类的数据
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]                        # 取得待分类样本的数量
    aggClassESt = mat(zeros((m, 1)))                # 初始化总体分类结果向量
    for i in range(len(classifierArr)):             # 对于所有的分类器进行循环，即每种分类器运行一次
        ClassEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassESt += classifierArr[i]['alpha'] * ClassEst     # 取得总体分类器的加权预测结果
        print (aggClassESt)             # 在实际应用时，可以注释掉输出中间过程
    return sign(aggClassESt)            # 输出时应该输出的是总体分类结果的符号，以表示正类或者负类



# ROC曲线的绘制即AUC计算函数（ROC:接收者操作特征（Receiver Operating Charactersitic），AUC：曲线下的面积）

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)    # 画笔的当前位置，在右上角
    ySum = 0.0
    numPosClas = sum(array(classLabels) == 1.0)         # 通过数组过滤方法来获得正例样本的数量
    yStep = 1 / float(numPosClas)                       # 计算y轴上的步进长度，因为y轴是真正例
    xStep = 1 / float(len(classLabels) - numPosClas)    # 同上，计算x轴上的步进长度，因为x是假正例
    sortedIndicies = predStrengths.argsort()
    # 以下三行代码用来构建画笔
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:            # 如果该样本的实际是正例
            delX = 0
            delY = yStep                         # 则在y轴上下降一个步长（因为起点在右上角）
        else:
            delX = xStep                         # 则在x轴上下降一个步长（因为起点在右上角）
            delY = 0
            ySum += cur[1]                        # 为了计算AUC，因为所有的小矩形的宽度都是xStep，所以只需把所有矩形的高度加起来
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')    # 画实线
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0, 1], [0, 1], 'b--')                # 画虚线
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print ("the Area Under The Curve is: ", ySum * xStep)


