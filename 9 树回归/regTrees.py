#-*- coding:utf-8 -*-

from numpy import *

# CART树构建算法 CART:Classification And Regression Tree 分类和回归数

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))     # 将每一行的数据映射成浮点型
        dataMat.append(fltLine)
    return dataMat

# 二元切分数据集

def binSplitDataSet(dataSet, feature, value):
    '''
    :param dataSet:输入数据集
    :param feature: 待切分的特征
    :param value: 该特征的某个值
    :return:返回切分后的数据
    '''
    # dataSet[:,feature]>value 会将第feature列的元素与value比较大小，并返回结果为True所在行的索引
    # 所以，mat0或者mat1得到的是满足满足条件的列元素所在的那一行
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]     # 通过数组过滤的方式来选择数据
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

# 回归树的切分函数

def regLeaf(dataSet):           # 构建叶节点
    return mean(dataSet[:, -1])

def regErr(dataSet):            # 计算总方差
    return var(dataSet[:, -1]) * shape(dataSet)[0]      # 总方差通过均方差乘以总样本数得到

# 选择最佳切分特征

def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
    '''
    :param dataSet:输入数据集
    :param leafType:创建叶节点的函数
    :param errType:计算误差的函数
    :param ops:一个元组，包含某些可选的参数
    :return:最佳切分特征，及其取值
    该函数的过程如下：
    对每个特征：
        对每个特征值：
            将数据集切分成两部分
            计算切分的误差
            如果当前误差小于当前最小误差，则将当前切分设定为最佳切分，并更新最小误差
    返回最佳切分的特征及其取值
    '''
    # 以下是两个提前终止的条件
    tolS = ops[0]   # 允许的误差下降值      # 这两个实际是用来进行预剪枝的（prepruning）
    tolN = ops[1]   # 切分的最少样本数
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:     # 如果剩余的特征值都相等，则创建叶节点
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)        # 计算当前数据集的误差
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):                      # 对每个特征循环
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):     # 对每个特征值循环，set函数将某一特征的所有特征值存在一个集合中
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)        # 计算划分后的新误差
            if newS < bestS:                                # 如果新误差比较小，则进行更新
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:              # 如果切分后，误差降低的不够多，则创建叶节点并返回
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):      # 检查切分后两个子集的大小，如果子集大小小于用户定义的tolN，则创建叶节点并退出
        return None, leafType(dataSet)
    return bestIndex, bestValue


# 构建树的函数

def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
    '''
    :param dataSet:输入数据集
    :param leafType: 建立叶节点的函数，根据该函数及下面的误差计算函数的不同选择， 可以分别构建回归树和模型树
    :param errType: 误差计算函数
    :param ops: 包含树构建所需其他参数的元组
    :return:
    该函数的过程如下：
    找到最佳的待切分特征：
        如果该节点不能再分，则将该节点存为叶节点
        执行二元切分
        在右子树调用createTree()方法
        在左子树调用createTree()方法
    '''
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)    # 找到最佳切分特征
    if feat == None:    # 如果没有最佳特征了，即不能再分了
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree



# 回归树剪枝函数

def isTree(obj):
    '''
    :param obj:输入对象
    :return: bool类型，true or false
    用来判断该对象是不是一棵树，如果是树的话，其数据结构就是字典
    '''
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    '''
    :param tree:输入一棵树
    :return:递归计算两个叶节点的均值
    '''
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

# 树的后剪枝函数

def prune(tree, testData):
    '''
    :param tree:输入一棵树
    :param testData:测试数据集
    :return:
    该函数过程如下：
    基于已有的树切分测试数据：
        如果存在任一子集是一棵树，则在该子集递归剪枝过程
        计算将当前两个叶节点合并后的误差
        计算不合并的误差
        如果合并后的误差更小，那些就将叶节点合并
    '''
    if shape(testData)[0] == 0:     # 如果没有测试数据，则对树进行塌陷处理（即求树的平均值）
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])    # 将测试数据按照特征和特征值分成两份（因为树当初是按照二元切分构建的）
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)

    if not isTree(tree['left']) and not isTree(tree['right']):      # 如果当前节点下面没有树，而只有叶节点时，考虑能不能合并
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))   # 计算不合并的误差
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))      # 计算合并后的误差
        if errorMerge < errorNoMerge:                               # 如果合并使得误差更小，则执行合并
            print ("merging")
            return treeMean
        else:
            return tree                                             # 如果不能合并则直接返回
    else:
        return tree

# 模型树的叶节点构建函数

def linearSolve(dataSet):       # 将数据集格式化成自变量X和目标标量Y
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]       # 将X0设为1，其余的元素与原数据集中的前n-1列数据相同
    Y = dataSet[:, -1]                  # 把原数据集的最后一列元素赋给Y
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse.\n try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

def modelLeaf(dataSet):         # 负责生成叶节点的线性模型的回归系数
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):          # 用于计算线性模型和实际数据的误差
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


# 用树回归进行预测

def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(X * model)

def treeForeCast(tree, inData, modelEval = regTreeEval):
    '''
    :param tree:输入一棵树
    :param inData: 输入测试数据
    :param modelEval: 选择是对回归树叶进行预测，还是对模型树叶进行预测
    :return:
    '''
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

# 使用训练好的树，在测试集上进行测试

def createForeCast(tree, testData, modelEval = regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat

# 在测试集上测试完成之后，会得到预测值yHat，可通过corrcoef函数，来对yHat和实际值y进行相关性分析，越接近1则相关性越好