##使用了ID3算法

from math import log
import operator

def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

##计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)       ##取得数据集中数据的个数
    labelCounts = {}                ##创建标签字典，标签最为键，标签出现的次数作为值
    for featVec in dataSet:         ##对于数据集中的每一个数据
        currentLabel = featVec[-1]  ##标签在数据的最后一个元素中存着
        if currentLabel not in labelCounts.keys():  ##如果当前字典的键中没有该标签
            labelCounts[currentLabel] = 0           ##则在字典中添加该标签，并赋值为0
        labelCounts[currentLabel] += 1              ##对当前标签的值加1
    shannonEnt = 0.0                                ##初始化香农熵
    for key in labelCounts:                         ##对于每一个标签（分类）
        prob = float(labelCounts[key]/numEntries)       ##计算该类别出现的概率
        shannonEnt -= prob*log(prob,2)                  ##计算香农熵
    return shannonEnt


##按照给定特征划分数据集
def splitDataSet(dataSet,axis,value):
    retDataSet = []                     ##创建新的列表，防止修改原始数据
    for featVec in dataSet:             ##对于数据集中的每一个数据
        if featVec[axis] == value:      ##如果数据中的特征与给定的值相等
            reducedFeatVec = featVec[:axis]         ##将数据中从0到axis-1的元素取出来
            reducedFeatVec.extend(featVec[axis+1:]) ##将数据中从axis+1到末尾的元素取出来
            retDataSet.append(reducedFeatVec)       ##将上面取得的数据作为一个列表放到返回数据集中
    return retDataSet

##选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1           ##取得数据中一共有多少中特征
    baseEntropy = calcShannonEnt(dataSet)       ##计算原始数据集的香农熵
    bestInfoGain = 0.0
    bestFeature = -1                            ##最好的划分特征的索引
    for i in range(numFeatures):                ##尝试根据每一种特征进行划分
        featList = [example[i] for example in dataSet]  ##将数据集中的第i种特征的值都取出来
        uniqueVals = set(featList)                      ##通过python内置的set函数将featList中重复值去掉，构建唯一的特征值列表
        newEntropy = 0.0                                ##定义新的熵变量为0
        for value in uniqueVals:                        ##按照某一特征值划分子集后，计算所有子集的信息熵
            subDataSet = splitDataSet(dataSet,i,value)  ##先根据特征进行子集划分
            prob = len(subDataSet)/float(len(dataSet))  ##计算按照该特征划分后的子集的概率
            newEntropy += prob*calcShannonEnt(subDataSet)   ##计算划分后每一个子集的熵
        infoGain = baseEntropy - newEntropy                 ##计算划分子集后的信息增益
        if(infoGain > bestInfoGain):                        ##判断信息增益是否为最大
            bestInfoGain = infoGain                         ##如果是，则更新最好的信息增益，并更新最好的划分特征的索引
            bestFeature = i
    return bestFeature                                      ##返回最大的特征的索引

##多数表决程序
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)   ##对于python3，是items()
    return sortedClassCount[0][0]


##通过递归的方法构建决策树
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]            ##取得数据集中每个数据的标签
    if classList.count(classList[0]) == len(classList):         ##判断分类列表中是否只有一种标签，作为递归停止的条件，即分支下的所有实例都有相同的条件
        return classList[0]
    if len(dataSet[0]) == 1:                                    ##递归停止的第二个条件，当遍历完所有的特征后，仍不能将数据集划分成仅包含唯一类别的分组
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)                ##选择最好的划分特征
    bestFeatLabel = labels[bestFeat]                            ##取得上一步是按照哪个特征划分的
    myTree = {bestFeatLabel:{}}                                 ##通过字典来创建树，以上面找到的最优划分特征为键值存在树（字典）中
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]     ##从数据集中取出每一个数据对应这个特征的属性值
    uniqueVals = set(featValues)                                ##通过set函数得到包含唯一特征值的集合（即剔除上面列表中的重复值）
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

##使用决策树的分类函数
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]                    ##取得决策树的第一个节点的特征
    secondDict = inputTree[firstStr]                        ##获得第一个节点的所有分支
    featIndex = featLabels.index(firstStr)                  ##.index方法返回第一个节点特征在标签列表中的索引
    for key in secondDict.keys():                           ##对于每一个子节点对应的特征
        if testVec[featIndex]==key:                         ##比较testVec变量中的值与树节点的值，如果相等，则往这个方向上继续寻找
            if type(secondDict[key]).__name__=='dict':      ##如果该子节点也是一个字典，那么就递归调用
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]               ##否则就说明该节点是叶子节点，则返回该叶子节点的分类标签
    return classLabel

##通过序列化将决策树存储在硬盘上
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')                ##对于python3代码，必须要写成wb
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,'rb')                ##对于python3代码，必须要写成rb
    return pickle.load(fr)