from numpy import *
import operator
from os import listdir

##生成数据集和对应的标签
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

##k-近邻算法的实现
def classify0(inX,dataSet,labels,k):    ##dataSet和labels分别为训练样本集及其标签
    dataSetSize = dataSet.shape[0]      ##取得训练样本集的大小
    diffMat = tile(inX,(dataSetSize,1)) - dataSet   ##求训练样本集和未知样本各个特征值间的差值
    sqDiffMat = diffMat**2              ##将上一步求得的差值进行平方
    sqDistances = sqDiffMat.sum(axis=1) ##将上一步求得的平方进行求和
    distances = sqDistances**0.5        ##将上一步求得的平方和进行开方   第11、12、13、14行合起来就是求欧式距离
    sortedDistIndicies = distances.argsort()   ##将上面计算的距离进行从小到大排序
    classCount={}                        ##用一个字典来存储每种标签的出现频率
    for i in range(k):                   ##确定前k个点所在类别的频率
        voteIlabel = labels[sortedDistIndicies[i]]  ##取得第i个训练样本的标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1   ##对每一种标签进行计数
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)   ##对前k个样本的标签频率进行排序
    return sortedClassCount[0][0]   ##返回频率最大的标签

##tile是numpy中的一个，当前这句话的意思在行方向上重复[0,0]dataSetSize次，在列方向上重复[0,0]1次
##argsort()是numpy中的一个方法，返回数组中元素从小到大的索引值
##第21行的dict.get('key',default=none)方法：获取关键字的值，如果不存在则返回默认值none，或设置的值


##将文本记录解析为NumPy的解析程序
def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines =len(arrayOfLines)       ##取得文件的行数
    returnMat = zeros((numberOfLines,3))   ##定义返回矩阵，numberOfLines行，3列，且元素全为0
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()             ##截取掉所有的回车字符
        listFromLine = line.split('\t') ##将每一行以\t(tab)字符分开，分成几部分，存在list中
        returnMat[index,:] = listFromLine[0:3] ##将listFromLines中的前3个元素存储在返回矩阵的一整行里
        classLabelVector.append(int(listFromLine[-1])) ##通过索引-1将listFromLines最后一列的数据存储到标签向量中
        index += 1
    return returnMat,classLabelVector

##进行归一化处理   归一化公式：newValue = (oldValue-min)/(max-min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)    ##取得列中的最小值
    maxVals = dataSet.max(0)    ##取得列中的最大值
    ranges = maxVals - minVals  ##数据的范围
    normDataSet = zeros(shape(dataSet)) ##生成和原始数据集大小相等的全0矩阵
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))     ##用旧的数据集减去最小值
    normDataSet = normDataSet/tile(ranges,(m,1))    ##上一步得到的数据集除以数据的范围
    return normDataSet,ranges,minVals
##为什么要进行归一化处理：因为某些特征本身的取值就比其他的特征大的多，所以在计算式会造成很大的影响

##检验分类器性能的测试代码
def datingClassTest():
    hoRatio = 0.10          ##定义测试集占总体数据集的比例
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')   ##用file2matrix函数解析文本
    normMat,ranges,minVals = autoNorm(datingDataMat)                ##用autoNorm函数进行归一化
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)    ##计算出测试集的数量
    errorCount = 0.0                ##用于计算分类错误的个数
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)   ##调用classify0函数，即使用k-近邻算法
        print ("the classifier came back with: %d, the real answer is: %d" %(classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0       ##若用#classifi0预测的结果和实际结果不同，则将errorCount加1
    print ("the total error rate is: %f" % (errorCount/float(numTestVecs)))

def classifiPerson():
    resultList = ['not at all','in small does','in large does']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    inArr = array([percentTats,ffMiles,iceCream])
    datingDataMat,datingLabels = file2matrix("datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDataMat)
    classifierResult = classify0((inArr-minVals)/ranges,datingDataMat,datingLabels,3)
    print ("You will probably like this person: ",resultList[classifierResult-1])

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline();
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

##手写数字识别系统的测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')    ##通过listdir函数列出给定目录下的文件名
    m = len(trainingFileList)                       ##得到总文件数
    trainingMat = zeros((m,1024))                   ##创建一个m行1024列的矩阵，每一行存储一个图像
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]         ##将文件名以点号分隔开，得到省去文件后缀的文件名
        classNumStr = int(fileStr.split('_')[0])    ##将文件名以短横线分隔开，得到该文件表示是哪个数字
        hwLabels.append(classNumStr)                ##将表示的数字存在标签列表中
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)    ##将trainingDigits文件夹下的每一个图像存在一行中
    testFileList = listdir('testDigits')            ##得到测试文件夹下的所有文件
    errorCount = 0.0                                 ##记录预测错误的数量，用来计算分类器的错误率
    mTest = len(testFileList)                        ##得到测试文件总数
    for i in range(mTest):
        fileNameStr = testFileList[i]               ##每次取一个测试文件
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])    ##测试文件的真实数字
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)     ##将待测试文件存储在数组中
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)    ##在这里，训练样本集中一共有2000个样本，所以待测试样本要与2000个样本计算距离，每次距离计算涉及到1024维浮点数
        print ("the classifier came back with: %d,the real answer is:%d" % (classifierResult,classNumStr))
        if(classifierResult!=classNumStr):
            errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))
