#-*- coding:utf-8 -*-
from numpy import *
import re

##定义创建实验样本
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]        ## 1代表侮辱性文字，0代表正常言论
    return postingList,classVec

##创建词汇表
def createVocabList(dataSet):
    vocabSet = set([])                          ##创建一个包含文档中的词并且不重复的列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)     ##使用 | 运算符求两个集合的并集
    return list(vocabSet)

##词表到向量的转换函数——基于词集模型，以每个词的出现与否作为特征
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)        ##创建一个所含元素都为0，并且和词汇表等长的向量
    for word in inputSet:                   ##判断输入文档中的每一个单词是否在词汇表中
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1  ##如果文档中的某个单词在词汇表中，则在输出的文档向量的相应位置写1，表示该单词出现过
        else:
            print ("the word: %s is not in my vocabulary!" % word)
    return returnVec

##词表到向量的转换函数——基于词袋模型，以每个词出现的次数作为特征
def bagOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print ("the word: %s is not in my vocabulary!" % word)
    return returnVec

##朴素贝叶斯(Naive Bayes)分类器训练函数        计算P(wi|c1)、P(wi|c0)和P(c1)（P(c0)=1-P(c1）)
def trainNB0(trainMatrix,trainCategory):    ##分别为每篇文档的测试矩阵（数字矩阵）和每篇文档的分类标签
    numTrainDocs = len(trainMatrix)         ##获得文档数目
    numWords = len(trainMatrix[0])          ##获得每一篇文档中的单词数目(这几篇文档的单词数目是相同的)
    pAbusive = sum(trainCategory)/float(numTrainDocs)   ##计算所有文档中属于侮辱性的概率，即P(c1)(sum计算trainCategory中的元素之和，而trainCategory中元素只有0和1）
    p0Num = ones(numWords)                  ##计算P(wi|c0)的分子变量向量，并初始化为1（初始化为1的目的是防止属性集中的其他属性值被没有出现的属性值（因为没有出现的属性值的频率为0）抹掉）
    p1Num = ones(numWords)                   ##计算P(wi|c1)的分子变量向量，并初始化为1
    p0Denom = 2.0                           ##计算P(wi|c0)的分母变量向量，并初始化为2
    p1Denom = 2.0                           ##计算P(wi|c1)的分母变量向量，并初始化为2
    for i in range(numTrainDocs):           ##遍历训练集中的所有文档
        if trainCategory[i] == 1:           ##如果该篇文档属于侮辱性的文档（前提条件，即条件概率），则对p1Num和p1Denom进行计算
            p1Num += trainMatrix[i]         ##p1Num向量中各个元素加上第i篇文档中各个词条（即各个属性/特征）的数目
            p1Denom += sum(trainMatrix[i])  ##将概率计算式的分母赋值为第i篇文档的词条数目之和
        else:                               ##否则，该文档属于非侮辱性的，则对p0Num和p0Denom进行计算
            p0Num += trainMatrix[i]         ##同if中的语句
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)             ##计算侮辱性文档中每个单词出现的概率，即P(wi|c1)，取log是为了防止很多很小的概率相乘造成下溢变成0
    p0Vect = log(p0Num/p0Denom)             ##计算非侮辱性文档中每个单词出现的概率，即P(wi|c0)
    return p0Vect,p1Vect,pAbusive           ##虽然在该函数中只计算了P(c1)，但是P(c0)可由 1-P(c0)轻松可得

##朴素贝叶斯分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  ##因为取了代数，则P(w|ci)P(ci)变成了logP(w|ci)+logP(ci)
    p0 = sum(vec2Classify * p0Vec) + log(1.0-pClass1)   ##这里实际上是在实现P(w|ci)基于属性独立情况下的连乘运算，P(w|ci)=P(w0|ci)*P(w1|ci)*···*P(wn|ci)
    if p1 > p0:
        return 1
    else:
        return 0

##验证朴素贝叶斯分类器
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))  ##将每一个文档中的所有词表转换成其对应的出现与否数组
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['my','love','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))  ##因为还是基于myVocabList这个词汇表,所以还是32维的向量
    print (testEntry," classified as ",classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))  ##同上
    print (testEntry," classified as ",classifyNB(thisDoc,p0V,p1V,pAb))

##文本解析器
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

##垃圾邮件测试函数
def spamTest():
    docList = []                        ##存放每一篇文章的词向量
    classList = []                      ##存放标签的列表
    fullText = []                       ##用来存放所有文档种的每一个单词
    for i in range(1,26):               ##依次交叉读入25个垃圾邮件和25个非垃圾邮件
        wordList = textParse(open("email/spam/%d.txt" % i).read())  ##将邮件进行解析，存在列表中
        docList.append(wordList)        ##append是把整个列表作为一个元素存在列表中
        fullText.extend(wordList)       ##extend是把原列表中的每个元素存在新列表中
        classList.append(1)             ##标记类型为1
        wordList = textParse(open("email/ham/%d.txt" % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)                     ##标记类型为0
    vocabList = createVocabList(docList)        ##创建词汇表，即将docList中的重复元素剔除掉创建一个单词列表
    trainingSet = list(range(50))               ##一个整数列表(因此必须要强制转成list)，值从0到49，用于挑选训练集的索引
    testSet = []                                ##存放测试集样本的索引
    for i in range(10):                         ##在训练集中随机挑出10个构成测试集，并从训练集找那个删除这10个样本
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])           ##这种随机选择总样本一部分作为训练集，剩下的一部分作为测试集的过程称为——留存交叉验证（hold-out cross validation）
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:            ##利用训练集构建贝叶斯分类器
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorcount = 0
    for docIndex in testSet:                ##对测试集中的每个样本进行测试，并记录错误的样本数量
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorcount += 1
    print ("the error rate is: ", float(errorcount)/len(testSet))

##RSS源分类器及高频词去除函数
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = list(sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True))
    return sortedFreq[0:30]

def localWords(feed1,feed0):
    import feedparser
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:                    ##从词汇表中去除出现频率最高的30个词
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(minLen*2))
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorcount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorcount += 1
    print ("the error rate is ", float(errorcount)/len(testSet))
    return vocabList, p0V, p1V

##打印出最具表征性的词汇
def getTopWords(ny,sf):
    import operator
    vocabList, p0V, p1V = localWords(ny,sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:                             ##返回大于某个阈值的单词，因为之前取了对数，所以阈值为负数
            topSF.append((vocabList[i], p0V[i]))      ##把单词和它出现的频率做为一个元素存在列表中
        if p1V[i] > -6.0:
            topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF,key=lambda pair:pair[1], reverse=True)     ##按元祖中的第一个元素即概率值进行排序
    print ("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF")
    for item in sortedSF:
        print (item[0])     ##输出元组的第一个元素，即单词
    sortedNY = sorted(topNY, key=lambda pair:pair[1], reverse=True)
    print ("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY")
    for item in sortedNY:
        print (item[0])

