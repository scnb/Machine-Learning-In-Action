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

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup): # kTup是一个包含核函数信息的元组
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m= shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))        # m*1的矩阵
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))        # 误差缓存 m*2的矩阵，第一列是是否有效的标志位，第二列是E值
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def calcEk(oS,k):                                   # 该函数计算对于给定的α值对应的E值
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] + oS.b)
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
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS,j)
    return j, Ej

def updateEk(oS,k):                                 # 计算误差值并存入缓存当中
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]

# 完整版Platt SMO算法中的优化例子

def innerL(i, oS):
    """innerL
    内循环代码
    Args:
        i   具体的某一行
        oS  optStruct对象

    Returns:
        0   找不到最优的值
        1   找到了最优的值，并且oS.Cache到缓存中
    """

    # 求 Ek误差：预测值-真实值的差
    Ei = calcEk(oS, i)

    # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
    # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
    # 表示发生错误的概率：labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
    '''
    # 检验训练样本(xi, yi)是否满足KKT条件
    yi*f(i) >= 1 and alpha = 0 (outside the boundary)
    yi*f(i) == 1 and 0<alpha< C (on the boundary)
    yi*f(i) <= 1 and alpha = C (between the boundary)
    '''
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 选择最大的误差对应的j进行优化。效果更明显
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接return 0
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            # print("L==H")
            return 0

        # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
        # 参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # changed for kernel
        if eta >= 0:
            print("eta>=0")
            return 0

        # 计算出一个新的alphas[j]值
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # 并使用辅助函数，以及L和H对其进行调整
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新误差缓存
        updateEk(oS, j)

        # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            # print("j not moving enough")
            return 0

        # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # 更新误差缓存
        updateEk(oS, i)


        # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
        # w= Σ[1~n] ai*yi*xi => b = yi- Σ[1~n] ai*yi(xi*xj)
        # 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
        # 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

# 完整版Platt SMO外循环代码

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    """
    完整SMO算法外循环，与smoSimple有些类似，但这里的循环退出条件更多一些
    Args:
        dataMatIn    数据集
        classLabels  类别标签
        C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
            可以通过调节该参数达到不同的结果。
        toler   容错率
        maxIter 退出前最大的循环次数
        kTup    包含核函数信息的元组
    Returns:
        b       模型的常量值
        alphas  拉格朗日乘子
    """

    # 创建一个 optStruct 对象
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    # 循环遍历：循环maxIter次 并且 （alphaPairsChanged存在可以改变 or 所有行遍历一遍）
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0

        #  当entireSet=true or 非边界alpha对没有了；就开始寻找 alpha对，然后决定是否要进行else。
        if entireSet:
            # 在数据集上遍历所有可能的alpha
            for i in range(oS.m):
                # 是否存在alpha对，存在就+1
                alphaPairsChanged += innerL(i, oS)
                # print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1

        # 对已存在 alpha对，选出非边界的alpha值，进行优化。
        else:
            # 遍历所有的非边界alpha值，也就是不在边界0或C上的值。
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                # print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1

        # 如果找到alpha对，就优化非边界alpha值，否则，就重新进行寻找，如果寻找一遍 遍历所有的行还是没找到，就退出循环。
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas

# w的计算

def calcWs(alphas, dataArr, classLabels):
    """
    基于alpha计算w值
    Args:
        alphas        拉格朗日乘子
        dataArr       feature数据集
        classLabels   目标变量数据集

    Returns:
        wc  回归系数
    """
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

# 核转换函数

def kernelTrans(X, A, kTup): # kTup是一个元组，包含了核函数的信息，第一个参数是描述所用核函数类型的一个字符串，其他2个参数是核函数的可选参数
    m, n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A           # 对于矩阵的每个元素计算高斯函数的值
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1]**2))
    else:
        raise NameError('Houston We Have a Problem -- That kernel is not recognized')
    return K

# 使用径向基函数（Radial Basis Function）RBF

def testRbf(k1 = 1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = mat(dataArr)                  # 转为nunmpy中的矩阵形式
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]        # 得到alphas中大于0的数的索引
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print ("There are %d Support Vectors" % (shape(sVs)[0]))
    m,n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print ("The training error rate is: %f" % (float(errorCount/m)))
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print ("The training error rate is: %f" % (float(errorCount/m)))