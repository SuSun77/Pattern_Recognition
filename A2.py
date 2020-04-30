import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random

#using pandas to read dataset and rename the class label
dataset1 = pd.read_csv('/Users/sunsu/Desktop/tic-tac-toe.data', names = ['TL', 'TM', 'TR', 'ML', 'MM', 'MR', 'LL', 'LM', 'LR', 'Class'])
dataset2 = pd.read_csv('/Users/sunsu/Desktop/wine.data', header = None)

dataset2 = dataset2[[1,2,3,4,5,6,7,8,9,10,11,12,13,0]]
dataset2.columns = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                'OD280/OD315 of diluted wines', 'Proline', 'Class']

#calculate entropy and information gain
def calcIG(data):
    IG = list()
    numSample = data.shape[0]
    satInfo = data.iloc[:, -1].value_counts()
    p = satInfo / numSample
    initEnt = sum(-p * np.log2(p))
    numAttribute = data.shape[1] - 1
    for i in range(numAttribute):
        attributeValue = data.iloc[:,i].value_counts().index
        attributeEnt = 0
        for j in attributeValue:
            avData = data[data.iloc[:, i] == j]
            numSample = avData.shape[0]
            satInfo = avData.iloc[:, -1].value_counts()
            p = satInfo / numSample
            avEnt = sum(-p * np.log2(p))
            attributeEnt = attributeEnt + (avData.shape[0] / data.shape[0]) * avEnt
        IG.append(initEnt - attributeEnt)
    return IG

#calculate gain ratio for question 2.2
def calcGR(data):
    attributeInfo = []
    IG = calcIG(data)
    numSample = data.shape[0]
    satInfo = data.iloc[:, -1].value_counts()
    p = satInfo / numSample
    originalEnt = sum(-p * np.log2(p))
    numAttribute = data.shape[1] - 1
    for i in range(numAttribute):
        attributeValue = data.iloc[:,i].value_counts().index
        attributeEnt = 0
        satInfo = data.iloc[:, i].value_counts()
        p = satInfo / numSample
        iiEnt = sum(-p * np.log2(p))
        attributeInfo.append(iiEnt)
        for j in attributeValue:
            avData = data[data.iloc[:, i] == j]
            satInfo = avData.iloc[:, -1].value_counts()
            p = satInfo / avData.shape[0]
            avEnt = sum(-p * np.log2(p))
            attributeEnt = attributeEnt + (avData.shape[0] / data.shape[0]) * avEnt
        IG.append(originalEnt - attributeEnt)
        IGR = lambda x, y: 0 if x == y and x == 0 else x / y
        gainRatio0 = map(IGR, IG, attributeInfo)
        gainRatio = list(gainRatio0)
    return gainRatio

#construct decision tree in recursive way
def buildTree(data, algorithm):
    attributeList = list(data.columns)
    res = data.iloc[:, -1].value_counts()
    if data.shape[1] == 1 or res.values[0] == data.shape[0]:
        return res.index[0]
    IGList = algorithm(data)
    maxIGIndex = IGList.index(max(IGList))
    attributeName = attributeList[maxIGIndex]
    decisionTree = {attributeName: {}}
    attributeList.remove(attributeName)
    values = data.iloc[:, maxIGIndex].value_counts().index

    for value in values:
        if data.iloc[:,maxIGIndex].value_counts()[value] == 0:
            decisionTree[attributeName][value] = res.index[0]
        else:
            decisionTree[attributeName][value] = buildTree(data.loc[data[data.columns[maxIGIndex]] == value, :].drop(data.columns[maxIGIndex], axis=1), algorithm)
    return decisionTree

#determine whether the input data is a dictionary branch
def determineType(branches, twig):
    branchesType = type(branches[twig])
    if branchesType == dict:
        return True
    else:
        return False

#classify test dataset using decision tree
def classify(tree, classLabel, test, train):
    if type(tree) == str:
        return tree
    maxClassLabel = train.iloc[:, -1].value_counts().index[0]
    fatherNode = list(tree.keys())[0]
    branches = tree[fatherNode]
    for twig in branches.keys():
        if test[classLabel.index(fatherNode)] == twig:
            if determineType(branches, twig):
                if classify(branches[twig], classLabel, test, train) is None:
                    return maxClassLabel
                else:
                    return classify(branches[twig], classLabel, test, train)
            else:
                return branches[twig]
        else:
            continue

#calculate the accuracy for each dataset
def calcAccuracy(dataset):
    sameData = sum(dataset.iloc[:,-1] == dataset.iloc[:,-2])
    accuracy = sameData / dataset.shape[0]
    return accuracy

#plot confusion matrix
def plotConfusionMatrix(data):
    sample = data.iloc[:, -2]
    res = data.iloc[:, -1]
    return confusion_matrix(sample, res)

#calculate threshold for the second dataset, helping further discrete
def calcThreshold(data):
    numSample = data.shape[0]
    satInfo = data.iloc[:, -1].value_counts()
    p = satInfo / numSample
    originalEnt = sum(-p * np.log2(p))
    numSample = data.shape[0]
    numAttribute = data.shape[1] - 1
    headerName = data.columns
    thresholdList = []
    for j in range(numAttribute):
        column = data.iloc[:, [j,-1]]
        column = column.sort_values(headerName[j])
        IGList = []
        for i in range(numSample):
            preEnt = calcIG(column.iloc[:i+1, :])
            postEnt = calcIG(column.iloc[i+1:, :])
            conditionalEnt = ((i + 1) * preEnt[0] / numSample) + ((numSample - i - 1) * postEnt[0] / numSample)
            IGList.append(originalEnt - conditionalEnt)
        index = IGList.index(max(IGList))
        threshold = (column.iloc[index, 0] + column.iloc[index + 1, 0]) / 2
        thresholdList.append(threshold)
    return thresholdList

#separate the second dataset's data into discrete value
def separateData(thresholdList, data):
    if len(thresholdList) == (data.shape[1] - 1):
        headerName = list(data.columns)
        newData = pd.DataFrame(np.arange(data.shape[0] * data.shape[1]).reshape(data.shape[0], data.shape[1]))
        newData.iloc[:, -1] = data.iloc[:, -1]
        newData.columns = headerName
        numAttribute = data.shape[1] - 1
        for i in range(numAttribute):
            column = data.iloc[:,i]
            largerIndex = column[column > thresholdList[i]].index
            smallerIndex = column[column <= thresholdList[i]].index
            newData.iloc[largerIndex, i] = 1
            newData.iloc[smallerIndex, i] = 0
        return newData

#return the accuracy and confusion matrix list based on cross validation
def calcList(k, data, algorithm):
    data = shuffle(data)
    numSamples = data.shape[0]
    numCV = round(numSamples / k)
    accuracyList = []
    confusionMatrixList = []
    for i in range(k):
        cvFold = data.iloc[i * numCV:(i + 1) * numCV,:]
        trainFold = pd.concat([data.iloc[0:i * numCV,:], data.iloc[(i + 1) * numCV:,:]])
        numTestSample = cvFold.shape[0]
        tree = buildTree(trainFold, algorithm)
        labels = list(trainFold.columns)
        result = []
        for item in range(numTestSample):
            sample = cvFold.iloc[item, :-1]
            result.append(classify(tree, labels, sample, trainFold))
        cvFold.insert(cvFold.shape[1], 'predict', result)
        cvFold = cvFold
        accuracyList.append(calcAccuracy(cvFold))
        confusionMatrixList.append(plotConfusionMatrix(cvFold))
    return accuracyList, confusionMatrixList

#return the accuracy and confusion matrix list based on 10-times-10-fold cross validation
def tentenCVList(k, data, algorithm):
    accuracyList = []
    confusionMatrixList = []
    for i in range(10):
        accuracy, confusionMatrix = calcList(k, data, algorithm)
        accuracyList.append(accuracy)
        confusionMatrixList.append(confusionMatrix)
    return accuracyList, confusionMatrixList

#calculate the mean of accuracy
def calcMean(accuracyList):
    mean10CVFoldList = list(map(lambda item: np.mean(item), accuracyList))
    return np.mean(mean10CVFoldList)

#calculate the variance of accuracy
def calcVariance(accuracyList):
    var10CVFoldList = list(map(lambda item: np.var(item), accuracyList))
    return np.mean(var10CVFoldList)

#only output the max accuracy
def maxAccuracy(accuracyList):
    Max = []
    maxIndex = []
    for eachlist in accuracyList:
        Max.append(max(eachlist))
        maxIndex.append(eachlist.index(max(eachlist)))
    return [Max.index(max(Max)), maxIndex[Max.index(max(Max))]]

#get thresholdlist
print('calculating threshold list, which is very time-consuming')
thresholdList = calcThreshold(dataset2)

#dataset1 using information gain
print('IG1 begin, which will cost over 10 mins')
accuracyIG1, confusionMatrixIG1 = tentenCVList(10, dataset1, calcIG)
indexIG1 = maxAccuracy(accuracyIG1)
confusionMatrix1 = confusionMatrixIG1[indexIG1[0]][indexIG1[1]]
f1, ax1 = plt.subplots()
sns.heatmap(confusionMatrix1, square = True, annot = True, ax = ax1)
ax1.set_xticklabels(['Negative', 'Positive'])
ax1.set_yticklabels(['Negative', 'Positive'])
ax1.set_title('Confusion Matrix of IG1')
ax1.set_xlabel('Prediction')
ax1.set_ylabel('Ground Truth')
plt.show()
print('The mean and variance of the accuracy for IG1 are:\n%f and %f' % (calcMean(accuracyIG1), calcVariance(accuracyIG1)))

#dataset1 using gain ratio
print('GR1 begin, which will cost over 10 mins')
accuracyGR1, confusionMatrixGR1 = tentenCVList(10, dataset1, calcGR)
indexGR1 = maxAccuracy(accuracyGR1)
confusionMatrix1 = confusionMatrixGR1[indexGR1[0]][indexGR1[1]]
f2, ax2 = plt.subplots()
sns.heatmap(confusionMatrix1, square = True, annot = True, ax = ax2)
ax1.set_xticklabels(['Negative', 'Positive'])
ax1.set_yticklabels(['Negative', 'Positive'])
ax2.set_title('Confusion Matrix of GR1')
ax2.set_xlabel('Prediction')
ax2.set_ylabel('Ground Truth')
plt.show()
print('The mean and variance of the accuracy for GR1 are:\n%f and %f' % (calcMean(accuracyGR1), calcVariance(accuracyGR1)))

#separate the data in dataset2 into discrete value
wineDis = separateData(thresholdList, dataset2)

#dataset2 using information gain
print('IG2 begin, which will cost about 2 mins')
accuracyIG2, confusionMatrixIG2 = tentenCVList(10, wineDis, calcIG)
indexIG2 = maxAccuracy(accuracyIG2)
confusionMatrix1 = confusionMatrixIG2[indexIG2[0]][indexIG2[1]]
f3, ax3 = plt.subplots()
sns.heatmap(confusionMatrix1, square = True, annot = True, ax = ax3)
ax3.set_title('Confusion Matrix of IG2')
ax3.set_xticklabels(['3', '2', '1'])
ax3.set_yticklabels(['3', '2', '1'])
ax3.set_xlabel('Prediction')
ax3.set_ylabel('Ground Truth')
plt.show()
print('The mean and variance of the accuracy for IG2 are:\n%f and %f' % (calcMean(accuracyIG2), calcVariance(accuracyIG2)))

wineDis = separateData(thresholdList, dataset2)
#dataset2 using gain ratio
print('GR2 begin, which will cost about 2 mins')
accuracyGR2, confusionMatrixGR2 = tentenCVList(10, wineDis, calcGR)
indexGR2 = maxAccuracy(accuracyGR2)
confusionMatrix1 = confusionMatrixGR2[indexGR2[0]][indexGR2[1]]
f4, ax4 = plt.subplots()
sns.heatmap(confusionMatrix1, square = True, annot = True, ax = ax4)
ax4.set_xticklabels(['3', '2', '1'])
ax4.set_yticklabels(['3', '2', '1'])
ax4.set_title('Confusion Matrix of GR2')
ax4.set_xlabel('Prediction')
ax4.set_ylabel('Ground Truth')
plt.show()
print('The mean and variance of the accuracy for GR2 are:\n%f and %f' % (calcMean(accuracyGR2), calcVariance(accuracyGR2)))

#add attribute noise for question 3
def addAttributeNoise(data, L):
    numAttribute = data.shape[1] - 1
    numSample = data.shape[0]
    numL = round(L * numSample)
    if numAttribute == 13:
        valuesInAttribute = [0, 1]
    else:
        valuesInAttribute = ['o', 'x', 'b']
    data = shuffle(data)
    newData = data.iloc[:,:].copy()
    for i in range(numL):
        attributeRandomIndex = random.randint(0, numAttribute - 1)
        valuesInAttribute = list(data.iloc[:, attributeRandomIndex].value_counts().index)
        valueRandomIndex = random.randint(0, len(valuesInAttribute) - 2)
        valuesInAttribute.remove(data.iloc[i, attributeRandomIndex])
        noise = valuesInAttribute[valueRandomIndex]
        newData.iloc[i, attributeRandomIndex] = noise
    return shuffle(newData)

#add attribute noise for question 3
def addClassNoise(data, L, sources):
    newdata = data.iloc[:,:].copy()
    numSample = newdata.shape[0]
    LSampleCount = round(numSample * L)
    classLabelList = list(data.iloc[:,-1].value_counts())
    shuffle(newdata)
    if sources == 'con':
        newdata.drop(newdata.head(LSampleCount).index)
        noises = newdata.head(LSampleCount).copy()
        for sample in noises:
            originalValue = newdata.iloc[:,-1]
            classLabelList.remove(originalValue)
            newdata.iloc[:, -1] = classLabelList[0]
            classLabelList.append(originalValue)
        newdata.extend(noises)
        return newdata
    else:
        for sample in newdata.head(LSampleCount):
            originalValue = newdata.iloc[:,-1]
            classLabelList.remove(originalValue)
            newdata.iloc[:, -1] = classLabelList[0]
            classLabelList.append(originalValue)
        return newdata

#return the average and variance accuracy list after adding attribute noise
def ANList(k, data, labels, algorithm, noiseP, whom, case):
    shuffle(data)
    sampleCount = data.shape[0]
    cvCount = round(sampleCount / k)
    accuractList = []
    confusionMatrixList = []
    if whom == 'DC':#DvsC
        for i in range(k):
            cvFold = data.iloc[i*cvCount:(i+1)*cvCount,:]
            trainData = pd.concat([data.iloc[0:i*cvCount,:], data.iloc[(i+1)*cvCount:,:]])
            trainFold = addAttNoise(trainData, noiseP, case)
            numTestSample = cvFold.shape[0]
            tree = buildTree(trainFold, algorithm)
            labels = list(trainFold.columns)
            result = []
            for item in range(numTestSample):
                sample = cvFold.iloc[item, :-1]
                result.append(classify(tree, labels, sample, trainFold))
            cvFold.insert(cvFold.shape[1], 'predict', result)
            cvFold = cvFold
            accuractList.append(calcAccuracy(cvFold))
            confusionMatrixList.append(plotConfusionMatrix(cvFold))
        return accuractList, confusionMatrixList
    elif whom == 'CD':#CvsD
        for i in range(k):
            cvFoldData = data.iloc[i*cvCount:(i+1)*cvCount,:]
            cvFold = addAttNoise(cvFoldData, noiseP, case)
            trainFold = pd.concat([data.iloc[0:i*cvCount,:], data.iloc[(i+1)*cvCount:,:]])
            numTestSample = cvFold.shape[0]
            tree = buildTree(trainFold, algorithm)
            labels = list(trainFold.columns)
            result = []
            for item in range(numTestSample):
                sample = cvFold.iloc[item, :-1]
                result.append(classify(tree, labels, sample, trainFold))
            cvFold.insert(cvFold.shape[1], 'predict', result)
            cvFold = cvFold
            accuractList.append(calcAccuracy(cvFold))
            confusionMatrixList.append(plotConfusionMatrix(cvFold))
        return accuractList, confusionMatrixList
    elif whom == 'DD':#DvsD
        for i in range(k):
            cvFoldData = data.iloc[i * cvCount:(i + 1) * cvCount, :]
            cvFold = addAttNoise(cvFoldData, noiseP, case)
            trainData = pd.concat([data.iloc[0:i * cvCount, :], data.iloc[(i + 1) * cvCount:, :]])
            trainFold = addAttNoise(trainData, noiseP, case)
            numTestSample = cvFold.shape[0]
            tree = buildTree(trainFold, algorithm)
            labels = list(trainFold.columns)
            result = []
            for item in range(numTestSample):
                sample = cvFold.iloc[item, :-1]
                result.append(classify(tree, labels, sample, trainFold))
            cvFold.insert(cvFold.shape[1], 'predict', result)
            cvFold = cvFold
            accuractList.append(calcAccuracy(cvFold))
            confusionMatrixList.append(plotConfusionMatrix(cvFold))
        return accuractList, confusionMatrixList
    else:#CvsC
        for i in range(k):
            cvFold = data.iloc[i*cvCount:(i+1)*cvCount,:]
            trainFold = pd.concat([data.iloc[0:i*cvCount,:], data.iloc[(i+1)*cvCount:,:]])
            numTestSample = cvFold.shape[0]
            tree = buildTree(trainFold, algorithm)
            labels = list(trainFold.columns)
            result = []
            for item in range(numTestSample):
                sample = cvFold.iloc[item, :-1]
                result.append(classify(tree, labels, sample, trainFold))
            cvFold.insert(cvFold.shape[1], 'predict', result)
            cvFold = cvFold
            accuractList.append(calcAccuracy(cvFold))
            confusionMatrixList.append(plotConfusionMatrix(cvFold))
        return accuractList, confusionMatrixList

#return the average and variance accuracy list after adding class noise
def CNList(k, data, labels, function, noiseP, source):
    shuffle(data)
    sampleCount = data.shape[0]
    cvCount = round(sampleCount / k)
    accuractList = []
    confusionMatrixList = []
    noiseData = addClassNoise(data, noiseP, source)
    for i in range(k):
        cvFold = noiseData.iloc[i*cvCount:(i+1)*cvCount,:]
        trainFold = pd.concat([data.iloc[0:i*cvCount,:], data.iloc[(i+1)*cvCount:,:]])
        numTestSample = cvFold.shape[0]
        tree = buildTree(trainFold, algorithm)
        labels = list(trainFold.columns)
        result = []
        for item in range(numTestSample):
            sample = cvFold.iloc[item, :-1]
            result.append(classify(tree, labels, sample, trainFold))
        cvFold.insert(cvFold.shape[1], 'predict', result)
        cvFold = cvFold
        accuractList.append(calcAccuracy(cvFold))
        confusionMatrixList.append(plotConfusionMatrix(cvFold))
    return accuractList, confusionMatrixList

#return the average and variance accuracy list after adding attribute noise based on 10-times-10-fold cv
def tentenANList(k, data, labels, algorithm, noiseP, whom, case):
    accuracyList = []
    confusionMatrixList = []
    for i in range(10):
        accuracy, confusionMatrix = ANList(k, data, labels, algorithm, noiseP, whom, case)
        accuracyList.append(accuracy)
        confusionMatrixList.append(confusionMatrix)
    return accuracyList, confusionMatrixList

#return the average and variance accuracy list after adding class noise based on 10-times-10-fold cv
def tentenCNList(k, data, labels, function, noiseP, whom):
    accuracyList = []
    confusionMatrixList = []
    for i in range(10):
        accuracy, confusionMatrix = CNList(k, data, labels, function, noiseP, whom)
        accuracyList.append(accuracy)
        confusionMatrixList.append(confusionMatrix)
    return accuracyList, confusionMatrixList

gameHeaderNames = ['TL', 'TM', 'TR', 'ML', 'MM', 'MR', 'LL', 'LM', 'LR', 'Class']

#Attribute noise
#CvsC
print('Dataset1 after adding attribute noise, will cost around 70mins totally')
print('Dataset1 CvsC begin, which will take a long-long time.')
accuracyCC10, CMCC10 = tentenANList(10, dataset1, gameHeaderNames, calcIG, 0, 'CC', 'dis')
accuracyCC15, CMCC15 = tentenANList(10, dataset1, gameHeaderNames, calcIG, 0.05, 'CC', 'dis')
accuracyCC110, CMCC110 = tentenANList(10, dataset1, gameHeaderNames, calcIG, 0.1, 'CC', 'dis')
accuracyCC115, CMCC115 = tentenANList(10, dataset1, gameHeaderNames, calcIG, 0.15, 'CC', 'dis')

print('Dataset1 DvsC begin, which will take a long-long time.')
accuracyDC10, CMDC10 = tentenANList(10, dataset1, gameHeaderNames, calcIG, 0, 'DC', 'dis')
accuracyDC15, CMDC15 = tentenANList(10, dataset1, gameHeaderNames, calcIG, 0.05, 'DC', 'dis')
accuracyDC110, CMDC110 = tentenANList(10, dataset1, gameHeaderNames, calcIG, 0.1, 'DC', 'dis')
accuracyDC115, CMDC115 = tentenANList(10, dataset1, gameHeaderNames, calcIG, 0.15, 'DC', 'dis')

print('Dataset1 CvsD begin, which will take a long-long time.')
accuracyCD10, CMCD10 = tentenANList(10, dataset1, gameHeaderNames, calcIG, 0, 'CD', 'dis')
accuracyCD15, CMCD15 = tentenANList(10, dataset1, gameHeaderNames, calcIG, 0.05, 'CD', 'dis')
accuracyCD110, CMCD110 = tentenANList(10, dataset1, gameHeaderNames, calcIG, 0.1, 'CD', 'dis')
accuracyCD115, CMCD115 = tentenANList(10, dataset1, gameHeaderNames, calcIG, 0.15, 'CD', 'dis')

print('Dataset1 DvsD begin, which will take a long-long time.')
accuracyDD10, CMDD10 = tentenANList(10, dataset1, gameHeaderNames, calcIG, 0, 'DD', 'dis')
accuracyDD15, CMDD15 = tentenANList(10, dataset1, gameHeaderNames, calcIG, 0.05, 'DD', 'dis')
accuracyDD110, CMDD110 = tentenANList(10, dataset1, gameHeaderNames, calcIG, 0.1, 'DD', 'dis')
accuracyDD115, CMDD115 = tentenANList(10, dataset1, gameHeaderNames, calcIG, 0.15, 'DD', 'dis')

x = [0, 5, 10, 15]

#Output the plot
f5, ax5 = plt.subplots()
plt.plot(x, [calcMean(accuracyCC10), calcMean(accuracyCC15), calcMean(accuracyCC110), calcMean(accuracyCC115)], marker='*', label='CvsC',color = 'darkred')
plt.plot(x, [calcMean(accuracyDC10), calcMean(accuracyDC15), calcMean(accuracyDC110), calcMean(accuracyDC115)], marker='*', label='DvsC', color = 'black')
plt.plot(x, [calcMean(accuracyCD10), calcMean(accuracyCD15), calcMean(accuracyCD110), calcMean(accuracyCD115)], marker='*', label='CvsD',color = 'green')
plt.plot(x, [calcMean(accuracyDD10), calcMean(accuracyDD15), calcMean(accuracyDD110), calcMean(accuracyDD115)], marker='*', label='DvsD', color = 'brown')
plt.xlabel('Noise Level')
plt.ylabel('Accuracy')
plt.title('Classification Accuracy1 with Attribute Noise',color = 'r')
plt.grid(linestyle='-.', c='y')
plt.legend()
plt.savefig('AN1')
plt.show()

print('Dataset2 CvsC begin, which will take a long-long time.')
accuracyCC20, CM210 = tentenANList(10, wineDis, dataset2.columns, calcIG, 0, 'neither', 'dis')
accuracyCC25, CM215 = tentenANList(10, wineDis, dataset2.columns, calcIG, 0.05, 'neither', 'dis')
accuracyCC210, CM2110 = tentenANList(10, wineDis, dataset2.columns, calcIG, 0.1, 'neither', 'dis')
accuracyCC215, CM2115 = tentenANList(10, wineDis, dataset2.columns, calcIG, 0.15, 'neither', 'dis')

print('Dataset2 DvsC begin, which will take a long-long time.')
accuracyDC20, CM220 = tentenANList(10, wineDis, dataset2.columns, calcIG, 0, 'train', 'dis')
accuracyDC25, CM225 = tentenANList(10, wineDis, dataset2.columns, calcIG, 0.05, 'train', 'dis')
accuracyDC210, CM2210 = tentenANList(10, wineDis, dataset2.columns, calcIG, 0.1, 'train', 'dis')
accuracyDC215, CM2215 = tentenANList(10, wineDis, dataset2.columns, calcIG, 0.15, 'train', 'dis')

print('Dataset2 CvsD begin, which will take a long-long time.')
accuracyCD20, CM230 = tentenANList(10, wineDis, dataset2.columns, calcIG, 0, 'test', 'dis')
accuracyCD25, CM235 = tentenANList(10, wineDis, dataset2.columns, calcIG, 0.05, 'test', 'dis')
accuracyCD210, CM2310 = tentenANList(10, wineDis, dataset2.columns, calcIG, 0.1, 'test', 'dis')
accuracyCD215, CM2315 = tentenANList(10, wineDis, dataset2.columns, calcIG, 0.15, 'test', 'dis')

print('Dataset2 DvsD begin, which will take a long-long time.')
accuracyDD20, CM240 = tentenANList(10, wineDis, dataset2.columns, calcIG, 0, 'both', 'dis')
accuracyDD25, CM245 = tentenANList(10, wineDis, dataset2.columns, calcIG, 0.05, 'both', 'dis')
accuracyDD210, CM2410 = tentenANList(10, wineDis, dataset2.columns, calcIG, 0.1, 'both', 'dis')
accuracyDD215, CM2415 = tentenANList(10, wineDis, dataset2.columns, calcIG, 0.15, 'both', 'dis')

#Output the plot
f6, ax6 = plt.subplots()
plt.plot(x, [calcMean(accuracyCC20), calcMean(accuracyCC25), calcMean(accuracyCC210), calcMean(accuracyCC215)], marker='p', label='CvsC', color = 'green')
plt.plot(x, [calcMean(accuracyDC20), calcMean(accuracyDC25), calcMean(accuracyDC210), calcMean(accuracyDC215)], marker='p', label='DvsC', color = 'red')
plt.plot(x, [calcMean(accuracyCD20), calcMean(accuracyCD25), calcMean(accuracyCD210), calcMean(accuracyCD215)], marker='p', label='CvsD', color = 'blue')
plt.plot(x, [calcMean(accuracyDD20), calcMean(accuracyDD25), calcMean(accuracyDD210), calcMean(accuracyDD215)], marker='p', label='DvsD', color = 'cyan')
plt.xlabel('Noise Level')
plt.ylabel('Accuracy')
plt.grid(linestyle='-.', c='y')
plt.title('Classification Accuracy2 with Attribute Noise', color='r')
plt.legend()
plt.savefig('AN2')
plt.show()

#Class noise
print('Dataset1 contradictory begin, which will take a long-long time.')
accuracycon10, CM1con0 = tentenCNList(10, dataset1, gameHeaderNames, calcIG, 0, 'con')
accuracycon15, CM1con5 = tentenCNList(10, dataset1, gameHeaderNames, calcIG, 0.05, 'con')
accuracycon110, CM1con10 = tentenCNList(10, dataset1, gameHeaderNames, calcIG, 0.1, 'con')
accuracycon115, CM1con15 = tentenCNList(10, dataset1, gameHeaderNames, calcIG, 0.15, 'con')

print('Dataset1 misclassifications begin, which will take a long-long time.')
accuracymis10, CM1mis0 = tentenCNList(10, dataset1, gameHeaderNames, calcIG, 0, 'mis')
accuracymis15, CM1mis5 = tentenCNList(10, dataset1, gameHeaderNames, calcIG, 0.05, 'mis')
accuracymis110, CM1mis10 = tentenCNList(10, dataset1, gameHeaderNames, calcIG, 0.1, 'mis')
accuracymis115, CM1mis15 = tentenCNList(10, dataset1, gameHeaderNames, calcIG, 0.15, 'mis')

#Output the plot
f7, ax7 = plt.subplots()
plt.plot(x, [calcMean(accuracycon10), calcMean(accuracycon15), calcMean(accuracycon110), calcMean(accuracycon115)], marker='*', label='Contradictory examples noise', color = 'blue')
plt.plot(x, [calcMean(accuracymis10), calcMean(accuracymis15), calcMean(accuracymis110), calcMean(accuracymis115)], marker='*', label='Misclassifications noise', color = 'yellow')
plt.xlabel('Noise Level')
plt.ylabel('Accuracy')
plt.title('Classification Accuracy1 with Class Noise', color = 'green')
plt.grid(linestyle='-.', c='g')
plt.legend()
plt.savefig('CN1')
plt.show()

print('Dataset2 contradictory begin, which will take a long-long time.')
accuracycon20, CM2con0 = tentenCNList(10, wineDis, dataset2.columns, calcIG, 0, 'con')
accuracycon25, CM2con5 = tentenCNList(10, wineDis, dataset2.columns, calcIG, 0.05, 'con')
accuracycon210, CM2con10 = tentenCNList(10, wineDis, dataset2.columns, calcIG, 0.1, 'con')
accuracycon215, CM2con15 = tentenCNList(10, wineDis, dataset2.columns, calcIG, 0.15, 'con')

print('Dataset2 misclassifications begin, which will take a long-long time.')
accuracymis20, CM2mis0 = tentenCNList(10, wineDis, dataset2.columns, calcIG, 0, 'mis')
accuracymis25, CM2mis5 = tentenCNList(10, wineDis, dataset2.columns, calcIG, 0.05, 'mis')
accuracymis210, CM2mis10 = tentenCNList(10, wineDis, dataset2.columns, calcIG, 0.1, 'mis')
accuracymis215, CM2mis15 = tentenCNList(10, wineDis, dataset2.columns, calcIG, 0.15, 'mis')

#Output the plot
f8, ax8 = plt.subplots()
plt.plot(x, [calcMean(accuracycon20), calcMean(accuracycon25), calcMean(accuracycon210), calcMean(accuracycon215)], marker='o', label='Contradictory Examples Noise', color = 'cyan')
plt.plot(x, [calcMean(accuracymis20), calcMean(accuracymis25), calcMean(accuracymis210), calcMean(accuracymis215)], marker='o', label='Misclassifications Noise', color = 'black')
plt.xlabel('Noise Level')
plt.ylabel('Accuracy')
plt.title('Classification Accuracy2 with Class Noise', color = 'g')
plt.grid(linestyle='-.', c='y')
plt.legend()
plt.savefig('CN2')
plt.show()

