import copy
import random
import numpy as np
from sklearn.svm import SVC

class RandomKLabelsets:
    """RandomKLabelsets
    Reference Paper:
        Min-Ling Zhang and Zhi-Hua Zhou. A Review on Multi-Label Learning Algorithms
    """
    def __init__(self, classifier=SVC(kernel='linear')):
        self.classifier = classifier

    def fit(self, X, y, k=3, n=0):
        self.m = X.shape[0]
        self.label_count = y.shape[1]
        self.k = self.chooseLabelsetsSize(k)
        self.n = self.chooseLabelsetsNum(n)
        self.k_labelsets = np.zeros((self.n, self.label_count))
        self.classifiers = []
        for i in range(self.n):
            classifier = copy.deepcopy(self.classifier)
            k_labelset = self.generateRandomK_labelsets()
            y_subset = self.getSubsetOfy(y, k_labelset)
            classifier.fit(X, self.transform(y_subset))
            self.classifiers.append(classifier)
            self.k_labelsets[i, :] = k_labelset
        return self

    def predict(self, X_pre):
        result = np.zeros((X_pre.shape[0], self.label_count))
        ysubsets = []
        for i in range(self.n):
            ysubsets.append(self.inverse_transform(self.classifiers[i].predict(X_pre)))
        for sample in range(X_pre.shape[0]):
            for label in range(self.label_count):
                maxVotes = 0
                actualVotes = 0
                for i in range(self.n):
                    if ysubsets[i][sample, label] == 1:
                        actualVotes += 1
                    if self.k_labelsets[i, label] == 1:
                        maxVotes += 1
                if (actualVotes/maxVotes) > 0.5:
                    result[sample][label] = 1
        return result

    def chooseLabelsetsSize(self, k):
        if k > self.label_count:
            raise ValueError('the given size of labelsets is exceed') 
        else:
            return k

    def chooseLabelsetsNum(self, n):
        if n == 0:
            n = 2*self.label_count
        mostLabelsetsNum = 1
        for i in range(self.k):
            mostLabelsetsNum = mostLabelsetsNum * (self.label_count-i) / (self.k-i)
        return min(n, mostLabelsetsNum)

    def generateRandomK_labelsets(self):
        labelIndexes = set()
        labelset = np.zeros(self.label_count)
        while len(labelIndexes) < self.k:
            randomIndex = random.randint(0,self.label_count-1)
            labelIndexes.add(randomIndex)
            labelset[randomIndex] = 1   
        return labelset

    def getSubsetOfy(self, y, k_labelset):
        y_subset = np.zeros((self.m, self.label_count))
        for sample in range(self.m):
            for index in range(self.label_count):
                if y[sample, index]==1 and k_labelset[index]==1:
                    y_subset[sample, index] = 1             
        return y_subset

    def transform(self, y_subset):
        result = np.zeros(y_subset.shape[0])
        for i in range(y_subset.shape[0]):
            for j in range(y_subset.shape[1]):
                result[i] += y_subset[i][j] * (2**j)
        return result

    def inverse_transform(self, y):
        result = np.zeros((y.shape[0], self.label_count))
        for row in range(result.shape[0]):
            number = y[row]
            for col in range(result.shape[1]):
                result[row][col] = number % 2
                number = int(number/2)
        return result
