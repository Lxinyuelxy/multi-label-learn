import numpy as np
import copy
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer

class CalibratedLabelRanking:
    """CalibratedLabelRanking
    Reference Paper:
        Min-Ling Zhang and Zhi-Hua Zhou. A Review on Multi-Label Learning Algorithms
        Johannes Fürnkranz. Multilabel classification via calibrated label ranking
    """
    def __init__(self, classifier=SVC(kernel='linear')):
        self.classifier = classifier

    def fit(self, X, y):
        self.label_count = y.shape[1]
        self.virtualLabel = y.shape[1]
        self.m = X.shape[0]
        self.X = X
        self.y = y
        self.realLabelClassifiers = []
        self.virtualLabelClassifiers = []
        self.trainRealLabelClassifiers()
        self.trainVirtualLabelClassifiers()
        return self

    def trainRealLabelClassifiers(self):
        for i in range(self.label_count):
            for j in range(i+1, self.label_count):
                X_subset = []
                y_subset = []
                classifier = copy.deepcopy(self.classifier)
                for k in range(self.m):
                    if self.y[k, i] == self.y[k, j]:
                        continue
                    elif self.y[k, i] == 1:
                        X_subset.append(self.X[k, :])
                        y_subset.append(i)
                    else:
                        X_subset.append(self.X[k, :])
                        y_subset.append(j)
                try:
                    classifier.fit(np.vstack(X_subset), np.array(y_subset))
                except ValueError:
                    if y_subset[0] == i:
                        classifier = i
                    else:
                        classifier = j
                self.realLabelClassifiers.append(classifier)

    def trainVirtualLabelClassifiers(self):
        for i in range(self.label_count):
            classifier = copy.deepcopy(self.classifier)
            y_temp= self.y[:, i]
            y_subset = []
            for j in range(self.m):
                y_subset.append(i if y_temp[j] == 1 else self.virtualLabel)
            classifier.fit(self.X, y_subset)
            self.virtualLabelClassifiers.append(classifier)

    def predict(self, X_pre):
        result = np.zeros((X_pre.shape[0], self.label_count))
        threshold = self.voteForVirtualLabel(X_pre)
        votes = self.voteForRealLabel(X_pre)
        for i in range(X_pre.shape[0]):
            result[i, votes[i,:] > threshold[i]] = 1
        return result

    def voteForVirtualLabel(self, X_pre):
        votes = np.zeros(X_pre.shape[0])
        for i in range(len(self.virtualLabelClassifiers)):
            predict = self.virtualLabelClassifiers[i].predict(X_pre)
            for j in range(X_pre.shape[0]):
                if predict[j] == self.virtualLabel:
                    votes[j] += 1
        return votes

    def voteForRealLabel(self, X_pre):
        votes = np.zeros((X_pre.shape[0], self.label_count))
        for i in range(len(self.realLabelClassifiers)):
            if isinstance(self.realLabelClassifiers[i], int):
                predict = np.array([self.realLabelClassifiers[i]] * X_pre.shape[0])
            else:
                predict = self.realLabelClassifiers[i].predict(X_pre)
            for j in range(X_pre.shape[0]):
                votes[j, predict[j]] += 1
        for j in range(len(self.virtualLabelClassifiers)):
            predict = self.virtualLabelClassifiers[j].predict(X_pre)
            for j in range(X_pre.shape[0]):
                if predict[j] < self.virtualLabel:
                    votes[j, predict[j]] += 1
        return votes
            