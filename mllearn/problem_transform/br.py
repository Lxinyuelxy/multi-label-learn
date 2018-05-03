import copy
import numpy as np
from sklearn.svm import SVC

class BinaryRelevance:
    def __init__(self, classifier=SVC(kernel='linear')):
        self.classifier = classifier

    def fit(self, X, y):
        self.label_count = y.shape[1]
        self.classifiers = []
        for i in range(self.label_count):
            classifier = copy.deepcopy(self.classifier)
            y_subset = y[:, i]
            classifier.fit(X, y_subset)
            self.classifiers.append(classifier)
        return self

    def predict(self, X_pre):
        result = np.zeros((X_pre.shape[0], self.label_count))
        for i in range(self.label_count):
            result[:, i] = self.classifiers[i].predict(X_pre)
        return result
