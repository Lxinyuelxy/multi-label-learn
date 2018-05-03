import copy
import numpy as np
from sklearn.svm import SVC

class ClassifierChain:
    """ClassifierChain
    Reference Paper:
        Min-Ling Zhang and Zhi-Hua Zhou. A Review on Multi-Label Learning Algorithms
    """
    def __init__(self, classifier=SVC(kernel='linear')):
        self.classifier = classifier

    def fit(self, X, y, order=None):
        """Fit classifier with training data

        Parameters
        ----------
        order: list
               The order of the labels to join training
        """
        self.m = X.shape[0]
        self.label_count = y.shape[1]
        if order is None:
            self.order = list(range(y.shape[1]))
        elif sorted(order) != list(range(y.shape[1])):
            raise ValueError('invalid order of labels')
        else:
            self.order = order
        self.classifiers = []
        X_extended = copy.deepcopy(X)
        for i in range(self.label_count):
            classifier = copy.deepcopy(self.classifier)
            y_subset = y[:, self.order[i]]
            classifier.fit(X_extended, y_subset)
            self.classifiers.append(classifier)
            X_extended = np.hstack([X_extended, np.vstack(y_subset)])
        return self

    def predict(self, X_pre):
        res = np.zeros((X_pre.shape[0], self.label_count))
        for i in range(self.label_count):
            y_subset = self.classifiers[i].predict(X_pre)
            X_pre = np.hstack([X_pre, np.vstack(y_subset)])
            res[:, self.order[i]] = y_subset
        return res
