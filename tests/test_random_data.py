import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import datasets

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from problem_transform.clr import CalibratedLabelRanking
from problem_transform.cc import ClassifierChain
from problem_transform.klabelsets import RandomKLabelsets
from alg_adapt.mlknn import MLKNN
from alg_adapt.mldt import MLDecisionTree

X, y = make_multilabel_classification(n_samples=700,
                                      n_features = 80,
                                      n_classes=5, 
                                      n_labels=2,
                                      allow_unlabeled=False,
                                      random_state=1)                                  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

classif = ClassifierChain()
classif.fit(X_train, y_train)
predictions = classif.predict(X_test)
print('The Classifier Chain Result is %f' % hamming_loss(y_test, predictions))

classif = CalibratedLabelRanking()
classif.fit(X_train, y_train)
predictions = classif.predict(X_test)
print('The CalibratedLabelRanking Result is %f' % hamming_loss(y_test, predictions))

classif = RandomKLabelsets()
classif.fit(X_train, y_train)
predictions = classif.predict(X_test)
print('The Random K-Labelsets Result is %f' % hamming_loss(y_test, predictions))

classif = MLKNN()
classif.fit(X_train, y_train)
predictions = classif.predict(X_test)
print('The MLKNN is %f' % hamming_loss(y_test, predictions))

# classif = MLDecisionTree()
# classif.fit(X_train, y_train)
# predictions = classif.predict(X_test)
# print('The MLDecisionTree Result is %f' % hamming_loss(y_test, predictions))

classif = OneVsRestClassifier(SVC(kernel='linear'))
classif.fit(X_train, y_train)
predictions = classif.predict(X_test)
print('The Reference Result is %f' % hamming_loss(y_test, predictions))