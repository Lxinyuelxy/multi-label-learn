import numpy as np
from mllearn.problem_transform import BinaryRelevance
from mllearn.problem_transform import CalibratedLabelRanking
from mllearn.problem_transform import ClassifierChain
from mllearn.problem_transform import RandomKLabelsets
from mllearn.alg_adapt import MLKNN
from mllearn.alg_adapt import MLDecisionTree
from mllearn.metrics import hamming_loss
from mllearn.metrics import subset_acc

import arff
train_data = arff.load(open('datasets/scene-train.arff'))
train_data = np.array(train_data['data'], dtype=float)
X_train = train_data[:, :-6]
y_train = train_data[:, -6:]
test_data = arff.load(open('datasets/scene-test.arff'))
test_data = np.array(test_data['data'], dtype=float)
X_test = test_data[:, :-6]
y_test = test_data[:, -6:]

classif = BinaryRelevance()
classif.fit(X_train, y_train)
predictions = classif.predict(X_test)
print('The hamming_loss of BinaryRelevance is %f' % hamming_loss(y_test, predictions))
print('The subset_acc of BinaryRelevance is %f' % subset_acc(y_test, predictions))

classif = ClassifierChain()
classif.fit(X_train, y_train)
predictions = classif.predict(X_test)
print('The hamming_loss of Classifier Chain is %f' % hamming_loss(y_test, predictions))
print('The subset_acc of Classifier Chain is %f' % subset_acc(y_test, predictions))

classif = CalibratedLabelRanking()
classif.fit(X_train, y_train)
predictions = classif.predict(X_test)
print('The hamming_loss of CalibratedLabelRanking is %f' % hamming_loss(y_test, predictions))
print('The subset_acc of CalibratedLabelRanking is %f' % subset_acc(y_test, predictions))

classif = RandomKLabelsets()
classif.fit(X_train, y_train)
predictions = classif.predict(X_test)
print('The hamming_loss of Random K-Labelsets is %f' % hamming_loss(y_test, predictions))
print('The subset_acc of Random K-Labelsets is %f' % subset_acc(y_test, predictions))

classif = MLKNN()
classif.fit(X_train, y_train)
predictions = classif.predict(X_test)
print('The hamming_loss of MLKNN is %f' % hamming_loss(y_test, predictions))
print('The subset_acc of MLKNN is %f' % subset_acc(y_test, predictions))

# classif = MLDecisionTree()
# classif.fit(X_train, y_train)
# predictions = classif.predict(X_test)
# print('The MLDecisionTree Result is %f' % hamming_loss(y_test, predictions))