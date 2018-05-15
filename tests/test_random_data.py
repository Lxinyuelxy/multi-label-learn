from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from mllearn.problem_transform import BinaryRelevance
from mllearn.metrics import subset_acc
from mllearn.metrics import hamming_loss as hamming_loss
from mllearn.metrics import accuracy
from mllearn.metrics import precision
from mllearn.metrics import recall
from mllearn.metrics import F_beta

X, y = make_multilabel_classification(n_samples=700,
                                      n_features = 80,
                                      n_classes=5, 
                                      n_labels=2,
                                      allow_unlabeled=False,
                                      random_state=1)                                  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

classif = BinaryRelevance()
classif.fit(X_train, y_train)
predictions = classif.predict(X_test)
print('The subset_acc Result is %f' % subset_acc(y_test, predictions))
print('The hamming_loss Result is %f' % hamming_loss(y_test, predictions))
print('The accuracy Result is %f' % accuracy(y_test, predictions))
print('The precision Result is %f' % precision(y_test, predictions))
print('The recall Result is %f' % recall(y_test, predictions))
print('The F_beta Result is %f' % F_beta(y_test, predictions))
