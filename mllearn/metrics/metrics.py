import numpy as np

def _check_targets(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError('the shape of y_true is inconsistent with the shape of y_pred')

def _intersection(a, b):
    """
    a.shape = (m, )
    b.shape = (m, )
    """
    count = 0
    for i_a, i_b in zip(a, b):
        if i_a == 1 and i_b == 1:
            count += 1
    return count

def _convergence(a, b):
    count = 0
    for i_a, i_b in zip(a, b):
        if i_a == 1 or i_b == 1:
            count += 1
    return count

def _sym_difference(a, b):
    count = 0
    for i_a, i_b in zip(a, b):
        if i_a != i_b:
            count += 1
    return count

def subset_acc(y_true, y_pred):
    _check_targets(y_true, y_pred)
    sample_num, label_count = y_true.shape
    acc_count = 0
    for i in range(sample_num):
        if (y_true[i] == y_pred[i]).sum() == label_count:
            acc_count += 1
    return acc_count / sample_num

def hamming_loss(y_true, y_pred):
    _check_targets(y_true, y_pred)
    sample_num, label_count = y_true.shape
    count = 0
    for i in range(sample_num):
        count += _sym_difference(y_true[i], y_pred[i])
    return count / (sample_num * label_count)

def accuracy(y_true, y_pred):
    _check_targets(y_true, y_pred)
    sample_num, label_count = y_true.shape
    count = 0.
    for i in range(sample_num):
        count += _intersection(y_true[i], y_pred[i]) / _convergence(y_true[i], y_pred[i])
    return count / sample_num

def precision(y_true, y_pred):
    _check_targets(y_true, y_pred)
    sample_num, label_count = y_true.shape
    count = 0.
    for i in range(sample_num):
        if y_pred[i].sum() == 0:
            pass
        else:
            count += _intersection(y_true[i], y_pred[i]) / y_pred[i].sum()
    return count / sample_num

def recall(y_true, y_pred):
    _check_targets(y_true, y_pred)
    sample_num, label_count = y_true.shape
    count = 0.
    for i in range(sample_num):
        if y_true[i].sum() == 0:
            pass
        else:
            count += _intersection(y_true[i], y_pred[i]) / y_true[i].sum()
    return count / sample_num

def F_beta(y_true, y_pred, beta=1):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2) * p * r / (beta**2 * p + r)

if __name__ == '__main__':
    from sklearn.datasets import make_multilabel_classification
    from sklearn.model_selection import train_test_split
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC

    X, y = make_multilabel_classification(n_samples=700,
                                      n_features = 80,
                                      n_classes=5, 
                                      n_labels=2,
                                      allow_unlabeled=False,
                                      random_state=1)                                  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(X_train, y_train)
    predictions = classif.predict(X_test)
    
    print('The subset_acc Result is %f' % subset_acc(y_test, predictions))
    print('The hamming_loss Result is %f' % hamming_loss(y_test, predictions))
    print('The accuracy Result is %f' % accuracy(y_test, predictions))
    print('The precision Result is %f' % precision(y_test, predictions))
    print('The recall Result is %f' % recall(y_test, predictions))
    print('The F_beta Result is %f' % F_beta(y_test, predictions))