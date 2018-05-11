import numpy as np

def _check_targets(y_true, y_pred):
    if y_true.shape != y_pred:
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