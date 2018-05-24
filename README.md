# multi-label-learn

mlleran is a python library for multi-label classification bulti on scikit-learn and numpy.

## Implementation
The implementation is based on the paper [A Review on Multi-Label Learning Algorithms](https://ieeexplore.ieee.org/document/6471714/), and the implementated algorithms include:

- [x] Binary Relevance
- [x] Classifier Chains
- [x] Calibrated Label Ranking
- [x] Random k-Labelsets
- [x] Multi-Label k-Nearest Neighbor
- [x] Ranking Support Vector Machine
- [ ] Multi-Label Decision Tree
- [ ] Collective Multi-Label Classifier

## Installation
```bash
pip install mllearn
```

## Data Format
All data type shoud be `ndarray`, especially y should be binary format.For example, if your dataset totally have 5 labels and one of your sample has only first and last labels, then the corresponding output should be `[1, 0, 0, 0, 1]`.
```python
samples, features = X_train.shape
samples, labels = y_train.shape
samples_test, features = X_test.shape
samples_test, labels = y_test.shape
```
## Example Usage
This library includes 2 parts, algorithms and metrics.
```python
from mllearn.problem_transform import BinaryRelevance

classif = BinaryRelevance()
classif.fit(X_train, y_train)
predictions = classif.predict(X_test)
```

```python
from mllearn.metrics import subset_acc
acc = subset_acc(y_test, predictions)
```
