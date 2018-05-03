import copy
import numpy as np
from sklearn.neighbors import NearestNeighbors

class MLKNN:
    def __init__(self, k = 6):
        self.k = k

    def fit(self, X, y):
        self.X = X 
        self.y = y
        self.m = X.shape[0]
        self.label_count = y.shape[1]
        self.knn = NearestNeighbors(self.k)
        self.knn.fit(X)
        self.s = 1
        self.hj = self.countHj()
        self.kj = np.zeros(self.label_count)
        self.ir_kj = np.zeros(self.label_count)
        self.neighbors = self.knn.kneighbors(n_neighbors=self.k, return_distance=False)
        
        for label in range(self.label_count):
            for r in range(self.k+1):
                self.kj[label] +=  self.countKj(False, label, r)
                self.ir_kj[label] +=  self.countKj(True, label, r)
        return self

    def countHj(self):
        hj = np.sum(self.y, axis=0)
        hj = (self.s + hj) / (self.s*2 + self.m)
        hj = hj / (1 - hj)
        return hj

    def countKj(self, isContrary, label, r):
        kj = 0
        for sample in range(self.m):
            cj = np.sum(self.y[self.neighbors[sample], label])
            if isContrary == False:
                if self.y[sample, label] == 1 and cj == r:
                    kj += 1
            else:
                if self.y[sample, label] == 0 and cj == r:
                    kj += 1
        return kj

    def predict(self, X_pre):
        result = np.zeros((X_pre.shape[0], self.label_count))
        X_pre_neighbors = self.knn.kneighbors(X_pre, n_neighbors=self.k, return_distance=False)
        for sample in range(X_pre.shape[0]):       
            for label in range(self.label_count):
                cj = np.sum(self.y[X_pre_neighbors[sample], label])
                kj_cj = self.countKj(False, label, cj)
                ir_kj_cj = self.countKj(True, label, cj)
                cj_hj = (self.s + kj_cj) / (self.s*(self.k+1) + self.kj[label])
                cj_ir_hj = (self.s + ir_kj_cj) / (self.s*(self.k+1) + self.ir_kj[label])
                if self.hj[label] * cj_hj / cj_ir_hj > 1:
                    result[sample, label] = 1
        return result
