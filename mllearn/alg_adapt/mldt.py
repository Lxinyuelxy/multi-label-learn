import numpy as np
import time

class MLDecisionTree:
    """
    not implement the pruning
    """
    def __init__(self, leafSize = 10):
        self.leafSize = leafSize
   
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.m = X.shape[0]
        self.label_count = y.shape[1]
        self.selectedSplitPoints = {}

        start_time = time.time()
        self.tree = self.create_tree(np.hstack((X, y)), np.arange(X.shape[1]))
        print('the total of creat the tree: ', time.time() - start_time)

        print('self.tree: ', self.tree)

    def create_tree(self, dataset, originFeatures):
        print('============================')
        if dataset.shape[0] == 0:
            return
        # 递归返回情形1：得到的样本数量小于预先设定的阈值
        # 递归返回情形2：已经划分完所有特征
        if dataset.shape[0] <= self.leafSize or len(originFeatures) == 0:
            print('return')
            return self.getRelativeLabels(dataset)

        # 从dataset中挑选出最优划分属性及其划分点
        start_time = time.time()
        index, splitPoint  = self.chooseBestFeature(dataset)
        print('the total time of chooseBestFeature: ', time.time() - start_time)

        bestFeature = originFeatures[index]

        # 储存该属性特征的划分点       
        self.selectedSplitPoints[bestFeature] = splitPoint
        
        # 从原始的待划分属性中删除已挑选出的最优划分属性
        originFeatures = np.delete(originFeatures, index) 

        # 根据所选的最优划分属性以及其划分点将dataset划分
        start_time = time.time()
        leftdataset, rightdataset = self.splitdataset(dataset, index, splitPoint)    
        print('the total time of splitdataset: ', time.time() - start_time)

        leftdataset = np.delete(leftdataset, index, 1)
        rightdataset = np.delete(rightdataset, index, 1)
        # 递归继续对两个子集进行划分
        myTree =  {bestFeature:{}}
        myTree[bestFeature][0] = self.create_tree(leftdataset, originFeatures[:])
        myTree[bestFeature][1] = self.create_tree(rightdataset, originFeatures[:])
        return myTree

    # 根据所给的各属性的划分点(splitPointsSet)找出最优划分属性
    def chooseBestFeature(self, dataset):
        minEnts = float('inf')
        best_splitPoint = -1
        index = -1
        feature_num = dataset.shape[1] - self.label_count
        for feature in range(feature_num):
            sum_ent, splitpoint = self.selectSplitPoint(dataset, feature)
            # 用某属性feature对样本D进行划分所得的信息增益为IG = Ent(D) - sum_ent(feature, splitpoint)
            # 为使IG最大，这里简化为只计算后半部分sum_ent(feature, splitpoint),使其最小
            if sum_ent < minEnts:
                minEnts = sum_ent
                best_splitPoint = splitpoint
                index = feature 
        return index, best_splitPoint
    
    # 针对dataset中的feature属性挑选最佳划分点
    def selectSplitPoint(self, dataset, feature): 
        featureVals = dataset[:, feature]
        sortedFeatureVals = np.sort(featureVals) 
        pointsSet = []
        for i in range(len(featureVals)-1):
            pointsSet.append((sortedFeatureVals[i] + sortedFeatureVals[i+1]) / 2)
        minEnts = float('inf')
        bestSplitPoint = -1    
        for i in range(len(pointsSet)):
            sum_ent = self.cal_sum_ent(dataset, feature, pointsSet[i])
            if sum_ent < minEnts:
                minEnts = sum_ent
                bestSplitPoint = pointsSet[i]
        return minEnts, bestSplitPoint

    # 根据划分点(splitPoint), 计算用特征属性(feature)对样本集(dataset)
    # 划分所得的信息熵之和
    def cal_sum_ent(self, dataset, feature, splitPoint):
        child_data1, child_data2 = self.splitdataset(dataset, feature, splitPoint)
        size1 = child_data1.shape[0]
        size2 = child_data2.shape[0]
        sum_ent = size1/(size1+size2)*self.calcMLEnt(child_data1) + \
                  size2/(size1+size2)*self.calcMLEnt(child_data2)
        return sum_ent

    # 计算多标记数据类型的香农熵
    def calcMLEnt(self, dataset):
        if dataset.shape[0] == 0:
            return 0.0
        else:
            MLEnt = 0
            for yj in range(self.label_count):
                p_j = self.calcP_j(dataset, yj)
                if p_j == 0 or p_j == 1:
                    MLEnt += 0.0
                else:
                    MLEnt += -p_j*np.log2(p_j) - (1-p_j)*np.log2(1-p_j)
            return MLEnt
    
    # 计算在某一数据集中，含有标记j的样本所占比重
    def calcP_j(self, dataset, j):
        p_j = 0
        n = dataset.shape[0]
        for i in range(n):
            if dataset[i, -self.label_count+j] == 1:
                p_j += 1
        return p_j / n

    # 根据特征属性feature的划分点splitPoint,将数据集dataset划分为两个子集
    def splitdataset(self, dataset, feature, splitPoint):
        lrows = [row[feature] <= splitPoint for row in dataset]
        leftdataset = dataset[lrows, :]
        rrows = [row[feature] > splitPoint for row in dataset]
        rightdataset = dataset[rrows, :]
        return leftdataset, rightdataset
    
    # 根据最终划分得的dataset,给出相关标记
    def getRelativeLabels(self, dataset):
        relativeLabels = []
        for j in range(self.label_count):
            if self.calcP_j(dataset, j) > 0.5:
                relativeLabels.append(j)
        return relativeLabels

    def predict(self, X_pre):
        result = np.zeros((X_pre.shape[0], self.label_count))  
        for i in range(X_pre.shape[0]):
            labels = self.classify(self.tree, X_pre[i,:])
            result[i, labels] = 1
        return result
    
    def classify(self, inputTree, testVec):
        featureIndex = list(inputTree.keys())[0]
        secondDict = inputTree[featureIndex]
        key = testVec[featureIndex]
        if key <= self.selectedSplitPoints[featureIndex]:
            valueOfFeat = secondDict[0]
        else:
            valueOfFeat = secondDict[1]
        if isinstance(valueOfFeat, dict): 
            classLabel = self.classify(valueOfFeat, testVec)
        else: 
            classLabel = valueOfFeat
        return classLabel

        
