from collections import defaultdict

import numpy as np


# ! 第一部分是核心ME算法，主要是模型的训练和预测模块，不直接作为函数调用。
class MaxEntropy(object):
    def __init__(self):
        self.trainset = []  # 训练数据集
        self.features = defaultdict(int)  # 用于获得(标签，特征)键值对
        self.labels = set([])  # 标签
        self.w = []

    def loadData(self, fName):
        for line in open(fName):
            fields = line.strip().split()
            # at least two columns
            if len(fields) < 2: continue  # 只有标签没用
            # the first column is label
            label = fields[0]
            self.labels.add(label)  # 获取label
            for f in set(fields[1:]):  # 对于每一个特征
                # (label,f) tuple is feature
                self.features[(label, f)] += 1  # 每提取一个（标签，特征）对，就自加1，统计该特征-标签对出现了多少次
            self.trainset.append(fields)
            self.w = [0.0] * len(self.features)  # 初始化权重
            self.lastw = self.w

    # 对于该问题，M是一个定值，所以delta有解析解
    def train(self, max_iter=500):
        # time很小
        self.initP()  # 主要计算M以及联合分布在f上的期望
        # 下面计算条件分布及其期望，正式开始训练
        for i in range(max_iter):  # 计算条件分布在特诊函数上的期望
            print(i, max_iter,end='')
            # 用时很大，主要部分
            self.ep = self.EP()
            # time很小
            self.lastw = self.w[:]
            # time不大，占大约不到1/20
            for i, w in enumerate(self.w):
                self.w[i] += (1.0 / self.M) * np.log(self.Ep_[i] / self.ep[i])
            if self.convergence():
                break

    def initP(self):
        # 获得M
        self.M = max([len(feature[1:]) for feature in self.trainset])
        self.size = len(self.trainset)
        self.Ep_ = [0.0] * len(self.features)
        # 获得联合概率期望
        for i, feat in enumerate(self.features):
            self.Ep_[i] += self.features[feat] / (1.0 * self.size)
            # 更改键值对为（label-feature）-->id
            self.features[feat] = i
        # 准备好权重
        self.w = [0.0] * len(self.features)
        self.lastw = self.w

    def EP(self):
        # 计算pyx
        ep = [0.0] * len(self.features)
        for record in self.trainset:
            features = record[1:]
            # cal pyx
            prob = self.calPyx(features)
            for f in features:  # 特征一个个来
                for pyx, label in prob:  # 获得条件概率与标签
                    if (label, f) in self.features:
                        id = self.features[(label, f)]
                        ep[id] += (1.0 / self.size) * pyx
        return ep

    # 获得最终单一样本每个特征的pyx
    def calPyx(self, features):
        # 传的feature是单个样本的
        wlpair = [(self.calSumP(features, label), label) for label in self.labels]
        Z = sum([w for w, l in wlpair])
        prob = [(w / Z, l) for w, l in wlpair]
        return prob

    def calSumP(self, features, label):
        sumP = 0.0
        # 对于这单个样本的feature来说，不存在于feature集合中的f=0所以要把存在的找出来计算
        for showedF in features:
            if (label, showedF) in self.features:
                sumP += float(self.w[self.features[(label, showedF)]])
        return np.exp(sumP)

    def convergence(self):
        sum = 0
        for i in range(len(self.w)):
            sum += abs(self.w[i] - self.lastw[i])
        print(' ' + str(sum / len(self.w))+' '+str(0.01))
        return sum < 0.01 * len(self.w)

    def predict(self, input):
        features = input.strip().split()
        prob = self.calPyx(features)
        prob.sort(reverse=True)

        return prob
