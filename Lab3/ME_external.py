from collections import defaultdict
import numpy as np


class maxEntropy(object):
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
    def train(self, max_iter=1000):
        self.initP()  # 主要计算M以及联合分布在f上的期望
        # 下面计算条件分布及其期望，正式开始训练
        for i in range(max_iter):  # 计算条件分布在特诊函数上的期望
            self.ep = self.EP()
            self.lastw = self.w[:]
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
        for i in range(len(self.w)):
            if abs(self.w[i] - self.lastw[i]) >= 0.001:
                return False
        return True

    def predict(self, input):
        features = input.strip().split()
        prob = self.calPyx(features)
        prob.sort(reverse=True)

        return prob







    def process(self,line,is_train):
        fields = line.strip().split()

        if len(fields)==0:
            return ''

        word = fields[0]

        if len(fields)==1 and not is_train:
            return str(len(word))
        elif len(fields)==2:
            label=fields[1]
            return label + ' ' + str(len(word))
        else:
            assert False


    def get_features(self,trainfile_path, featuresfile_path,is_train=True):
        """
        get feature.txt with trainfile
        :param trainfile_path:
        :param featuresfile_path:
        :return:
        """
        trainfile = open(trainfile_path, "r+", encoding="GB18030")
        featuresfile = open(featuresfile_path, "w", encoding="GB18030")
        while True:
            line = trainfile.readline()
            if line == '':
                break

            features = self.process(line,is_train)
            featuresfile.write(features + '\n')
        trainfile.close()
        featuresfile.close()

    def train_ME(self,featuresfile_path, ME_model_path):
        """
        get ME_model.txt with feature.txt
        :param featuresfile_path:
        :param ME_model_path:
        :return:
        """

        # TODO
        self.loadData(featuresfile_path)
        self.train()
        ME_model=open(ME_model_path,"w",encoding="GB18030")
        # for label in self.labels:
        #     ME_model.write(label+' ')
        # ME_model.write('\n')
        for label,f in self.features:
            id=self.features[(label,f)]
            fw=self.w[id]
            ME_model.write("%s %s %f\n"%(label,f,fw))
        ME_model.close()

    def getPredict(self,testfile_path, ME_model_path, test_feature_path, result_path):
        """
        get test_feature.txt result.txt with model.txt and testfile
        :param testfile_path:
        :param ME_model_path:
        :param test_feature_path:
        :param result_path:
        :return:
        """

        self.get_features(testfile_path,test_feature_path,is_train=False)


        ME_model_file = open(ME_model_path, "r", encoding="GB18030")
        self.features = defaultdict(int)  # 用于获得(标签，特征)键值对
        self.labels = set([])  # 标签
        self.w = []
        id=0
        while True:
            line = ME_model_file.readline()
            if line == '':
                break
            fields = line.strip().split()
            assert len(fields)==3
            label = fields[0]

            feature=fields[1]
            w=fields[2]
            self.features[(label,feature)]=id
            self.w.append(w)
            self.labels.add(label)
            id+=1
        ME_model_file.close()
        test_feature_file = open(test_feature_path, "r+", encoding="GB18030")
        testfile=open(testfile_path, "r+", encoding="GB18030")
        result_file = open(result_path, "w", encoding="GB18030")
        while True:
            line = test_feature_file.readline()
            line2=testfile.readline()
            if line == '':
                break
            if line=='\n':
                continue
            prop=mxEnt.predict(line)
            result_file.write(line2[:-1]+' '+prop[0][1]+'\n')
        test_feature_file.close()
        result_file.close()

    def evaluate(self,standard_answer_path, result_path, evaluation_result_path):
        """
        get evaluation_result.txt with standard_answer and result.txt by SharedTaskEval.pl
        :param standard_answer_path:
        :param result_path:
        :param evaluation_result_path:
        :return:
        """
        standard_answer_file=open(standard_answer_path,"r+",encoding="GB18030")
        result_file=open(result_path,"r+",encoding="GB18030")
        sum=0
        TP=0
        while True:
            line = standard_answer_file.readline()
            line2=result_file.readline()
            if line == '':
                break
            while line=='\n':
                line = standard_answer_file.readline()
            fields1 = line.strip().split()
            fields2 = line2.strip().split()
            assert fields1[0]==fields2[0]
            label1=fields1[1]
            label2=fields2[1]
            if label1==label2 and label1=='O':
                continue
            print(fields1[0],label1,label2)
            sum+=1
            if label1==label2:
                TP+=1
        print(TP,sum,TP/sum)
        standard_answer_file.close()
        result_file.close()

if __name__ == '__main__':
    # mxEnt = maxEntropy()
    # mxEnt.loadData('data/gameLocation.dat')
    # mxEnt.train()
    # print(mxEnt.predict('Sunny Cloudy\n'),mxEnt.predict('Sunny Cloudy\n')[0][1])

    trainfile_path="data/trainfile_path.txt"
    featuresfile_path="data/featuresfile_path.txt"
    ME_model_path="data/ME_model_path.txt"
    testfile_path="data/testfile_path.txt"
    test_feature_path="data/test_feature_path.txt"
    result_path="data/result_path.txt"
    standard_answer_path="data/standard_answer_path.txt"
    mxEnt=maxEntropy()
    # mxEnt.get_features(trainfile_path,featuresfile_path)
    # mxEnt.train_ME(featuresfile_path,ME_model_path)
    mxEnt.getPredict(testfile_path,ME_model_path, test_feature_path, result_path)
    mxEnt.evaluate(standard_answer_path, result_path, "")