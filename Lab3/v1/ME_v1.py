import time
from collections import defaultdict
from queue import Queue
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
        sum=0
        for i in range(len(self.w)):
            sum+=abs(self.w[i] - self.lastw[i])
        print(' '+str(sum/len(self.w)))
        return sum<0.01*len(self.w)

    def predict(self, input):
        features = input.strip().split()
        prob = self.calPyx(features)
        prob.sort(reverse=True)

        return prob

    keywords = ["human", "IL-2", "promoter", "HIV-1", "kappa", "c-fos", "LTR", "NF-kappa", "c-jun", "5", "AP-1",
                "enhancer", "reporter", "cDNA", "HIV", "immunoglobulin", "DNA", "transcription", "GM-CSF", "promoters",
                "binding", "gene", "promoter", "site", "genes", ")", "(", "element", "and", "sites", "enhancer",
                "region", "binding", "elements", "sequence", "B", "repeat", "DNA", "alpha", "'", "receptor", "terminal",
                "sequences", "regulatory", "reporter", "motif", "response", ",", "II", "to", "box", "promoters",
                "constructs", "LTR", "factor", "virus", "long", "1", "motifs", ";", "kappa", "bp", "2", "construct",
                "locus", "cDNA", "class", "regions", "beta", "3", "NF-kappa", "transcription", "NF-kappaB", "IL-2",
                "nuclear", "AP-1", "IL-4", "I", "TNF-alpha", "IFN-gamma", "glucocorticoid", "cytokine", "protein",
                "cytokines", "IL-10", "human", "NFAT", "TCR", "NF-AT", "Tax", "TNF", "tumor", "PKC", "CD40", "IL-6",
                "p50", "GR", "IL-12", "CD28", "p65", "GATA-1", "NF", "IL-1", "calcineurin", "T", "interleukin",
                "GM-CSF", "STAT3", "c-Jun", "STAT", "STAT1", "PU.1", "IL-2R", "Sp1", "transcriptional", "CIITA",
                "c-Rel", "LMP1", "interleukin-2", "VCAM-1", "IL-5", "Oct-2", "DNA-binding", "NFkappaB", "ER", "STAT5",
                "ICAM-1", "IFN", "cytoplasmic", "IL-13", "IL-8", "MHC", "CREB", "Stat3", "anti-CD3", "VDR", "tyrosine",
                "Bcl-2", "E2F", "IgE", "c-Fos", "T-cell", "Oct-1", "TCF-1", "CD3", "RAR", "Fas", "cellular", "BSAP",
                "Tat", "NFATp", "IL-1beta", "IkappaBalpha", "signal", "C/EBP", "TNFalpha", "B", "factor", "protein",
                "receptor", ")", "(", "factors", "proteins", "alpha", "kinase", "kappa", "transcription", "receptors",
                "and", "domain", "complex", "family", "binding", "II", "1", "beta", "necrosis", "complexes", "cell",
                "of", ",", "molecules", "C", "antigen", "kinases", "I", "subunit", "nuclear", "2", "T", "gamma",
                "antibody", "class", "activated", "molecule", "antibodies", "A", "adhesion", "virus", "domains",
                "activator", "type", "cells", "chain", "gene", "region", "tyrosine", "colony-stimulating", "cytokines",
                "regulatory", "NF-kappa", "surface", "mAb", "factor-alpha", "acid", "product", "3", "factor-kappa",
                "cytokine", "antigens", "kappaB", "3-kinase", "membrane", "protein-1", "T", "human", "B", "monocytes",
                "peripheral", "lymphocytes", "activated", "primary", "macrophages", "normal", "PBMC", "erythroid",
                "endothelial", "mononuclear", "monocytic", "neutrophils", "myeloid", "hematopoietic", "resting",
                "cells", "T", "lymphocytes", "blood", "monocytes", "cell", "B", "and", "mononuclear", "human",
                "peripheral", ")", "(", "leukocytes", "lineage", "macrophages", "progenitors", "human", "Jurkat",
                "U937", "THP-1", "T", "HeLa", "cell", "HL-60", "K562", "cells", "cell", "lines", "line", "T", "(", ")",
                "and", "B", "human", "clones", "T-cell", "leukemia", ",", "lymphocytes", "monocytic", "U937", "Jurkat",
                "mRNA", "mRNA", "transcripts", "RNA", "(", ")"]

    def getWordShape(self, word, isAbbr=False):
        wordshape = "Shape"
        for i in word:
            if i.isupper():
                wordshape += 'X'
            elif i.islower():
                wordshape += 'x'
            elif i.isdigit():
                wordshape += 'd'
            else:
                wordshape += '-'
            if isAbbr and wordshape[-1] == wordshape[-2]:
                wordshape = wordshape[0:-1]

        return wordshape

    def process(self, line, lastlinesqueue, nextline, is_train):
        field = line.strip().split()
        nextfield = nextline.strip().split()
        lastfields = []
        for i in range(lastlinesqueue.qsize()):
            lastline = lastlinesqueue.get()
            lastfields.append(lastline.strip().split())
            lastlinesqueue.put(lastline)
        # TODO 唯一改动，设计特征模板
        # 处理空行
        if len(field) == 0:
            return ''
        # 1-gram
        word = field[0]
        s = str(len(word))
        if word in self.keywords:
            s += ' ' + "Key" + word
        # if '-' in word:
        #     s += ' ' + '-'
        # if word.isupper():
        #     s += ' ' + "upper"
        # if any(char.isdigit() for char in word):
        #     s += ' ' + "has_digit"
        # if '.' in word:
        #     s += ' ' + '.'
        # if word[-1] == 's':
        #     s += ' ' + "plurality"
        # 加入word，提升到11 80 0.1375
        s += ' ' + word
        # 经测试wordshape加入到s，或者代替keyword都会造成性能下降
        s += ' shape' + self.getWordShape(word, isAbbr=False) + ' abbr' + self.getWordShape(word, isAbbr=True)
        # 2-gram
        for lastfield in lastfields:
            if not len(lastfield) == 0:
                lastword = lastfield[0]
                if lastword in self.keywords:
                    s += " lastKey" + lastword
        # 加入后向，略微提升279 1185 0.23544303797468355
        if not len(nextfield) == 0:
            nextword = nextfield[0]
            if nextword in self.keywords:
                s += " nextKey" + nextword
        # 更换keywords：279 1185 0.23544303797468355->301 1132 0.2659010600706714
        # 在21561行训练样本下测试为413 1095 0.3771689497716895
        # 以下代码不用更改
        # 测试
        if len(field) == 1 and not is_train:
            return s

        # 训练
        if len(field) == 2:
            label = field[1]
            s = label + ' ' + s
            return s

        # 错误
        assert False

    def get_features(self, trainfile_path, featuresfile_path, is_train=True):
        """
        get feature.txt with trainfile
        :param trainfile_path:
        :param featuresfile_path:
        :return:
        """
        trainfile = open(trainfile_path, "r+", encoding="GB18030")
        featuresfile = open(featuresfile_path, "w", encoding="GB18030")
        lastlinesqueue = Queue()
        lastlinesqueue.put('')
        nextline = trainfile.readline()
        while True:
            line = nextline

            if line == '':
                break
            nextline = trainfile.readline()
            features = self.process(line, lastlinesqueue, nextline, is_train)
            lastlinesqueue.put(line)
            limit = 1
            if lastlinesqueue.qsize() > limit:
                lastlinesqueue.get()
                assert lastlinesqueue.qsize() <= limit
            featuresfile.write(features + '\n')
        trainfile.close()
        featuresfile.close()

    def train_ME(self, featuresfile_path, ME_model_path):
        """
        get ME_model.txt with feature.txt
        :param featuresfile_path:
        :param ME_model_path:
        :return:
        """

        # TODO
        self.loadData(featuresfile_path)
        self.train()
        ME_model = open(ME_model_path, "w", encoding="GB18030")
        # for label in self.labels:
        #     ME_model.write(label+' ')
        # ME_model.write('\n')
        for label, f in self.features:
            id = self.features[(label, f)]
            fw = self.w[id]
            ME_model.write("%s %s %f\n" % (label, f, fw))
        ME_model.close()

    def getPredict(self, testfile_path, ME_model_path, test_feature_path, result_path):
        """
        get test_feature.txt result.txt with model.txt and testfile
        :param testfile_path:
        :param ME_model_path:
        :param test_feature_path:
        :param result_path:
        :return:
        """

        self.get_features(testfile_path, test_feature_path, is_train=False)

        ME_model_file = open(ME_model_path, "r", encoding="GB18030")
        self.features = defaultdict(int)  # 用于获得(标签，特征)键值对
        self.labels = set([])  # 标签
        self.w = []
        id = 0
        while True:
            line = ME_model_file.readline()
            if line == '':
                break
            fields = line.strip().split()
            assert len(fields) == 3
            label = fields[0]

            feature = fields[1]
            w = fields[2]
            self.features[(label, feature)] = id
            self.w.append(w)
            self.labels.add(label)
            id += 1
        ME_model_file.close()
        test_feature_file = open(test_feature_path, "r+", encoding="GB18030")
        testfile = open(testfile_path, "r+", encoding="GB18030")
        result_file = open(result_path, "w", encoding="GB18030")
        while True:
            line = test_feature_file.readline()
            line2 = testfile.readline()
            if line == '':
                break
            if line == '\n':
                result_file.write('\n')
                continue
            prop = mxEnt.predict(line)
            result_file.write(line2[:-1] + ' ' + prop[0][1] + '\n')
        test_feature_file.close()
        result_file.close()

    def evaluate(self, standard_answer_path, result_path, evaluation_result_path):
        """
        get evaluation_result.txt with standard_answer and result.txt by SharedTaskEval.pl
        :param standard_answer_path:
        :param result_path:
        :param evaluation_result_path:
        :return:
        """
        standard_answer_file = open(standard_answer_path, "r+", encoding="GB18030")
        result_file = open(result_path, "r+", encoding="GB18030")
        sum = 0
        TP = 0
        all = 0
        allTP = 0
        TPFN = 0
        TPFP = 0
        while True:
            all += 1
            line = standard_answer_file.readline()
            line2 = result_file.readline()
            if line == '':
                break
            if line == '\n' and line==line2:
                continue
            fields1 = line.strip().split()
            fields2 = line2.strip().split()
            assert fields1[0] == fields2[0]
            label1 = fields1[1]
            label2 = fields2[1]
            if label1 != 'O':
                TPFN += 1
            if label2 != 'O':
                TPFP += 1
            if label1 == label2:
                allTP += 1
            if label1 == label2 and label1 == 'O':
                continue
            sum += 1
            if label1 == label2:
                TP += 1
        print("TP, sum, TP / sum", TP, sum, TP / sum)
        print("allTP,all,allTP/all", allTP, all, allTP / all)
        print("TP,TPFP,TP/TPFP", TP, TPFP, TP / TPFP)
        print("TP,TPFN,TP/TPFN", TP, TPFN, TP / TPFN)
        # print(TP / sum,allTP / all,TP / TPFP,TP / TPFN,end='')
        standard_answer_file.close()
        result_file.close()


if __name__ == '__main__':
    # 不改变任何数据。同步测试
    trainfile_path = "../data/Genia4ERtraining/Genia4ERtask1.iob2"
    # trainfile_path = "../data/trainfile_path.txt"
    featuresfile_path = "featuresfile_path.txt"
    ME_model_path = "ME_model_for_test.txt"

    testfile_path = "../data/Genia4ERtest/Genia4EReval1.raw"
    standard_answer_path = "../data/Genia4ERtest/Genia4EReval1.iob2"

    test_feature_path = "test_feature_path.txt"
    result_path = "result_path.txt"




    mxEnt = maxEntropy()
    mxEnt.get_features(trainfile_path, featuresfile_path)
    print("finish feature")

    start = time.time()

    mxEnt.train_ME(featuresfile_path, ME_model_path)
    print("finish train")
    traintime = time.time() - start
    print(traintime, "s")

    start = time.time()

    mxEnt.getPredict(testfile_path, ME_model_path, test_feature_path, result_path)
    print("finish predict")
    predicttime = time.time() - start
    print(predicttime, "s")

    mxEnt.evaluate(standard_answer_path, result_path, "")
    # 对所有训练语料，使用sum <0.01迭代<500来测试，总共迭代49次，每次大概30s
    # 如下是结果。
    # TP, sum, TP / sum 10926 22303 0.48988925256691923
    # allTP,all,allTP/all 89662 101040 0.8873911322248614
    # TP,TPFP,TP/TPFP 10926 16866 0.647812166488794
    # TP,TPFN,TP/TPFN 10926 19392 0.5634282178217822
    #
    # 迭代1000词的结果
    # TP, sum, TP / sum 11174 22514 0.49631340499244914
    # allTP,all,allTP/all 89699 101040 0.8877573238321457
    # TP,TPFP,TP/TPFP 11174 17413 0.6417044736690978
    # TP,TPFN,TP/TPFN 11174 19392 0.57621699669967大概差1.5%不算多
    # 作为调试完全可以。