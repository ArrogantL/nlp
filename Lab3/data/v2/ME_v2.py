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
            print(i, max_iter)

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
        print(' ' + str(sum / len(self.w)))
        return sum < 0.01 * len(self.w)

    def predict(self, input):
        features = input.strip().split()
        prob = self.calPyx(features)
        prob.sort(reverse=True)

        return prob

    keywords = ["sequence", "BSAP", "Sp1", "terminal", "A", "binding", "CIITA", "NF", "TNFalpha", "macrophages", "Fas",
                "IL-5", "E2F", "NF-kappa", "virus", "TNF-alpha", "VDR", "2", "B", "resting", "motifs", "primary",
                "endothelial", "monocytes", "kappaB", "HIV", "cytokines", "HeLa", "3-kinase", "promoters", "proteins",
                "transcripts", "element", "1", "calcineurin", "Tat", "C/EBP", "chain", "genes", "domains", "alpha",
                "cytoplasmic", "antibody", "T", "IL-10", "human", "box", "motif", "IFN-gamma", "p65", "CREB", "RAR",
                "C", "peripheral", "construct", "locus", "adhesion", "LTR", "type", "elements", "IkappaBalpha", "'",
                ")", "receptor", "class", "HL-60", "nuclear", "bp", "HIV-1", "mRNA", "AP-1", "TCR", "Bcl-2", "region",
                "immunoglobulin", "I", "NFATp", "IL-6", "signal", "antigens", "kinase", "IL-12", "ER", "acid",
                "product", "Jurkat", "neutrophils", "response", "transcription", "NFAT", "VCAM-1", "complex", "3",
                "MHC", "Oct-1", "and", "regions", "membrane", "hematopoietic", "p50", "IL-2R", "T-cell", "sites",
                "subunit", "repeat", "cytokine", "long", "leukocytes", "antibodies", "PKC", "IL-2", "enhancer",
                "NF-kappaB", "Oct-2", "activated", "TCF-1", "anti-CD3", "IL-4", "clones", "GATA-1", "IFN", "c-Fos",
                "glucocorticoid", "molecules", "tyrosine", "CD28", "LMP1", "PU.1", ",", "NF-AT", "regulatory", "TNF",
                "DNA-binding", "necrosis", "IL-8", "tumor", "c-Jun", "kinases", "blood", "domain", ";", "reporter",
                "IL-1beta", "site", "STAT5", "K562", "molecule", "IL-1", "antigen", "(", "gene", "STAT", "GM-CSF",
                "factor-alpha", "THP-1", "CD3", "progenitors", "transcriptional", "Tax", "c-Rel", "receptors", "gamma",
                "activator", "promoter", "myeloid", "interleukin-2", "to", "STAT1", "mAb", "mononuclear", "line",
                "STAT3", "factors", "complexes", "kappa", "protein", "CD40", "interleukin", "PBMC", "5", "U937",
                "cells", "lines", "lymphocytes", "DNA", "constructs", "lineage", "factor-kappa", "GR", "II", "IL-13",
                "beta", "RNA", "family", "IgE", "NFkappaB", "ICAM-1", "monocytic", "c-fos", "c-jun", "factor",
                "sequences", "cDNA", "colony-stimulating", "surface", "erythroid", "leukemia", "protein-1", "of",
                "cellular", "cell", "Stat3", "normal"]
    keylastword = ["IL-2", "and", "NF-kappa", "through", "by", "the", "surface", "of", "(", "In", "primary", "T",
                   "that", "for", "protein", "tyrosine", "induce", "The", "B", "human", "immunodeficiency", "virus",
                   "type", "2", "in", "HIV-1", "transcription", "factor", "kappa", "long", "terminal", "cis-acting",
                   "two", "binding", ",", "a", "novel", "not", "T-cell", "an", "enhancer", "nuclear", "both",
                   "peripheral", "blood", "this", ".", "NK", "transfected", "target", "viral", "on", "class", "I",
                   "MHC", "infected", "different", "CD4", "with", "major", "histocompatibility", "complex", "II",
                   "using", "monoclonal", "or", "to", "activate", "distinct", "kinase", "erythroid", "progenitor",
                   "cell-specific", "including", "N-terminal", "activation", ")", "interleukin-2", "receptor", "alpha",
                   "control", "IL-2R", "murine", "lymphocyte", "mouse", "+", "which", "normal", "STAT", "hematopoietic",
                   "cell", "myeloid", "lymphoid", "epithelial", "EBV", "gene", "fusion", "encoding", "Human", "several",
                   "DNA", "promoter", "putative", "natural", "killer", "line", "from", "colony-stimulating", "GM-CSF",
                   "growth", "cytoplasmic", "cytokine", "induced", "factors", "functional", "antigen", "multiple", "3",
                   "Jurkat", "amino", "-induced", "beta", "reporter", "interferon", "necrosis", "TNF", "either",
                   "isolated", "factor-kappa", "consensus", "increased", "A", "zinc", "Ets", "its", "as", "between",
                   "glucocorticoid", "monocytic", "specific", "U937", "HIV", "regulatory", "Sp1", "expressed",
                   "interleukin", "activated", "inhibits", "5", "'", "bp", "K562", "other", "active", "cDNA",
                   "purified", "gamma", "into", "X", "proximal", "adhesion", "endothelial", "inhibited", "IL-1",
                   "tumor", "macrophage", "inflammatory", "1", "mature", "negative", "mononuclear", "transcriptional",
                   "cellular", "B-cell", "family", "these", "signal", "DNA-binding", "p65", "containing", "GATA",
                   "early", "stimulated", "inhibit", "at", "CD4+", "AP-1", "activator", "leukemia", "leukemic", "HL-60",
                   "recombinant", "mutant", "express", "THP-1", "response", "element", "c-fos", "induces", "upstream",
                   "inducible", "endogenous", "globin", "adult", "Epstein-Barr", "NF", "membrane", "positive",
                   "beta-globin", "intracellular", "acid", "IL-4", "wild-type", "HeLa", "three", "Ig", "chain",
                   "expressing", "chromosome", "whereas", ";", "resting", "immunoglobulin", "p50", ":", "octamer",
                   "c-jun", "C/EBP", "delta", "IL-6", "Th2", "TCR", "mitogen-activated", "NF-kappaB", "NFAT"]
    keynextword = ["gene", "expression", "B", "activation", "requires", ".", "surface", "receptor", "(", ")", "T",
                   "lymphocytes", "-mediated", "complex", "and", "signaling", "tyrosine", "kinase", "activity", "site",
                   "immunodeficiency", "virus", "type", "2", "enhancer", "but", "cells", "region", "of", "factor",
                   "NF-kappa", "to", "sites", "in", "long", "terminal", "repeat", ",", "is", "elements", ":", "binding",
                   "element", "within", "lines", "which", "has", "blood", "monocytes", "binds", "or", "may", "proteins",
                   "I", "antigen", "NK", "that", "infected", "regions", "class", "II", "molecules", "on", "antibodies",
                   "transcription", "mAb", "could", "kinases", "protein", "involved", "C", "by", "erythroid",
                   "cell-specific", "genes", "human", "promoter", "containing", "domain", "alpha", "lymphocyte", "+",
                   "induces", "mRNA", "transcripts", "lineage", "factors", "cell", "myeloid", "was", "product", "locus",
                   "are", "activated", "induced", "peripheral", "were", "line", "during", "DNA", "sequences", "through",
                   "for", "as", "production", "killer", "colony-stimulating", "growth", "induction", "complexes",
                   "motifs", "3", "domains", "alone", "stimulation", "-induced", "does", "plays", "contains", "beta",
                   "subunits", "A", "subunit", "reporter", "Jurkat", "showed", "isoforms", "gamma", "necrosis", "at",
                   "factor-kappa", "sequence", "function", "motif", "from", "phosphorylation", "stimulated",
                   "regulatory", "constructs", "box", "phosphatase", "LTR", "with", "can", "kappa", "progenitors", "'",
                   "bp", "fragment", "levels", "family", "differentiation", "clone", "cDNA", "U937", "into", "upstream",
                   "pathway", "-dependent", "cytokines", "endothelial", "adhesion", "molecule-1", "nuclear",
                   "expressed", "1", "mononuclear", "also", "activator", "promoters", "did", "tumor", "cytokine",
                   "receptors", "monocytic", "[", "factor-alpha", "CD4+", "resulted", "leukemia", "molecule",
                   "transcriptional", "response", "protein-1", "AP-1", "construct", "ligand", "secretion", "leukocytes",
                   "clones", "membrane", ";", "after", "antibody", "antigens", "have", "inhibitor", "mutant",
                   "encoding", "-", "RNA", "signal", "had", "chain", "synthesis", "early", "DNA-binding", "T-cell",
                   "acid", "expressing", "IL-2", "macrophages", "NF-kappaB", "kappaB", "3-kinase"]

    # TP, sum, TP / sum 4750 23163 0.20506842809653325
    # allTP,all,allTP/all 82626 101040 0.8177553444180523
    # TP,TPFP,TP/TPFP 4750 11746 0.40439298484590497
    # TP,TPFN,TP/TPFN 4750 19392 0.2449463696369637

    # TP, sum, TP / sum 4903 23294 0.21048338627972868
    # allTP,all,allTP/all 82648 101040 0.8179730799683294
    # TP,TPFP,TP/TPFP 4903 12013 0.4081411803879131
    # TP,TPFN,TP/TPFN 4903 19392 0.25283622112211224

    # TP, sum, TP / sum 5648 23608 0.23924093527617757
    # allTP,all,allTP/all 83079 101040 0.8222387173396675
    # TP,TPFP,TP/TPFP 5648 14157 0.3989545807727626
    # TP,TPFN,TP/TPFN 5648 19392 0.29125412541254125
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
        if '-' in word:
            s += ' ' + '-'
        if word.isupper():
            s += ' ' + "upper"
        if any(char.isdigit() for char in word):
            s += ' ' + "has_digit"
        if '.' in word:
            s += ' ' + '.'
        if word[-1] == 's':
            s += ' ' + "plurality"
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
                # if lastword in self.keylastword:
                #     s += " keylastword" + lastword
        # 加入后向，略微提升279 1185 0.23544303797468355
        if not len(nextfield) == 0:
            nextword = nextfield[0]
            if nextword in self.keywords:
                s += " nextKey" + nextword
            # if nextword in self.keynextword:
            #     s += " keynextword" + nextword
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
        standard_answer_file.close()
        result_file.close()


if __name__ == '__main__':
    # mxEnt = maxEntropy()
    # mxEnt.loadData('data/gameLocation.dat')
    # mxEnt.train()
    # print(mxEnt.predict('Sunny Cloudy\n'),mxEnt.predict('Sunny Cloudy\n')[0][1])

    # trainfile_path = "data/trainfile_path.txt"

    trainfile_path = "data/Genia4ERtraining/Genia4ERtask1.iob2"

    featuresfile_path = "data/featuresfile_path.txt"
    ME_model_path = "data/ME_model_path.txt"
    # testfile_path = "data/testfile_path.txt"
    # standard_answer_path = "data/standard_answer_path.txt"
    testfile_path = "data/Genia4ERtest/Genia4EReval1.raw"
    standard_answer_path = "data/Genia4ERtest/Genia4EReval1.iob2"
    test_feature_path = "data/test_feature_path.txt"
    result_path = "data/result_path.txt"
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
    # 仅仅使用词长作为特征 2 69 0.028985507246376812

    # ff=open("timeforalltrainfile.txt","w")
    # ff.write(str(traintime)+' '+str(predicttime))
    # ff.close()

# 2339.090096950531 s
# finish predict
# 10.032165288925171 s
# TP, sum, TP / sum 10822 22218 0.4870825456836799
# allTP,all,allTP/all 89643 101040 0.8872030878859858
# TP,TPFP,TP/TPFP 10822 16609 0.6515744475886568
# TP,TPFN,TP/TPFN 10822 19392 0.5580651815181518
