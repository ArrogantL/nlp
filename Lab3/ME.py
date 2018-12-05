import time
from collections import defaultdict
from queue import Queue
import numpy as np
from ME_external import MaxEntropy
import os


KEYWORDS = ["sequence", "BSAP", "Sp1", "terminal", "A", "binding", "CIITA", "NF", "TNFalpha", "macrophages", "Fas",
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


def get_word_shape(word, is_abbr=False):
    """
    获取词形
    :param word: 目标词
    :param is_abbr: 是否将词形中连续字符合并，如Xxxdxx合并为Xxdx
    :return: 词型或缩写词型
    """
    word_shape = "shape:"
    if is_abbr:
        word_shape = "abbr_" + word_shape
    for i in word:
        if i.isupper():
            word_shape += 'X'
        elif i.islower():
            word_shape += 'x'
        elif i.isdigit():
            word_shape += 'd'
        else:
            word_shape += '-'
        if is_abbr and word_shape[-1] == word_shape[-2]:
            word_shape = word_shape[0:-1]

    return word_shape


def process(line, last_line_squeue, next_line, isTrain):
    """
    按行将训练文本或测试文本转为特征文本
    :param line:
    :param last_line_squeue:
    :param next_line:
    :param isTrain:
    :return:
    """
    field = line.strip().split()
    next_field = next_line.strip().split()
    last_fields = []
    for i in range(last_line_squeue.qsize()):
        last_line = last_line_squeue.get()
        last_fields.append(last_line.strip().split())
        last_line_squeue.put(last_line)
    # 处理空行
    if len(field) == 0:
        return ''
    # 1-gram
    word = field[0]
    s = str(len(word))
    if word in KEYWORDS:
        s += ' ' + "keyword:" + word
    if '-' in word:
        s += ' ' + 'has_-'
    if word.isupper():
        s += ' ' + "upper"
    if any(char.isdigit() for char in word):
        s += ' ' + "has_digit"
    if '.' in word:
        s += ' ' + 'has_.'
    if word[-1] == 's':
        s += ' ' + "is_plurality"
    s += ' ' + word
    s += ' ' + get_word_shape(word, is_abbr=False) + ' ' + get_word_shape(word, is_abbr=True)
    # 2-gram
    for lastfield in last_fields:
        if not len(lastfield) == 0:
            lastword = lastfield[0]
            if lastword in KEYWORDS:
                s += " lastKey" + lastword
    if not len(next_field) == 0:
        nextword = next_field[0]
        if nextword in KEYWORDS:
            s += " nextKey" + nextword
    # 测试
    if len(field) == 1 and not isTrain:
        return s
    # 训练
    if len(field) == 2:
        label = field[1]
        s = label + ' ' + s
        return s
    # 错误
    assert False


def get_features(targetfile_path, featuresfile_path, is_train=True):
    trainfile = open(targetfile_path, "r+", encoding="GB18030")
    featuresfile = open(featuresfile_path, "w", encoding="GB18030")
    lastlinesqueue = Queue()
    lastlinesqueue.put('')
    nextline = trainfile.readline()
    while True:
        line = nextline
        if line == '':
            break
        nextline = trainfile.readline()
        features = process(line, lastlinesqueue, nextline, is_train)
        lastlinesqueue.put(line)
        # 经测试1最好
        limit = 1
        if lastlinesqueue.qsize() > limit:
            lastlinesqueue.get()
            assert lastlinesqueue.qsize() <= limit
        featuresfile.write(features + '\n')
    trainfile.close()
    featuresfile.close()


def train_ME(featuresfile_path, ME_model_path):
    mxEnt = MaxEntropy()
    mxEnt.loadData(featuresfile_path)
    mxEnt.train()
    ME_model = open(ME_model_path, "w", encoding="GB18030")
    for label, f in mxEnt.features:
        id = mxEnt.features[(label, f)]
        fw = mxEnt.w[id]
        ME_model.write("%s %s %f\n" % (label, f, fw))
    ME_model.close()


def getPredict(testfile_path, ME_model_path, test_feature_path, result_path):
    get_features(testfile_path, test_feature_path, is_train=False)
    ME_model_file = open(ME_model_path, "r", encoding="GB18030")
    mxEnt = MaxEntropy()
    mxEnt.features = defaultdict(int)  # 用于获得(标签，特征)键值对
    mxEnt.labels = set([])  # 标签
    mxEnt.w = []
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
        mxEnt.features[(label, feature)] = id
        mxEnt.w.append(w)
        mxEnt.labels.add(label)
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
            result_file.write(line)
            continue
        prop = mxEnt.predict(line)
        result_file.write(line2[:-1] + "\t" + prop[0][1] + '\n')
    test_feature_file.close()
    result_file.close()
    testfile.close()


def evaluate(result_path,
             standard_answer_path, evaluate_path):
    os.system("perl data/JNLPBA2004_eval/evalIOB2.pl " + result_path + "  " + standard_answer_path + " >"+
    evaluate_path)
    os.system("cat " + evaluate_path)

if __name__ == '__main__':
    # ctrl+'/'快速取消/添加'#'注释
    # 动态配置
    # trainfile_path = "data/train.txt"
    trainfile_path = "data/Genia4ERtraining/Genia4ERtask1.iob2"

    # testfile_path = "data/test.txt"
    # standard_answer_path = "data/standard_answer.txt"
    testfile_path = "data/Genia4ERtest/Genia4EReval1.raw"
    standard_answer_path = "data/Genia4ERtest/Genia4EReval1.iob2"
    # 静态配置
    ME_model_path = "data/ME_model.txt"
    featuresfile_path = "data/features.txt"
    test_feature_path = "data/test_feature.txt"
    result_path = "data/result.txt"
    evaluate_path = "data/evaluate_result.txt"

    # 提取特征
    get_features(trainfile_path, featuresfile_path)
    print("finish feature")
    # 训练模型
    start = time.time()
    train_ME(featuresfile_path, ME_model_path)
    print("finish train：",time.time() - start,'s')
    # 识别实体
    start = time.time()
    getPredict(testfile_path, ME_model_path, test_feature_path, result_path)
    print("finish predict:",time.time() - start,'s')
    # 评分
    evaluate(result_path,standard_answer_path=standard_answer_path,evaluate_path=evaluate_path)
