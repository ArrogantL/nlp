import os
import re
import time
from queue import Queue

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
        if is_abbr and wordshape[-1] == wordshape[-2]:
            wordshape = wordshape[0:-1]
    return wordshape


def process(line, lastlinesqueue, nextline, is_train):
    field = line.strip().split()
    lastfields = []
    for i in range(lastlinesqueue.qsize()):
        lastline = lastlinesqueue.get()
        lastfields.append(lastline.strip().split())
        lastlinesqueue.put(lastline)
    if len(field) == 0:
        return ''
    # len key - UP has_digit . plurality word shapeF shapeT
    word = field[0]
    s = str(len(word))
    s += ' ' + str(word in KEYWORDS) + "Key" + word
    s += ' ' + str('-' in word) + '-'
    s += ' ' + str(word.isupper()) + "upper"
    s += ' ' + str(any(char.isdigit() for char in word)) + "has_digit"
    s += ' ' + str('.' in word) + '.'
    s += ' ' + str(word[-1] == 's') + "plurality"
    s += ' ' + word
    s += ' shape' + get_word_shape(word, is_abbr=False) + ' abbr' + get_word_shape(word, is_abbr=True)
    # 以下代码不用更改
    # 测试
    if len(field) == 1 and not is_train:
        return s
    # 训练
    if len(field) == 2:
        label = field[1]
        s = s + '\t' + label
        return s

    # 错误
    assert False

def get_features(trainfile_path, featuresfile_path, is_train=True):
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
        features = process(line, lastlinesqueue, nextline, is_train)
        lastlinesqueue.put(line)
        limit = 1
        if lastlinesqueue.qsize() > limit:
            lastlinesqueue.get()
            assert lastlinesqueue.qsize() <= limit
        featuresfile.write(features + '\n')
    trainfile.close()
    featuresfile.close()


def train_ME(template_path, featuresfile_path, ME_model_path):
    os.system("crf_learn -c 4.0 -p 8 " + template_path + " " + featuresfile_path + " " + ME_model_path)


def getPredict(testfile_path, ME_model_path, test_feature_path, result_path):
    get_features(testfile_path, test_feature_path, is_train=False)
    os.system("crf_test -m " + ME_model_path + " " + test_feature_path + " >> tempresult.txt")
    tempfile = open("tempresult.txt", "r+")
    result_file = open(result_path, "w", encoding="GB18030")
    testfile = open(testfile_path, "r+", encoding="GB18030")
    while True:
        line = tempfile.readline()
        line2 = testfile.readline()
        if line == '' or line2 == '':
            break
        if line == '\n':
            assert line2 == '\n'
            result_file.write('\n')
            continue
        assert len(line2.strip().split())==1
        result_file.write(line2.strip() + '\t' + line.strip().split()[-1] + '\n')
    tempfile.close()
    result_file.close()
    testfile.close()
    os.system("rm tempresult.txt")


def evaluate(result_path, evaluate_path,
             standard_answer_path):
    os.system("perl data/JNLPBA2004_eval/evalIOB2.pl " + result_path + "  " + standard_answer_path+">"+evaluate_path)
    os.system("cat "+evaluate_path)

if __name__ == '__main__':
    # ctrl+'/'快速取消/添加'#'注释
    # 动态配置
    # trainfile_path = "data/train.txt"
    trainfile_path = "data/Genia4ERtraining/Genia4ERtask1.iob2"

    # ME_model_path = "data/CRF_model.txt"
    ME_model_path = "data/CRF_model_3.txt"

    # testfile_path = "data/test.txt"
    # standard_answer_path = "data/standard_answer.txt"
    testfile_path = "data/Genia4ERtest/Genia4EReval1.raw"
    standard_answer_path = "data/Genia4ERtest/Genia4EReval1.iob2"

    # 静态配置
    template_path = "data/template.txt"
    featuresfile_path = "data/features.txt"
    test_feature_path = "data/test_feature.txt"
    result_path = "data/result.txt"
    evaluate_path = "data/evaluate_result.txt"

    # 提取特征文件
    get_features(trainfile_path, featuresfile_path)
    print("finish feature")
    # 训练模型
    start = time.time()
    train_ME(template_path, featuresfile_path, ME_model_path)
    print("finish train：", time.time() - start, 's')
    # 识别实体
    start = time.time()
    getPredict(testfile_path, ME_model_path, test_feature_path, result_path)
    print("finish predict:", time.time() - start, 's')
    # 评分
    evaluate(result_path, standard_answer_path=standard_answer_path, evaluate_path=evaluate_path)

