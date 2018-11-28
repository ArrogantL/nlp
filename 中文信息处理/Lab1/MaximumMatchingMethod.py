from Dict import *
import time
from AnalyzeSeg import *
from HashList import *


def FMM(dict, str):
    """
    最大正向匹配
    :param dict: 一个tuple_list，tuple=(词，词频)
    :param str: 要分词的句子
    :return: 分词结果list
    """
    maxWordLength = 0
    for word in dict:
        if len(word[0]) > maxWordLength:
            maxWordLength = len(word[0])
    lenlist=[0] * (maxWordLength+1)
    for word in dict:
        lenlist[len(word[0])]=1
    seg = []
    i = 0
    hashlist = list2HashList([c[0] for c in dict])
    while i < len(str):
        if i + maxWordLength >= len(str):
            r = range(i, len(str))
        else:
            r = range(i, i + maxWordLength)
        for j in r[::-1]:
            word = str[i:j + 1]
            if lenlist[len(word)]==0:
                continue
            if j == i or findWordInHashList(hashlist, word) or word == "\n":
                i = j + 1
                break
        seg.append(word)
    return seg


def BMM(dict, str):
    """
    最大反向匹配
    :param dict: 一个tuple_list，tuple=(词，词频)
    :param str: 分词结果list
    :return: 分词结果list
    """
    maxWordLength = 0
    for word in dict:
        if len(word[0]) > maxWordLength:
            maxWordLength = len(word[0])
    lenlist = [0] * (maxWordLength + 1)
    for word in dict:
        lenlist[len(word[0])] = 1
    seg = []
    j = len(str) - 1
    hashlist = list2HashList([c[0] for c in dict])
    while j >= 0:
        if j + 1 - maxWordLength >= 0:
            r = range(j + 1 - maxWordLength, j + 1)
        else:
            r = range(0, j + 1)
        for i in r:
            word = str[i:j + 1]
            if lenlist[len(word)]==0:
                continue
            if j == i or findWordInHashList(hashlist, word) or word == "\n":
                j = i - 1
                break
        seg.append(word)
    seg.reverse()
    return seg


def IOMM(sentfilename,dicfilename):
    """
    根据字典来正反向最大匹配sent，输出到结果到seg_FMM.txt seg_BMM.txt
    存储格式：每个划分之间有一个空格，行末一个空格。换行与原文相同。
    :param sentfilename:待分词文件
    :param dicfilename:词典名，词典格式：一行一个记录，“词”+“ ”+“词频”
    :return timealls,dicttime,stringtime,ftime,btime,savetime
    """
    timealls = time.time()
    dicttime=time.time()
    dict = readDict(dicfilename)
    dicttime=time.time()-dicttime

    stringtime=time.time()
    fo = open(sentfilename, "r+", encoding="GB18030")
    targetstring=''
    while True:
        line = fo.readline()
        if line == '':
            break
        targetstring+=line
        # 关闭打开的文件
    fo.close()
    stringtime=time.time()-stringtime

    ftime=time.time()
    segFMM = FMM(dict, targetstring)
    ftime=time.time()-ftime

    btime=time.time()
    segBMM = BMM(dict, targetstring)
    btime=time.time()-btime
    savetime=time.time()
    with open("data/seg_FMM.txt", 'w') as f:
        for word in segFMM:
            if word == "\n":
                f.write(word)
            else:
                f.write(word + " ")
    with open("data/seg_BMM.txt", 'w') as f:
        for word in segBMM:
            if word == "\n":
                f.write(word)
            else:
                f.write(word + " ")
    savetime=time.time()-savetime
    timealls=time.time()-timealls
    #(155.89176988601685, 0.08421754837036133, 0.024581193923950195, 79.16486144065857, 76.1103937625885, 0.5077102184295654)
    return timealls,dicttime,stringtime,ftime,btime,savetime
if __name__ == '__main__':
    print(IOMM("data/199801_sent.txt", "data/dic.txt"))
    TP, Pall, Tall, P, R, F = analyzeSeg("data/199801_seg.txt", "data/seg_FMM.txt")
    TP, Pall, Tall, P, R, F = analyzeSeg("data/199801_seg.txt", "data/seg_BMM.txt")

