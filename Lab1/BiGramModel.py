import re
from math import log
from MaximumMatchingMethod import *
def getBigramDict(filename):
    """
    根据文件，获取包括^$的二元概率列表
    格式：
    dict{条件词:{(目标词1,概率),(目标词2,概率),,,(目标词n,概率)}}
    :param filename:
    :return: dict
    """
    # 打开 "doc/199801_seg_normalized.txt"

    fo = open(filename, "r+",encoding="GB18030")
    dict = {}

    while True:
        line = fo.readline()
        if line == '':
            break

        words = re.split("^.*?/m +|/[a-zA-Z\]]+ *|\[", line)
        firstWord = "HEAD"
        for word in words:

            if word == '' or word == '\n':
                dict.setdefault(firstWord, {})
                dict[firstWord].setdefault("TAIL", 1)
                # TODO +1法平滑
                dict[firstWord]["TAIL"] += 1
                continue

            dict.setdefault(firstWord, {})
            dict[firstWord].setdefault(word, 1)
            # TODO +1法平滑
            dict[firstWord][word] += 1
            firstWord = word

        # 关闭打开的文件
    fo.close()
    for w in dict:
        s = sum(dict[w].values())

        for t in dict[w]:
            dict[w][t] /= s
        dict[w]["sum_of_next_word"] = s

    return dict
def getPYX(Y, X, dict):
    # TODO 平滑算法
    """
    返回条件概率。
    :param Y:
    句首 “HEAD”
    句尾 “TAIL”
    :param X:
    :param dict:
    :return:
    """
    if X in dict.keys() and Y in dict[X].keys():

        return dict[X][Y]
    elif X in dict.keys():
        return 1 / dict[X]["sum_of_next_word"]
    else:
        return 1
def getPrefdict(filename):
    """
    获得前缀词典
    :param filename:
    :return:
    """
    lfreq = {}

    with open(filename) as f:
        line = f.readline()
        while len(line) > 0:

            word, freq = line.split()[0:2]
            freq = int(freq)
            lfreq[word] = freq

            for ch in range(len(word)):
                wfrag = word[:ch + 1]
                if wfrag not in lfreq:
                    lfreq[wfrag] = 0
            line = f.readline()
    return lfreq
def getDAG(sentence, lfreq):
    """
    返回DAG list
    格式
    {k1:[i1,i2,,,in]，k2:[i1,i2,,,in]，，，kn:[i1,i2,,,in]}
    其中sen[k,i+1]表视为i
    :param sentence:
    :param lfreq:
    :return:
    """
    DAG = {}
    N = len(sentence)
    for k in range(N):
        tmplist = []
        i = k
        frag = sentence[k]
        while i < N and frag in lfreq:
            if lfreq[frag] > 0:
                tmplist.append(i)
            i += 1
            frag = sentence[k:i + 1]
        if not tmplist:
            tmplist.append(k)
        DAG[k] = tmplist
    return DAG
def bigramMatching(dict, sentence, lfreq):
    DAG = getDAG(sentence, lfreq)
    routeDAG = {}
    N = len(sentence)
    routeDAG[N - 1] = []
    routeDAG[N - 1].append((N - 1, "TAIL", 1))
    for idx in range(N - 2, -1, -1):
        routeDAG[idx] = []
        for x in DAG[idx]:
            if x ==N-1:
                value, y = 1, "TAIL"

            else:
                value, y = max(
                (getPYX(sentence[idx:x + 1], sentence[x + 1:y + 1], dict) * v, y) for y, m, v in routeDAG[x + 1])
            routeDAG[idx].append((x, y, value))
    value, x = max((getPYX(sentence[0:x + 1], "HEAD", dict) * v, x) for x, m, v in routeDAG[0])
    route = []
    s = 0
    next = x
    while True:
        route.append(s)
        for m, n, v in routeDAG[s]:
            if next == m:
                s = next + 1
                next = n
                break
        if next == "TAIL":
            break

    seg = []
    first = 0
    for t in route:
        if t == 0:
            continue
        seg.append(sentence[first:t])
        first = t
    seg.append(sentence[first:])
    return route, value, seg
def IOBiGramModel(sentfilename,dicfilename):
    fo=open(sentfilename, "r+", encoding="GB18030")
    sentence=''
    while True:
        line = fo.readline()
        if line == '':
            break
        sentence+=line
        # 关闭打开的文件
    fo.close()
    lfreq = getPrefdict(dicfilename)
    dict = getBigramDict("doc/199801_seg.txt")
    route, value, seg = bigramMatching(dict, sentence, lfreq)
    with open("doc/seg_BiGram.txt", 'w') as f:
        for word in seg:
            if word=="\n":
                f.write(word)
            else:
                f.write(word+' ')

if __name__ == '__main__':
    IOBiGramModel("doc/199801_sent.txt","doc/dic.txt")
    TP, Pall, Tall, P, R, F = analyzeSeg("doc/199801_seg.txt", "doc/seg_BiGram.txt")
    # TP, Pall, Tall, P, R, F = analyzeMM(corpusfilename, targetfilename)


