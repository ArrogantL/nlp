import re
from math import log
from MaximumMatchingMethod import analyzeSeg, analyzeMM


def getPrefdict(filename):
    """
    获得前缀词典
    :param filename:
    :return:
    """
    lfreq = {}
    ltotal = 0
    with open(filename) as f:
        line = f.readline()
        while len(line) > 0:

            word, freq = line.split()[0:2]
            freq = int(freq)
            lfreq[word] = freq
            ltotal += freq
            for ch in range(len(word)):
                wfrag = word[:ch + 1]
                if wfrag not in lfreq:
                    lfreq[wfrag] = 0
            line = f.readline()
    return lfreq, ltotal


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

def getFreq(lfreq,word):
    if word in lfreq.keys() and lfreq[word]>0:

        return lfreq[word]
    else:
        return 1

def unigramMatching(sentence, filename_of_dict):
    lfreq, ltotal = getPrefdict(filename_of_dict)
    DAG = getDAG(sentence, lfreq)
    N = len(sentence)
    route = [(0, 0)] * (N + 1)
    route[N] = (0, -1)
    logtotal = log(ltotal)
    for idx in range(N - 1, -1, -1):
        route[idx] = max((log(getFreq(lfreq,sentence[idx:x + 1])) - logtotal + route[x + 1][0], x) for x in DAG[idx])
    r = []
    s = 0
    next = route[0][1]
    next += 1
    while True:
        r.append(s)
        s = next
        next = route[s][1] + 1
        if next == 0:
            break
    seg = []
    first = 0
    for t in r:
        if t == 0:
            continue
        seg.append(sentence[first:t])
        first = t
    seg.append(sentence[first:])
    return route, route[0][0], seg

def IOUniGramModel(sentfilename,dicfilename):
    fo=open(sentfilename, "r+", encoding="GB18030")
    sentence=''
    while True:
        line = fo.readline()
        if line == '':
            break
        sentence+=line
        # 关闭打开的文件
    fo.close()
    route, v, seg=unigramMatching(sentence,dicfilename)
    with open("doc/seg_UniGram.txt", 'w') as f:
        for word in seg:
            if word=="\n":
                f.write(word)
            else:
                f.write(word+' ')

if __name__ == '__main__':
    IOUniGramModel("doc/199801_sent.txt","doc/dic.txt")
    TP, Pall, Tall, P, R, F = analyzeSeg("doc/199801_seg.txt", "doc/seg_UniGram.txt")

