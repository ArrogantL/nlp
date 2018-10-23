from Dict import *
import time

def list2HashList(wordlist):
    hashList = [None] * len(wordlist) * 2
    l = len(hashList)
    for word in wordlist:
        hash = getHashCode(word)
        for i in range(l):
            index = (hash + i) % l
            if hashList[index] == None:
                hashList[index] = word
                break
    return hashList


def getHashCode(words):
    s = 17
    for c in words:
        s = 31 * s + ord(c)
    return s


def findWordInHashList(hashList, word):
    hash = getHashCode(word)
    l = len(hashList)
    for i in range(l):
        index = (hash + i) % l
        c = hashList[index]
        if c == word:
            return True
        elif c == None:
            return False
    return False


def findWordInDict(dict, word):
    """
    在字典中查找单词-顺序查找法
    :param dict: 一个tuple_list，tuple=(词，词频)
    :param word: 要查找的单词
    :return: 找到返回True，否则False
    """

    for entry in dict:
        if entry[0] == word:
            return True
    return False


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

            # b = findWordInDict(dict, word)
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
            if j == i or findWordInHashList(hashlist, word) or word == "\n":
                j = i - 1
                break
        seg.append(word)
    seg.reverse()
    return seg



def analyzeMM(seg_corpus, seg_MM):
    """
    正常检测模式：每次字符相同，如果是空格就+1分。开始新的匹配。如果出现不同字符进入重定位异步匹配。
    重定位异步匹配：是空格的一方继续后移动到不是空格，检测字符是否相等，不等就说明测试文件不对，报错。无错误就进入重定位同步匹配过程。
    重定位同步匹配：每次字符相同，如果是空格就进入正常检测模式，但是不加分。如果出现不同，则进入重定位异步匹配。
    :param seg_corpus:
    :param seg_MM:
    :return:
    """
    # "doc/199801_seg_normalized.txt"
    fcorpus = open(seg_corpus, "r+")
    fMM = open(seg_MM, "r+")
    TP = 0
    Tall=0
    Pall=0
    while True:
        lc = fcorpus.readline()
        lm = fMM.readline()
        if lc == '' or lm == '':
            break
        len_lc = len(lc)
        len_lm = len(lm)

        c=0
        m=0
        while True:
            if c==len_lc:
                break
            if lc[c]==' ':
                Tall+=1
            c+=1
        while True:
            if m ==len_lm:
                break
            if lm[m]==' ':
                Pall+=1
            m+=1



        i = 0
        j = 0
        flag = 0



        while True:
            if i == len_lc or j == len_lm:
                break
            if flag == 0:
                if lc[i] != lm[j]:
                    flag = 1
                    continue
                if lc[i] == " ":
                    TP += 1
                if i + 1 == len_lc and j + 1 == len_lm:
                    TP += 1
                i += 1
                j += 1
            elif flag == 1:
                if lc[i] == " ":
                    i += 1
                else:
                    j += 1
                assert lc[i] == lm[j]
                flag = 2
            else:
                if lc[i] != lm[j]:
                    flag = 1
                    continue
                if lc[i] == " ":
                    flag = 0
                i += 1
                j += 1


    # 关闭打开的文件
    fcorpus.close()
    fMM.close()


    return TP,Tall,Pall
def testAnalyze():
    TP = 0

    lc = "as d fg hj kl"
    lm = "as d fg hjkl"

    i = 0
    j = 0
    flag = 0
    len_lc = len(lc)
    len_lm = len(lm)
    while True:
        if i == len_lc or j == len_lm:

            break
        if flag == 0:
            if lc[i] != lm[j]:
                flag = 1
                continue
            if lc[i] == " " :
                TP += 1
            if (i+1==len_lc or lc[i+1]==" ") and (j+1==len_lm or lm[j+1]==" "):
                TP+=1
            i += 1
            j += 1
        elif flag == 1:
            if lc[i] == " ":
                i += 1
            else:
                j += 1
            assert lc[i] == lm[j]
            flag = 2
        else:
            if lc[i] != lm[j]:
                flag = 1
                continue
            if lc[i] == " ":
                flag = 0
            i += 1
            j += 1
        print(TP)
    print(TP)



if __name__ == '__main__':
    dict = readDict("doc/dic.txt")
    testr = """
"""

    #start = time.time()
    #segFMM = FMM(dict, testr)

    #segBMM = BMM(dict, testr)
    #print(time.time()-start)
    #TP, Tall, Pall=analyzeMM("doc/199801_seg_normalized.txt","doc/seg_BMM.txt")
    #TP, Tall, Pall=analyzeMM("doc/seg_test.txt","doc/seg_BMM_test.txt")
    #print(TP,TP/Tall,TP/Pall)


"""
    with open("doc/seg_FMM.txt", 'w') as f:
        for word in segFMM:
            if word=="\n":
                f.write(word)

            else:
                f.write(word+" ")
"""

