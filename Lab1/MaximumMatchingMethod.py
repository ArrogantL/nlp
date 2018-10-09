from Dict import *
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
        if len(word) > maxWordLength:
            maxWordLength = len(word)
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
            # b = findWordInDict(dict, word)
            if j == i or findWordInHashList(hashlist, word) or word == "\n":
                i = j + 1
                break
        if word == "\n":
            continue
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
        if len(word) > maxWordLength:
            maxWordLength = len(word)
    seg = []
    j = len(str) - 1
    hashlist = list2HashList([c[0] for c in dict])
    while j >= 0:
        if j +1- maxWordLength>=0:
            r = range(j +1- maxWordLength, j + 1)
        else:
            r = range(0, j + 1)

        for i in r:
            word = str[i:j + 1]
            if j == i or findWordInHashList(hashlist, word) or word == "\n":
                j=i-1
                break
        if word=="\n":
            continue
        seg.append(word)
    seg.reverse()
    return seg


if __name__ == '__main__':
    dict = readDict("doc/dic.txt")
    testr = """
AlphaGo之父亲授深度强化学习十大法则
"""
    segFMM = FMM(dict, testr)
    segBMM = BMM(dict, testr)
    for word in segFMM:
        if word == "。":
            print(word)
        else:
            print(word,end=" ")
    print("\n")

    for word in segBMM:
        if word == "。":
            print(word)
        else:
            print(word,end=" ")


