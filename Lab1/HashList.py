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