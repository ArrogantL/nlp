import re
import sys


def getDict(filename, encoding='GB18030'):
    """
    从已经分好词的「PFR人民日报标注语料库文件」中读取出按出现频率由大到小排列的字典。
    :param filename:PFR人民日报标注语料库文件
    :return:一个tuple_list，tuple=(词，词频)
    """
    fo = open(filename, "r+", encoding=encoding)
    dict = {}
    while True:
        line = fo.readline()
        if line == '':
            break
        # ^.*?/m| |/[a-zA-Z\]]+
        words = re.split("^.*?/m +|/[a-zA-Z\]]+ +|\n|\[[^/]", line)

        for word in words:
            if word == '':
                continue
            dict.setdefault(word, 1)
            dict[word] += 1
    # 关闭打开的文件
    fo.close()
    return sorted(dict.items(), key=lambda item: item[1], reverse=True)


def saveDict(dict, filename):
    """
    将getDict获取的dict存入文件。
    格式： 词+空格+词频+换行
    :param dict: 一个tuple_list，tuple=(词，词频)
    :param filename:  存储路径
    """
    with open(filename, 'w') as f:
        for entry in dict:
            f.write(entry[0] + ' ' + str(entry[1]) + '\n')


def readDict(filename, encoding='UTF-8'):
    fo = open(filename, "r+", encoding=encoding)
    dict = []
    while True:
        line = fo.readline()
        if line == '':
            break
        words = re.split(" +|\n",line)
        dict.append((words[0],words[1]))
    # 关闭打开的文件
    fo.close()
    return dict


def analyzeDict(dict):
    """
    简单的对不同长度的词在dict中出现的概率进行统计。
    :param dict:
    :return:字典{长度：频率}
    """
    result = {}
    for entry in dict:
        le = len(entry[0])
        result.setdefault(le, 1)
        result[le] += 1
    return result


if __name__ == '__main__':
    #dict = getDict("doc/199801_seg.txt")
    # saveDict(dict, "doc/dic.txt")
    #analyzeResult = analyzeDict(dict)
    #print(analyzeResult)
    dict=readDict("doc/dic.txt")
    print(dict)

