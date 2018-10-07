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
        elif c ==None:
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
    seg = []
    i = 0
    hashlist=list2HashList([c[0] for c in dict])
    while i < len(str):
        r = range(i, len(str))
        b = False
        for j in r[::-1]:
            word = str[i:j + 1]


            #b = findWordInDict(dict, word)
            b=findWordInHashList(hashlist,word)

            if word == "\n":
                b=True
            if b:
                i = j + 1
                break
        if b == False:
            return "UnMatch Part: " + str[i:-1]
        seg.append(word)
        print(word)
    return seg


def BMM(dict, str):
    """
    最大反向匹配
    :param dict: 一个tuple_list，tuple=(词，词频)
    :param str: 分词结果list
    :return: 分词结果list
    """
    seg = []
    j = len(str) - 1
    while j >= 0:
        r = range(0, j + 1)
        b = False
        for i in r:
            word = str[i:j + 1]
            b = findWordInDict(dict, word)
            if word == "\n":
                b=True
            if b:
                j = i - 1
                break

        if b == False:
            return "UnMatch Part: " + str[0:j + 1]
        seg.append(word)
    seg.reverse()
    return seg


if __name__ == '__main__':
    dict = readDict("doc/dic.txt")
    testr = """
徐武丁
生性喜爱大树，每见到它们，就涌起一番崇敬之情。我
山、路口，或蓊蓊郁郁，或瘦削峭拔巍然兀立；有时枯槁中却见出新枝，嫩叶在枝桠间簇拥成团，一棵或几棵，多者则成片，永远是那么安详地守望着朝霞和炊烟，成为村子的屏障和福佑，其中倘能享受村人设坛以香火供奉的，已被农人视为村子的龙脉风水。一到夏日，许多大树底下也是人们下桩拴牛的地方；有的还是农民和路人遮荫蔽日的所在。大树多半以经年的常绿渲染着村子的生命韵律，升华为一种境界：沉着、淳朴、富于耐力。大树是田园牧歌的重音符号，它像是一篇寓言，托举着一片神秘、象征的天空。
我幼小时候住处不远有一棵古枫，一到落霜之后，叶片被染得鲜亮通红，那被霜浸透的红色像是慢慢洇开去，留下好看的色韵。红叶落满一地，成了松软的地毯，便在上翻滚、摔跤，把红叶拥在身上假寐，仿佛唯此才不亏待这良辰美景，现在想来，那欢乐的情景真像绝妙的童话世界。那时正值饥寒时月，古枫却像母亲一样给予我不少温暖时光。广东新会有棵古榕，因巴金一篇散文而成名胜，叫“小鸟天堂”，我想那名字真个是好，其实，所有的大树又何尝不是小鸟的家园小鸟的天堂！
我说的大树可以是经几番枯荣历千年风霜的老寿星，也可以是小视群芳、一览周遭数人十数人方可合抱的伟丈夫，它们有着抵御冰雪雷霆的丰富经历，见过河东河西的变迁，甚至有过抗争烽火离乱的感人故事，它们的根深入人生、自然、历史，它们的情感凝集着人类心灵的信息。多多的纷繁世象、功罪评说，全都蕴含在它们迎风颔首之中。倘若《天仙配》中做媒的不是槐荫耆宿，那姻缘便要失去许多色彩。
我不能具体地计算出一棵千年大树一天一年可涵养多少水量，十吨？百吨？但我看见大树失去的地方河水确实无误地减少。大树的根总是永不懈怠地扎向地层深处，大树的臂膀总是举向万籁复响的天空，它朴拙却玄妙，一副以不变应万变的王者风仪。它把生命的过程与风霜雨雪日月星辰融于一体，落叶或蘖枝，都记录着它们相互的对话和情感历程，一一把它们刻入了年轮。我不知道许多大树的消逝于我们是否仅仅失去一些相和相亲的同伴，仅仅失去谆谆教诲和常常叮咛告诫并用绿荫庇护我们的长者？值得思考的是，现代人能三天一栋地盖起摩天高楼，却不能高速度造出用生命的和弦弹奏天籁的百年千年的古树。每当我见到一些大树被人们以围栏或支撑加以护卫的情景时，就感到欣慰。
文物是历史的记录，而古树，就是历史，它是自然和生活的乐章。
"""
    segFMM = FMM(dict, testr)
    #segBMM = BMM(dict, testr)
    print(segFMM)
    #print(segBMM)

