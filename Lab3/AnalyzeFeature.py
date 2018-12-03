from collections import defaultdict

def nexttopword():
    f = open("data/Genia4ERtraining/Genia4ERtask1.iob2", "r")
    ff = open("task1nextwordup50.txt", "w")
    nextword=''
    topwords = defaultdict(int)
    list=[]
    while True:
        line = f.readline()
        if line == '':
            break
        fields = line.strip().split()
        if len (fields)==0:
            continue
        word=fields[0]
        label=fields[1]
        list.append((word,label))
    f.close()

    for i in range(len(list)):
        if list[i][1]!='O':
            topwords[list[i+1][0]]+=1


    for word in topwords:
        if topwords[word]>50:

            ff.write('"' + word + '",')

    ff.close()
def lasttopword():
    f = open("data/Genia4ERtraining/Genia4ERtask1.iob2", "r")
    ff = open("task1lastwordup50.txt", "w")
    lastword=''
    topwords = defaultdict(int)

    while True:
        line = f.readline()
        if line == '':
            break
        fields = line.strip().split()
        if len (fields)==0:
            continue
        word=fields[0]
        label=fields[1]
        if label!='O':
            topwords[lastword]+=1
        lastword=word
    f.close()
    for word in topwords:
        if topwords[word]>50:

            ff.write('"' + word + '",')

    ff.close()
def task1():
    f = open("data/Genia4ERtraining/Genia4ERtask1.iob2", "r")
    ff = open("task1lastwordup50.txt", "w")
    dict = {}

    while True:
        line = f.readline()
        if line == '':
            break
        fields = line.strip().split()
        if len(fields) == 0 or fields[1] == 'O':
            continue
        dict.setdefault(fields[1], defaultdict(int))
        dict[fields[1]][fields[0]] += 1
    f.close()
    topwords = set()
    for label in dict:
        # ff.write(label + ' #' + '\n')
        sorted_list = sorted(dict[label].items(), key=lambda item: item[1], reverse=True)
        for word, count in sorted_list:
            # ff.write(word +' '+str(count)+ '\n')
            if count > 50:
                topwords.add(word)
    for word in topwords:
        ff.write('"' + word + '",')

    ff.close()
    # 先解决重点，因此对出现的数量进行统计
    # 对数量进行观察，发现是集中式的。考虑从关键词入手
    # 针对关键词在各个label中出现的数量进行分析，考察其类后验概率是否足够大，
    # 将这些关键词作为类标记，当词为keyword是加入本身，如果前一个词是则加上前缀last：8 79 0.10126582278481013
    # 效果提升但是不太好
    # 加入考察是否全部字母大写：8 79 0.10126582278481013没区别，但是认为提升了O的识别率
    # 加入考察是否含有数字：10 78 0.1282051282051282
if __name__ == '__main__':
    nexttopword()



