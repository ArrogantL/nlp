from collections import defaultdict

src="data/Genia4ERtest/Genia4EReval1.iob2"
def allnotO():
    srcfile=open(src,"r")
    tagfile=open("notOwithOrder.txt","w")
    dict = {}
    while True:
        line = f.readline()
        if line == '':
            break
        fields = line.strip().split()
        if len(fields) == 0 or fields[1] == 'O':
            continue

    f.close()
    ff.close()
if __name__ == '__main__':
    f = open("data/Genia4ERtraining/Genia4ERtask1.iob2", "r")
    ff = open("task1-1o%.txt", "w")
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
    for label in dict:
        # ff.write(label + ' #' + '\n')
        sorted_list = sorted(dict[label].items(), key=lambda item: item[1], reverse=True)
        i=0
        limit=len(sorted_list)//10
        for word,count in sorted_list:
            # ff.write(word +' '+str(count)+ '\n')
            if i<limit:
                 ff.write('"'+str(count)+'",')


    ff.close()
# 先解决重点，因此对出现的数量进行统计
# 对数量进行观察，发现是集中式的。考虑从关键词入手
# 针对关键词在各个label中出现的数量进行分析，考察其类后验概率是否足够大，
# 将这些关键词作为类标记，当词为keyword是加入本身，如果前一个词是则加上前缀last：8 79 0.10126582278481013
# 效果提升但是不太好
# 加入考察是否全部字母大写：8 79 0.10126582278481013没区别，但是认为提升了O的识别率
# 加入考察是否含有数字：10 78 0.1282051282051282


