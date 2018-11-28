from Dict import *
import time
def analyzeSeg(corpusfilename, targetfilename):
    """
    正常检测模式：每次字符相同，如果是空格就+1分。开始新的匹配。如果出现不同字符进入重定位异步匹配。
    重定位异步匹配：是空格的一方继续后移动到不是空格，检测字符是否相等，不等就说明测试文件不对，报错。无错误就进入重定位同步匹配过程。
    重定位同步匹配：每次字符相同，如果是空格就进入正常检测模式，但是不加分。如果出现不同，则进入重定位异步匹配。
    :param seg_corpus:
    :param seg_MM:
    :return:TP,TP+FP,TP+FN,precision=TP/(TP+FP),recall=TP/(TP+FN),F=2*PR/(P+R)
    """
    # "data/199801_seg_normalized.txt"
    fcorpus = open(corpusfilename, "r+",encoding="GB18030")
    #fcorpus = open(corpusfilename, "r+")
    corpusseg=[]
    while True:
        line = fcorpus.readline()
        if line == '':
            break
        words = re.split("^.*?/m +|/[a-zA-Z\]]+ *|\[", line)
        #words = re.split(" ", line)
        for word in words:
            if word == '' or word == '\n':
                continue
            corpusseg.append(word)
    fcorpus.close()
    ftarget = open(targetfilename, "r+")
    targetseg = []
    while True:
        line = ftarget.readline()
        if line == '':
            break
        words = re.split("[ \n]", line)
        for word in words:
            #TODO 特别容易写错的地方！下面是错误实例
            #if word == '' or '\n':
            if word == '' or word=='\n':
                continue
            targetseg.append(word)
    ftarget.close()

    sum=0
    for w in corpusseg:
        sum+=len(w)
    for w in targetseg:
        sum-=len(w)
    assert sum==0

    #开始比对
    i=0
    j=0
    lcorpus=len(corpusseg)
    ltarget=len(targetseg)
    TP=0
    #0：匹配阶段 False：重定位阶段，即出现不匹配到最近的下一次匹配之间的阶段
    flag=0
    # 移进的字符数量差corpos-target
    ct=0

    while True:
        if i==lcorpus or j==ltarget:
            break

        if flag==0:
            ct += len(corpusseg[i]) - len(targetseg[j])
        elif flag==1:
            ct -= len(targetseg[j])
        else:
            ct += len(corpusseg[i])
        if ct==0:
            if flag==0:
                TP+=1
            i+=1
            j+=1
            flag = 0
        elif ct>0:
            j+=1
            flag = 1
        else:
            i+=1
            flag = 2
    FP=len(targetseg)-TP
    FN=len(corpusseg)-TP
    P=TP / (TP + FP)
    R=TP / (TP + FN)
    print("TP, Pall, Tall, P, R, F = ", end='')
    print(TP, TP + FP, TP + FN,  P,R,  2 * P*R / (P + R))
    return TP, TP + FP, TP + FN,  P,R,  2 * P*R / (P + R)
