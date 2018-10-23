from math import log

def getBigramDict(filename):
    """
    根据文件，获取包括^$的二元概率列表
    格式：
    dict{条件词:[(目标词1,概率),(目标词2,概率),,,(目标词n,概率)]}
    :param filename:
    :return: dict
    """



def getPrefdict(filename):
    """
    获得前缀词典
    :param filename:
    :return:
    """
    lfreq={}
    ltotal=0
    with open(filename) as f:
        line=f.readline()
        while len(line)>0:

            word,freq=line.split()[0:2]
            freq=int(freq)
            lfreq[word]=freq
            ltotal+=freq
            for ch in range(len(word)):
                wfrag=word[:ch+1]
                if wfrag not in lfreq:
                    lfreq[wfrag]=0
            line=f.readline()
    return lfreq,ltotal



def getDAG(sentence,lfreq):
    """
    返回DAG list
    格式
    {k1:[i1,i2,,,in]，k2:[i1,i2,,,in]，，，kn:[i1,i2,,,in]}
    其中sen[k,i+1]表视为i
    :param sentence:
    :param lfreq:
    :return:
    """
    DAG={}
    N=len(sentence)
    for k in range(N):
        tmplist=[]
        i=k
        frag=sentence[k]
        while i < N and frag in lfreq:
            if lfreq[frag]>0:
                tmplist.append(i)
            i+=1
            frag=sentence[k:i+1]
        if not tmplist:
            tmplist.append(k)
        DAG[k]=tmplist
    return DAG

def getPYX(Y, X, dict):
    #TODO 平滑算法
    """
    返回条件概率。
    :param Y:
    句首 “HEAD”
    句尾 “TAIL”
    :param X:
    :param dict:
    :return:
    """
    for word,wfreq in dict[X]:
        if word==Y:
            return wfreq
    return 0



def bigramMatching(dict,sentence,DAG,ltotal):
    routeDAG={}
    N=len(sentence)

    logtotal=log(ltotal)
    routeDAG[N-1].append((N-1,"TAIL",1))
    for idx in range(N-2,-1,-1):
        print(idx)
        routeDAG[idx] = []
        for x in DAG[idx]:
            value,y=max((getPYX(sentence[idx:x+1],sentence[x+1:y+1])*v,y) for y, m,v in routeDAG[x + 1])

            routeDAG[idx].append((x,y,value))
        #route[idx]=max((log(lfreq[sentence[idx:x+1]] or 1)-logtotal+route[x+1][0],x) for x in DAG[idx])

    value,x=max(getPYX((sentence[0:x+1],"HEAD",dict)*v,x) for x,m,v in routeDAG[0])

    route=[]
    route.append(0)
    s=0
    next=x
    while True:
        route.append(s)
        for m,n,v in routeDAG[s]:
            if next==m:
                s=next+1
                next=n
                break
        if next=="TAIL":
            break



    return route,value
if __name__ == '__main__':
    #lfreq, ltotal=getPrefdict("doc/dic.txt")
    #DAG=getDAG("根据刘澜涛同志生前遗愿和家属的意见，刘澜涛同志的丧事从简，不举行仪式、不保留骨灰。",lfreq)
    #print(DAG)
    print(len("\n"))