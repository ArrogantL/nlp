if __name__ == '__main__':
    f=open("dsad.txt","w")
    for i in range(10):
        f.write("%d %d"%(i,i+1))