import os
import subprocess
import CRFPP


class CRFModel(object):
    def __init__(self, model='model_name'):
        """
        函数说明: 类初始化
        :param model: 模型名称
        """
        self.model = model

    def add_tagger(self, tag_data_file):
        """
        函数说明: 添加语料
        :param tag_data: 数据
        :return:
        """
        tag_data_file=open(tag_data_file,'r')
        if not os.path.exists(self.model):
            print('模型不存在,请确认模型路径是否正确!')
            exit()
        tagger = CRFPP.Tagger("-m {} -v 3 -n2".format(self.model))
        tagger.clear()
        while True:
            line=tag_data_file.readline()
            if line =='':
                break
            tagger.add(line.strip())
        tagger.parse()
        return tagger

    def crf_test(self, tag_data, separator='_'):
        """
        函数说明: crf测试
        :param tag_data:
        :param separator:
        :return:
        """
        tagger = self.add_tagger(tag_data)
        size = tagger.size()
        data=[]
        for i in range(0, size):
            word, tag = tagger.x(i, 0), tagger.y2(i)
            data.append((word,tag))
        return data

    def crf_learn(self, filename,template_path,f,c,e,p):
        """
        函数说明: 训练模型
        :param filename: 已标注数据源
        :return:
        """
        crf_bash = "crf_learn -f {} -c {} -e {} -p {} {} {} {}".format(f,c,e,p,template_path,filename, self.model)
        process = subprocess.Popen(crf_bash.split(), stdout=subprocess.PIPE)
        output = process.communicate()[0]
        print(output.decode(encoding='utf-8'))

# os.system("crf_learn -c 4.0 -p 8 " + template_path + " " + featuresfile_path + " " + ME_model_path)
def crf_learn(template_path,featuresfile_path,ME_model_path,f=3,c=4.0,e=0.0001,p=1):
    crf_model=CRFModel(model=ME_model_path)
    crf_model.crf_learn(featuresfile_path,template_path,f,c,e,p)
# os.system("crf_test -m " + ME_model_path + " " + test_feature_path + " >> tempresult.txt")
def crf_test(ME_model_path,test_feature_path,result_path):
    crf_model = CRFModel(model=ME_model_path)
    data=crf_model.crf_test(test_feature_path)
    result_file=open(result_path,"w")
    test_feature_file=open(test_feature_path,'r')
    for word,label in data:
        line=test_feature_file.readline()
        if line =='\n':
            result_file.write('\n')
            line = test_feature_file.readline()
        assert len(word)!=0
        result_file.write(word+'\t'+label+'\n')
    result_file.close()
    test_feature_file.close()


