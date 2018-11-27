from ME_external import *
def process(line):
    return None

def get_features(trainfile_path,featuresfile_path):
    """
    get feature.txt with trainfile
    :param trainfile_path:
    :param featuresfile_path:
    :return:
    """
    trainfile = open(trainfile_path, "r+", encoding="GB18030")
    featuresfile = open(featuresfile_path, "w", encoding="GB18030")
    while True:
        line = trainfile.readline()
        if line == '':
            break
        features=process(line)
        featuresfile.write(features+'\n')

def train_ME(featuresfile_path,ME_model_path):
    """
    get ME_model.txt with feature.txt
    :param featuresfile_path:
    :param ME_model_path:
    :return:
    """
    mxEnt = maxEntropy()
    # TODO
    mxEnt.loadData('data/gameLocation.dat')

    mxEnt.train()
    print(mxEnt.predict('Sunny'))



def getPredict(testfile_path,ME_model_path,test_feature_path,result_path):
    """
    get test_feature.txt result.txt with model.txt and testfile
    :param testfile_path:
    :param ME_model_path:
    :param test_feature_path:
    :param result_path:
    :return:
    """
    pass

def evaluate(standard_answer_path,result_path,evaluation_result_path):
    """
    get evaluation_result.txt with standard_answer and result.txt by SharedTaskEval.pl
    :param standard_answer_path:
    :param result_path:
    :param evaluation_result_path:
    :return:
    """

    pass
def read_features_file(featuresfile_path):
    return features

def read_ME_model(ME_model_path):
    return ME_model