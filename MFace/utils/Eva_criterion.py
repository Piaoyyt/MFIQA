def get_TP(predict_result:list, thresh):
    result = [1 if similarity >= thresh and label==1 else 0 for similarity, label in predict_result]
    return sum(result)
def get_TN(predict_result:list, thresh):
    result = [1 if similarity <= thresh and label == 0 else 0 for similarity, label in predict_result]
    return sum(result)
def get_FP(predict_result:list, thresh):
    result = [1 if similarity > thresh and label == 0 else 0 for similarity, label in predict_result]
    return sum(result)
def get_FN(predict_result:list, thresh):
    result = [1 if similarity < thresh and label == 1 else 0 for similarity, label in predict_result]
    return sum(result)
def get_PositiveNums(predict_result:list):
    result = [1 if label == 1 else 0 for similarity, label in predict_result]
    return sum(result)
def get_NegativeNums(predict_result:list):
    result = [1 if label == 0 else 0 for similarity, label in predict_result]
    return sum(result)
def get_FAR(predict_result:list, thresh)-> float:
    '''
    :param predict_result: 预测的结果 （相似度，标签<0或者1>）
    :param thresh: 识别设置的阈值
    :return: 误识率，即negative对里面被错误识别成positive的比率
    '''
    false_accept = get_FP(predict_result, thresh)
    negative_nums = get_NegativeNums(predict_result)
    return false_accept * 1.0 / negative_nums
def get_FRR(predict_result:list, thresh)->float:
    '''
    :param predict_result: 预测的结果 （相似度，标签<0或者1>）
    :param thresh: 识别设置的阈值
    :return: 误拒率，即positive对里面被错误识别成negative的比率
    '''
    false_reject =get_FN(predict_result, thresh)
    positive_nums = get_PositiveNums(predict_result)
    return  false_reject * 1.0 / positive_nums
def get_TAR(predict_result:list, thresh)->float:
    '''
    :param predict_result: 预测的结果 （相似度，标签<0或者1>）
    :param thresh: 识别设置的阈值
    :return: 误拒率，即positive对里面被正确识别的比率
    '''
    true_accept =get_TP(predict_result, thresh)
    positive_nums = get_PositiveNums(predict_result)
    return  true_accept * 1.0 / positive_nums
def get_TRR(predict_result:list, thresh)->float:
    '''
    :param predict_result: 预测的结果 （相似度，标签<0或者1>）
    :param thresh: 识别设置的阈值
    :return: 误拒率，即negative对里面被正确识别的比率
    '''
    true_reject =get_TN(predict_result, thresh)
    negative_nums = get_NegativeNums(predict_result)
    return  true_reject * 1.0 / negative_nums
def get_Acc(predict_result:list, thresh)->float:
    '''
    :param predict_result:
    :param thresh:
    :return: 人脸识别的准确率，即预测结果正确的比例
    '''
    return (get_TP(predict_result, thresh) + get_TN(predict_result, thresh)) *\
        1.0 / (len(predict_result))
def get_Precision(predict_result:list, thresh)->float:
    '''
     :param predict_result: 预测的结果 （相似度，标签<0或者1>）
    :param thresh: 识别设置的阈值
    :return: 精确度=(TP)/(TP + FP)即预测为positive里面实际真的为positive的比例
    '''
    return get_TP(predict_result, thresh) * 1.0 / \
           (get_TP(predict_result, thresh) + get_FP(predict_result, thresh)+1e-7)
def get_Recall(predict_result:list, thresh)->float:
    '''
    :param predict_result: 预测的结果 （相似度，标签<0或者1>）
    :param thresh: 识别设置的阈值
    :return: 识别的召回率=(TP)/(TP+FN)，即真实值为positive里被实际预测正确的比例
    '''
    return get_TP(predict_result, thresh) * 1.0 / \
           (get_TP(predict_result, thresh) + get_FN(predict_result, thresh) + 1e-7)
def get_FScore(predict_result:list, thresh)->float:
    '''
   :param predict_result: 预测的结果 （相似度，标签<0或者1>）
    :param thresh: 识别设置的阈值
    :return: F-Score = Precision 和 Recall的加权调和平均
    '''
    return 2 * (get_Precision(predict_result, thresh) * get_Recall(predict_result, thresh))/\
           (get_Precision(predict_result, thresh) + get_Recall(predict_result, thresh) + 1e-7)
