import os,sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'
import torch
# print(torch.cuda.device_count())
# torch.cuda.set_device(0)
# print(torch.cuda.is_available())
# print(torch.cuda.current_device())
project_path = "\\".join(os.getcwd().split("\\")[:-1]) if not sys.platform=="linux"\
    else "/".join(os.getcwd().split("/")[:-1])
print(project_path)
sys.path.append(project_path)
import random
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

import logging
#from sklearn.metrics import roc_curve, auc
from pyIUA.utils.correlation import  correlation_analysis
from pyIUA.dataset.database import ImgDatabase, get_image_path, filter_base_image, shuffle, get_all_images
from pyIUA.core.config import  cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger =logging.getLogger('global')
parser = argparse.ArgumentParser(description="Image Utility Assessment")
parser.add_argument('--test_annotation_path', type=str, default=cfg.Dataset.test_annotation_path,
                    help='the testing face image scores annotation path')
parser.add_argument('--annotation_method',type=str,default=cfg.Dataset.annotation_method,
                    help="the Face recognition method,seleted from[deep_face_score,]")
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--config',type=str, default=cfg.Config_path,
                    help="the config file path")
args = parser.parse_args()

def build_test_data():
    logging.info("building the test dataset")
    root_path = cfg.Dataset.root_path  # 图像的路径
    #train_annotation = args.train_annotation_path  # 图像标注的路径
    test_annotation = args.test_annotation_path
    Annotation_method = args.annotation_method
    # landmark_path =args.landmark_path
    # train_paths = get_all_images(os.path.join(root_path, "Train"))
    test_paths = get_all_images(os.path.join(root_path, "Test"))
    test_shuffle = shuffle(test_paths)[0:200]
    test_database = ImgDatabase(img_paths=test_shuffle, annotation_path=test_annotation,
                                annotation_method=Annotation_method)
    # test_database.filter(choice="2", thresh=cfg.Dataset.thresh)
    return  test_database
def model_predict(predict_model,database, batch, batch_number):
    prior_feature = None
    predict_result = []
    test_number = len(database)
    for i in range(batch_number):
        if i == batch_number - 1:
            # index = test_index[i * batch: test_number]
            index = [j for j in range(i * batch,test_number)]
            test_batch, test_batch_mask, test_batch_brisque, _ = database.getbatch(indices=index)
        else:
            index = [j for j in range(i * batch, (i + 1) * batch)]
            test_batch, test_batch_mask, test_batch_brisque, _ = database.getbatch(indices=index)
        with torch.no_grad():
            if cfg.TRAIN.prior =="brisque":
                prior_feature = database.get_brisque_batch(index)
            output = np.squeeze(predict_model(test_batch.cuda(),test_batch_mask.cuda()).detach().cpu().numpy())
                                              # feature_layer=cfg.TRAIN.layer,
                                              # label_type=cfg.Dataset.label_type, prior=cfg.TRAIN.prior,
                                              # quality_feature=prior_feature.cuda() if prior_feature!=None else None
                                              # ).detach().cpu().numpy())
            predict_result = predict_result + list(output)
    all_index = [i for i in range(0,test_number)]
    _,_,_,test_label = database.getbatch(all_index)
    test_label = np.squeeze(test_label.numpy())
    output = np.squeeze(predict_result)
    return test_label,output
def main():
    test_database = build_test_data()
    # layer = 15 # 设置选取的网络结构，layer=[1-15,"final","low_high_1","low_integrated","high_integrated"]
    model_path = cfg.TEST.Model_path
    predict_model = torch.load(model_path).cuda().eval()
    test_number = len(test_database)
    print(f"测试集图:{test_number}")
    batch = cfg.TEST.Batch_size
    batch_time = test_number // batch  # 向下取整
    test_database.shuffle()
    #test_label, predict_output = model_predict(predict_model=predict_model,database=test_database,batch=batch,batch_number=batch_time)
    # print(predict_output.shape)
    # print(test_label.shape)
    predict_thresh = [i for i in range(30,90,10)]
    # number_count = [0]*len(predict_thresh)
    # for index,thresh in enumerate(predict_thresh):
    #     for score in predict_result:
    #         if score>thresh:
    #             number_count[index]+=1
    # for index, count in enumerate(number_count):
    #     print(f"Predicting score above {predict_thresh[index]}:{count}")
    test_index =[i for i in range(len(test_database))]
    random.shuffle(test_index)
    test_images_num = 10
    row = 2
    img_tensor= torch.zeros(1, 3, 112, 112)
    score = predict_model(img_tensor.cuda())
    print(score)
    colums = int(test_images_num / row)
    # plt.subplots(figsize=(640, 480))
    # for i in range(row):
    #     for j in range(colums):
    #         data = test_database.__getitem__(test_index[i * 5 +j])
    #         print(test_database.img_paths[test_index[i * 5 + j]])
    #         image = cv2.imread(test_database.img_paths[test_index[i * 5 + j]])
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         predict = round(predict_model(data["input"].view(-1,3,112,112).cuda(),\
    #                                       data["input_mask"].view(-1,3,112,112).cuda()).item(),\
    #                                      3)
    #         ground_truth = data["quality_label"]
    #         # print(groud_truth)
    #         plt.subplot(row, colums, i * 5 + j + 1)
    #         plt.imshow(image)
    #         plt.title("GT: {0},PS:{1}".format(ground_truth, predict),fontdict={'size':8})
    # #pic_path = "\\".join(model_path.split("\\")[:-1])+"\\"+model_path.split("\\")[-1].split(".")[0]+"_result.jpg"
    # pic_path = "./model_2_result.jpg"
    # plt.savefig(pic_path, dpi=200)
    # plt.show()
    #
    # # # print("True score:", test_lable)
    # # # print("Predicted score:", output)
    # # binary_test=[1 if i>=78 else 0 for i in test_label]
    # # binary_predict=[i/100 for i in output]
    # # fpr, tpr, thresholds = roc_curve(binary_test, binary_predict);
    # # roc_auc = auc(fpr, tpr)
    # # frr = 1-tpr
    # # ##确定最佳阈值
    # #
    # # right_index = (tpr + (1 - fpr) - 1)
    # # yuzhi = max(right_index)
    # # index = right_index.tolist().index(max(right_index))
    # # tpr_val = tpr[index]
    # # fpr_val = fpr[index]
    # # ## 绘制roc曲线图
    # # plt.subplots(figsize=(7,5.5))
    # # plt.plot(fpr, tpr, color='darkorange',
    # #          lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # # plt.xlim([0.0, 1.0])
    # # plt.ylim([0.0, 1.05])
    # # plt.xlabel('False Positive Rate')
    # # plt.ylabel('True Positive Rate')
    # # plt.title('ROC Curve')
    # # plt.legend(loc="lower right")
    # # plt.show()
    # # plt.subplots(figsize=(7,5.5))
    # # plt.plot(fpr, frr, color='darkorange',
    # #          lw=2, label='DET curves ')
    # # # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # # plt.xlim([0.0, 1.0])
    # # plt.ylim([0.0, 1.05])
    # # plt.xlabel('False Acceptance Rate')
    # # plt.ylabel('False Rejection Rate')
    # # plt.title('DET Curve')
    # # plt.legend(loc="lower right")
    # # plt.show()
    # correlation_analysis(test_label,predict_output)

if __name__ =="__main__":
    main()
