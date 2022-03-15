import argparse
import logging
import os,sys
project_path = "\\".join(os.getcwd().split("\\")[:-2]) if not sys.platform=="linux"\
    else "/".join(os.getcwd().split("/")[:-2])
sys.path.append(project_path)
import torch
from torch.utils.data import DataLoader

from MFace.face_dataset_utils.Dataset import Face_Dataset
from MFace.face_dataset_utils.Generate_face_information import generate_face_id,generate_face_quality,generate_face_images_paths
from MFace.net_struture.net_sphere import AngleLoss
from MFace.net_struture.net_mface import MFaceLoss
from MFace.Conf.config import cfg

logger = logging.getLogger("globals")
def parse_args():
    parser = argparse.ArgumentParser(description='Train face recognition network')
    # general
    parser.add_argument('--method', default=cfg.Test.method, help='the method of traing model')
    parser.add_argument('--fiqa_model', default=cfg.Test.fiqa, help='the Face Image Quality Model')
    #parser.add_argument('--root', default=cfg.Test.root_path, help='the training root path')
    parser.add_argument('--test_model',default=cfg.Test.model_path, help='the model to be tested')
    parser.add_argument('--batch_size', type=int, default=cfg.Test.batch_size, help='batch size in each context')
    #parser.add_argument('--root', default=cfg.Train.root_path, help='the training root path')

    args = parser.parse_args()
    return args
def set_logger(args):
    #today = time.strftime("%Y_%m_%d", time.localtime(time.time()))
    global today
    train_details = args.test_model.split("\\")[-2]
    model_name = args.test_model.split("\\")[-1]
    logger_path = f"..\\logging_test\\{args.method}\\{train_details}\\{model_name}"
    logger_path = logger_path.replace("\\", "/") if sys.platform=="linux" else logger_path
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)

    fh = logging.FileHandler(os.path.join(logger_path,"Test_details.log"))
    formats = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formats)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)
def get_dataset_infotmation(args):
    imgs_root_path = getattr(cfg.Eval.imgs_root, cfg.Test.dataset)
    if sys.platform == "linux":
        imgs_root_path = imgs_root_path.replace("\\", "/")
    #print(imgs_root_path)
    faces_id = generate_face_id(imgs_root_path)
    faces_imgs_paths = generate_face_images_paths(imgs_root_path)
    #print(faces_imgs_paths[0])
    faces_quality_score = dict(generate_face_quality(imgs_root_path, cfg.Test.dataset,\
                            faces_imgs_paths, store_path=cfg.fiqa_score_path,\
                                                     fiqa_model=cfg.Train.fiqa))
    return faces_imgs_paths, faces_id, faces_quality_score
def get_test(faces_imgs_paths):
    faces_number = len(faces_imgs_paths)
    test_index = [i for i in range(faces_number) if i%5==0]
    train_index = [i for i in range(faces_number)if i not in test_index ]
    train_imgs_paths = [faces_imgs_paths[i] for i in train_index]
    test_imgs_paths = [faces_imgs_paths[i] for i in test_index]
    return train_imgs_paths, test_imgs_paths
def class_accuracy(predict,target):
    '''
    :param predict: the predict tensor of the trained model
    :param target:  the true label of the face image
    :return: classification accuracy of the model
    '''

    predict_label = torch.argmax(predict, dim=1)
    predict_label = torch.squeeze(predict_label).int()
    target =torch.squeeze(target).int()
    result = predict_label.eq_(target)
    result = torch.squeeze(result)
    accuracy = torch.sum(result)/result.shape[0]
    return accuracy

def main(args):
    # *-----------------------数据准备-----------------------*
    logger.info("\n" + "-" * 8 + "开始加载训练人脸的数据" + "-" * 8)
    faces_imgs_paths, faces_id, faces_quality_score = get_dataset_infotmation(args=args)
    train_imgs_paths, test_imgs_paths = get_test(faces_imgs_paths)

    test_dataset = Face_Dataset(getattr(cfg.Eval.imgs_root, cfg.Test.dataset), test_imgs_paths, faces_id, faces_quality_score, seed=1)
    train_dataset = Face_Dataset(getattr(cfg.Eval.imgs_root, cfg.Test.dataset), train_imgs_paths, faces_id, faces_quality_score, seed=1)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
# *-----------------------加载模型-----------------------*
    model_path = args.test_model.replace("\\", "/") if sys.platform=="linux" else args.test_model
    model =  torch.load(model_path).eval().cuda()
    model_name = args.test_model.split("\\")[-2]+"_"+args.test_model.split("\\")[-1].split(".")[0]
    accuracy = 0
    batch_number = 0
    train_loss = 0
    test_loss = 0
    if args.method =="MFace":
        loss = MFaceLoss()
    elif args.method == "Sphereface":
        loss = AngleLoss()
    loss = AngleLoss()
    for index,batch in enumerate(test_loader):
        img_batch  = batch["face_img"]
        label_batch = batch["face_id"]
        data_score = batch["face_score"]
        with torch.no_grad():
            model_predict = model(img_batch.cuda())
        if args.method == "MFace":
            batch_loss=loss(model_predict, label_batch.cuda(), data_score.cuda())
        elif args.method == "Sphereface":
            batch_loss = loss(model_predict, label_batch.cuda())
        Loss = loss(model_predict,label_batch.cuda())
        test_loss += Loss
        model_predict = model_predict[0] if cfg.Train.loss == "Sphereface" else model_predict
        batch_accuracy =  class_accuracy(model_predict,label_batch.cuda())
        accuracy +=batch_accuracy
        batch_number+=1
    accuracy = accuracy/batch_number
    test_total_loss = test_loss /batch_number
    logger.info(f"\n {model_name} 在测试集上的分类准确率为{accuracy}, 平均损失{test_total_loss}\n")
    accuracy = 0
    batch_number = 0
    for index,batch in enumerate(train_loader):
        img_batch  = batch["face_img"]
        label_batch = batch["face_id"]
        data_score = batch["face_score"]
        with torch.no_grad():
            model_predict = model(img_batch.cuda())
        if args.method == "MFace":
            batch_loss=loss(model_predict, label_batch.cuda(), data_score.cuda())
        elif args.method == "Sphereface":
            batch_loss = loss(model_predict, label_batch.cuda())
        Loss = loss(model_predict,label_batch.cuda())
        train_loss += Loss
        model_predict = model_predict[0] if cfg.Train.loss == "Sphereface" else model_predict
        batch_accuracy =  class_accuracy(model_predict,label_batch.cuda())

        accuracy +=batch_accuracy
        batch_number+=1
    accuracy = accuracy/batch_number
    train_total_loss = train_loss /batch_number
    logger.info(f"\n {model_name} 在训练集上的分类准确率为{accuracy}, 平均损失{train_total_loss}")
if __name__ =="__main__":
    args = parse_args()
    set_logger(args)
    main(args)
