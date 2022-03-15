import os,sys

project_path = "\\".join(os.getcwd().split("\\")[:-1]) if not sys.platform=="linux"\
    else "/".join(os.getcwd().split("/")[:-1])

sys.path.append(project_path)
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform=="linux":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import logging
import argparse, random
import time
import torch.nn as nn
import torch
from pyIUA.models.Model_train_attribute_single_aadb import Associate_Style_Model,StyleModel
from pyIUA.models.Testt_model import Model_test
from tqdm import tqdm
import numpy as np
from torch import optim
from pyIUA.dataset.database import ImgDatabase, shuffle, get_all_images
from pyIUA.models.backbone.net_structure import *
from pyIUA.models.loss import batch_sum_loss, ContrastiveLoss, rank_loss
from pyIUA.models.Mask_model import ModelBuilder
from pyIUA.models.backbone.Quality_regre_net import Quality_net
from pyIUA.core.config import cfg
from torch.utils.data import DataLoader

from torchvision import models
logger = logging.getLogger('global')
today = time.strftime("%Y_%m_%d_%H", time.localtime(time.time()))
parser = argparse.ArgumentParser(description="Image Utility Assessment")
parser.add_argument('--train_annotation_path', type=str, default=cfg.Dataset.train_annotation_path,
                    help='the training face image scores annotation path')
parser.add_argument('--test_annotation_path', type=str, default=cfg.Dataset.test_annotation_path,
                    help='the testing face image scores annotation path')
parser.add_argument('--annotation_method', type=str, default=cfg.Dataset.annotation_method,
                    help="the Face recognition method,seleted from[deep_face_score,]")
parser.add_argument('--landmark_path', type=str, default=cfg.Dataset.landmark_path,
                    help="the landmark path of the face images")
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--config', type=str, default=cfg.Config_path,
                    help="the config file path")
args = parser.parse_args()


def print_grad(grad):
    grad_list.append(grad)


def seed_torch(seed=0):
    #random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def set_logger():
    global today
    file_path = cfg.TRAIN.store_root + f"\\{today}_{cfg.TRAIN.dataset}"
    if sys.platform=="linux":
        file_path = file_path.replace("\\","/")
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    fh_path = os.path.join(file_path,"Train_details.log")
    fh = logging.FileHandler(fh_path)
    formats = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formats)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)


def build_data():
    logger.info("build dataset")
    root_path = cfg.Dataset.root_path  # 图像的路径
    train_annotation = args.train_annotation_path  # 图像标注的路径
    test_annotation = args.test_annotation_path
    Annotation_method = args.annotation_method

    train_paths = get_all_images(os.path.join(root_path, "Train"))
    test_paths = get_all_images(os.path.join(root_path, "Test"))

    print("初始训练集最后一张图片名称：", train_paths[-1])
    # print("初始验证集第一张图片名称：", valid_paths[0])
    print("初始测试集第一张图片名称：", test_paths[0])
    train_paths = train_paths # + test_base_paths

    train_shuffle = shuffle(train_paths[:-1])
    # # valid_shuffle = shuffle(valid_paths)
    test_shuffle = shuffle(test_paths[:5000])
    print("训练集图片总数：", len(train_shuffle))
    train_database = ImgDatabase(img_paths=train_shuffle, annotation_path=train_annotation,
                                 annotation_method=Annotation_method)

    test_database = ImgDatabase(img_paths=test_shuffle, annotation_path=test_annotation,
                                annotation_method=Annotation_method)

    dataset = [train_database, test_database]
    return dataset


def train(dataset):
    torch.multiprocessing.set_start_method('spawn')
    logger.setLevel(logging.DEBUG)
    train_database, test_database = dataset[:]
    if sys.platform=="linux":
        train_dataloader = DataLoader(train_database, batch_size=cfg.TRAIN.batch_size, shuffle=False, pin_memory=True, num_workers=8)
        test_dataloader = DataLoader(test_database, batch_size=cfg.TRAIN.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    else:
        train_dataloader = DataLoader(train_database, batch_size=cfg.TRAIN.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_database, batch_size=cfg.TRAIN.batch_size, shuffle=False)
    logger.info("building the model......")
    if cfg.TRAIN.pretrained_model == None:
        #train_model = Model_test().train()
        # model_ft = models.resnet34(pretrained=True)  # 主干预训练模型
        # model_ft.aux_logits = False
        # num_ftrs = model_ft.fc.out_features
        # net2 = StyleModel(1, 0.5, num_ftrs)  # 美学模型
        # train_model = Associate_Style_Model(resnet=model_ft, style_net=net2)
        #train_model = torch.load(project_path + "\\Evaluation\\FIQA_model\\models\\model_39.ckpt").cuda()
        train_model = ModelBuilder().train()
        #train_model.show_model_details()
        #train_model = Quality_net().train()
    else:
        train_model = torch.load(cfg.TRAIN.pretrained_model).train()
       # train_model.show_model_details()
    device_ids = [0, 1]
    if sys.platform=="linux":
        train_model = nn.DataParallel(train_model, device_ids=device_ids)
    if torch.cuda.is_available():
        train_model = train_model.cuda()
    # for chidren in train_model.backbone.children():
    #     for param in chidren.parameters():
    #         param.requires_grad = False
    # 定义损失函数和优化器
    loss_functiono_list = {"binary_cross_entropy": nn.BCELoss, \
                           "mean_square_loss": nn.MSELoss, \
                           "contrast_loss": ContrastiveLoss,\
                           "huber_loss":nn.SmoothL1Loss}
    loss_parameters ={"beta":0.2} if cfg.TRAIN.loss_function=="huber_loss" else {}
    criterion = loss_functiono_list[cfg.TRAIN.loss_function](**loss_parameters).cuda()
    if cfg.TRAIN.optim_name == 'Adam':
        optimizer = optim.Adam(train_model.parameters(), lr=cfg.TRAIN.learning_rate, \
                               betas=(0.9, 0.99),
                               eps=1e-06,
                               weight_decay=cfg.TRAIN.weight_decay)
    elif cfg.TRAIN.optim_name == 'SGD':
        optimizer = optim.SGD(train_model.parameters(), lr=cfg.TRAIN.learning_rate,
                              weight_decay=cfg.TRAIN.weight_decay)
    if sys.platform=="linux":
        optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
        optimizer = optimizer.module
    # optimizer = optim.FRSGD(train_model.parameters(), lr=0.1)
    train_number = len(train_database)
    # valid_number = len(valid_database)
    test_number = len(test_database)
    logger.info(f"train number is {train_number}\\nvalid number is {test_number}\\ntest number is {test_number}")
    # print(f"train number is {train_number}\\nvalid number is {valid_number}\\ntest number is {test_number}")
    batch_num = int(train_number / cfg.TRAIN.batch_size)
    batch_test = cfg.TEST.Batch_size
    #
    final_size = train_number % cfg.TRAIN.batch_size
    lr_list = []
    train_loss_list = []
    train_mse_loss_list = []
    train_rank_loss_list = []
    valid_loss_list = []
    valid_mse_loss_list = []
    valid_rank_loss_list = []
    prior_feature = None  # range(batch_num + 1)
    logger.info("The training beginning")
    # x_batch = torch.randn([2, 3, 160, 160])
    # y_batch = torch.randn([2,1])
    record_path  =cfg.TRAIN.store_root + f"\\{today}_{cfg.TRAIN.dataset}\\record_file"
    model_file_path = cfg.TRAIN.store_root + f"\\{today}_{cfg.TRAIN.dataset}\\model_file"
    if sys.platform=="linux":
        record_path = record_path.replace("\\", "/")
        print(record_path)
        model_file_path = model_file_path.replace("\\", "/")
    for i in tqdm(range(cfg.TRAIN.iter_times),ncols=80):
        if i == 0:
            if not os.path.exists(record_path):
                os.mkdir(record_path)
            with open(
                    "{0}".format( os.path.join(record_path, "detail.txt")),
                    'w') as f:
                f.write("initial_learning_rate: {} \n".format(cfg.TRAIN.learning_rate) + "decay_steps: {}\n".format(
                    cfg.TRAIN.decay_steps) + "decay_rate: {}\n".format(cfg.TRAIN.decay_rate) +
                        "iter_times: {}\n".format(cfg.TRAIN.iter_times) + \
                        "batch_size: {}\n".format(cfg.TRAIN.batch_size) +
                        "pretrained_model: {}\n".format(cfg.TRAIN.pretrained_model) + \
                        "Optimization: {}\n".format(cfg.TRAIN.optim_name) +
                        "Pre_model_drop_out_set:{0} full_drop_out:{1}\n".format(cfg.TRAIN.pre_dropout,
                                                                                cfg.TRAIN.full_dropout) +
                        "loss function: {}\n".format(cfg.TRAIN.loss_function) + \
                        "mse_lambda:{}\n".format(cfg.TRAIN.mse_lambda) +
                        "rank_lambda:{}\n".format(cfg.TRAIN.rank_lambda) + \
                        "label_type:{}\n".format(cfg.Dataset.label_type))
        for batch in tqdm(train_dataloader,ncols=80):#range(batch_num)
            train_model.train()
            # if batch == batch_num-1 and final_size != 0:
            #     index = [i for i in range(batch * cfg.TRAIN.batch_size , train_number)]
            #     x_batch, x_mask_batch,x_batch_brisque, y_batch = train_database.getbatch(index,label_type=cfg.Dataset.label_type)
            #
            # elif batch == batch_num and final_size == 0:
            #     continue
            # else:
            #     index = [i for i in range(batch * cfg.TRAIN.batch_size, (batch + 1) * cfg.TRAIN.batch_size)]
            #     x_batch,x_mask_batch,x_batch_brisque, y_batch = train_database.getbatch(index,label_type=cfg.Dataset.label_type)
            x_batch = batch["input"]
            x_batch_brisque = None#batch["brisque"]
            x_batch_mask = batch["input_mask"]
            y_batch = batch["quality_label"].float()

            # if cfg.TRAIN.prior =="brisque":
            #     prior_feature = train_database.get_brisque_batch(index)
            # print(cfg.prior_feature.shape)

            optimizer.zero_grad()
            output = train_model(x_batch.cuda(), x_batch_mask.cuda()).view(-1)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!, x_batch_mask, x_batch_brisque
            #loss = (criterion(output, y_batch.cuda()) ** 0.5).float()
            #print(loss)
            # feature_layer=cfg.TRAIN.layer,
            # label_type=cfg.Dataset.label_type, prior=cfg.TRAIN.prior,
            # quality_feature =prior_feature.cuda() if prior_feature!=None else None)
            # output1.requires_grad_()
            # train_model.tailnet[0].weight.retain_grad()
            # print(train_model.tailnet[0].weight.is_leaf)
            # target = torch.reshape(y_batch.cuda(), [-1, 1])
            if cfg.Dataset.label_type == "continuous":
                #mse_mine =  torch.sum(torch.sqrt(output-y_batch.cuda()))/y_batch.shape[0]
                #mse = criterion(output,y_batch.cuda())
                mse_Loss = criterion(output,y_batch.cuda()) ** 0.5
                rank_Loss = rank_loss(output, y_batch.cuda())
                loss = cfg.TRAIN.mse_lambda * mse_Loss + cfg.TRAIN.rank_lambda * rank_Loss
            # train_mse_loss, train_loss = \
            #     batch_sum_loss(train_database, model=train_model.eval(), data_num=6,
            #                    batch=batch_test, layer_name=cfg.TRAIN.layer,
            #                    label_type=cfg.Dataset.label_type,
            #                    mse_lambda=cfg.TRAIN.mse_lambda,
            #                    rank_lambda=cfg.TRAIN.rank_lambda)
            # print(loss)
            # print(rank_Loss)
            # else:
            #     loss = criterion(output1, y_batch.cuda())

            # loss.requires_grad_()
            # train_model.tailnet[0].weight.register_hook(print_grad)
            loss.backward()
            # grad = [x.grad for x in optimizer.param_groups[0]["params"]]

            #            print(train_model.tailnet[0].weight.grad)
            optimizer.step()

        train_mse_loss, train_loss = \
            batch_sum_loss(train_database, model=train_model.eval(), data_num=2000,
                           batch=batch_test, layer_name=cfg.TRAIN.layer,
                           label_type=cfg.Dataset.label_type,
                           mse_lambda=cfg.TRAIN.mse_lambda,
                           rank_lambda=cfg.TRAIN.rank_lambda)
        logger.info(f"epoch{i}: the train loss---->  {train_loss}")
        if i % cfg.TRAIN.decay_steps == 0:
            for p in optimizer.param_groups:
                p['lr'] *= cfg.TRAIN.decay_rate
            lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
            print(" learning_rate of iter_times {}:".format(i), optimizer.state_dict()['param_groups'][0]['lr'], "\n")

        logger.info(f"epoch{i}: the learning rate---->{optimizer.state_dict()['param_groups'][0]['lr']}")
        if i % cfg.TRAIN.print_loss_gap == 0:

            valid_mse_loss, valid_loss = \
                batch_sum_loss(test_database, model=train_model.eval(), data_num=2000,
                               batch=batch_test, layer_name=cfg.TRAIN.layer,
                               mse_lambda=cfg.TRAIN.mse_lambda,
                               rank_lambda=cfg.TRAIN.rank_lambda,
                               label_type=cfg.Dataset.label_type)
            train_model.train()
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            if train_mse_loss != None:
                train_mse_loss_list.append(train_mse_loss)

                valid_mse_loss_list.append(valid_mse_loss)

                print(
                    "train_loss of iter times {0} is:MSE-{1} RANK-{2} Total-{3}".format(i, train_mse_loss, 0,
                                                                                        train_loss))
                print(" valid_loss of iter times {0} is:{1} RANK-{2} Total-{3}\\n".format(i, valid_mse_loss, 0,
                                                                                         valid_loss))
                logger.info(f"epoch{i}: the valid loss---->{valid_loss}")
            else:
                print(f"Iteration [{i}] the train loss is {train_loss}")
                print(f"Iteration [{i}] the valid loss is {valid_loss}")
        if i % cfg.TRAIN.ckpt_store_gap == 0  or i == cfg.TRAIN.iter_times - 1:
            if os.path.exists(model_file_path) is False:
                os.mkdir(model_file_path)
            torch.save(train_model, "{0}".format(
                os.path.join(model_file_path, "model_{0}.ckpt".format(i))))
        if i == cfg.TRAIN.iter_times - 1:
            if not os.path.exists(record_path):
                os.mkdir(record_path)
            with open("{0}".format(
                    os.path.join(record_path, "train_loss.txt")),
                      'w') as f:
                for each in train_loss_list:
                    f.write(str(each)+"\n")

            with open("{0}".format(
                    os.path.join(record_path, "train_mse_loss.txt")),
                      'w') as f:
                for each in train_mse_loss_list:
                    f.write(str(each)+"\n")

            with open("{0}".format(
                    os.path.join(record_path, "train_rank_loss.txt")),
                      'w') as f:
                for each in train_rank_loss_list:
                    f.write(str(each)+"\n")

            with open("{0}".format(
                    os.path.join(record_path, "valid_loss.txt")),
                      'w') as f:
                for each in valid_loss_list:
                    f.write(str(each)+"\n")

            with open("{0}".format(
                    os.path.join(record_path,  "valid_mse_loss.txt")),
                      'w') as f:
                for each in valid_mse_loss_list:
                    f.write(str(each)+"\n")

            with open("{0}".format(
                    os.path.join(record_path,  "valid_rank_loss.txt")),
                      'w') as f:
                for each in valid_rank_loss_list:
                    f.write(str(each)+"\n")
            with open("{0}".format(
                    os.path.join(record_path,  "learning_rate.txt")),
                      'w') as f:
                for each in lr_list:
                    f.write(str(each)+"\n")



def main():
    logger.info("export the config file......")
    cfg.merge_from_file(args.config)
    logger.info("exporting successfully")
    dataset = build_data()
    train(dataset)


if __name__ == "__main__":
    seed_torch(args.seed)
    set_logger()
    main()
