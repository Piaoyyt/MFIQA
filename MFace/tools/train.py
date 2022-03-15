import sys, os

project_path = "\\".join(os.getcwd().split("\\")[:-2]) if not sys.platform == "linux" \
    else "/".join(os.getcwd().split("/")[:-2])
sys.path.append(project_path)
if sys.platform == "linux":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch.optim, time, os, warnings
from tqdm import tqdm

# 日志记录以及解析模块
import argparse, logging
# 数据加载模块
from MFace.face_dataset_utils.Dataset import Face_Dataset
from MFace.face_dataset_utils.Generate_face_information import generate_face_id, generate_face_quality, \
    generate_face_images_paths
from torch.utils.data import DataLoader

# 训练参数配置加载模块
from MFace.Conf.config import cfg
# 网络模型加载模块
# from MFace.net_struture.net_whole import sphere_MFIQA, sphere_MFIQA_v2
from MFace.net_struture.net_sphere import sphere20a, AngleLoss, AngleLinear
# from MFace.net_struture.backbone.resnet import *
# from MFace.net_struture.backbone.ir_resnet_se import Backbone
# from MFace.net_struture.backbone.sphere import sphere20a
# from MFace.net_struture.metricbone.metrics_bone import *
from MFace.net_struture.loss_function import *
from MFace.net_struture.backbone.Backbone_Load import load_backbone
from MFace.net_struture.metricbone.Metrics_Load import load_metricsbone
from MFace.utils.Preprocess import Preprocess
from MFace.utils.Eva_criterion import *

# MFIQA损失加载
# from MFace.net_struture.net_mface import MFaceLoss
# from MFace.net_struture.net_sphere import AngleLoss
logger = logging.getLogger('globals')
today = time.strftime("%Y_%m_%d_%H", time.localtime(time.time()))
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Train face recognition network')
    parser.add_argument('--method', default=cfg.Train.method, help='the method of traing model')
    # general
    parser.add_argument('--pretrained', default=cfg.Train.pretrained, help='the pretrained model')
    # parser.add_argument('--root', default=cfg.Train.root_path, help='the training root path')
    parser.add_argument('--epochs', default=cfg.Train.epochs, help='the training epochs')
    # parser.add_argument('--lr', type=float, default=cfg.Train.learning_rate, help='start learning rate')
    # parser.add_argument('--lr_step', default=cfg.Train.backbone_lstep, help='period of learning rate decay')
    # parser.add_argument('--lr_decay_rate', default=cfg.Train.backbone_ldecay, help='learning rate decay')
    parser.add_argument('--optimizer', default=cfg.Train.optimizer, help='optimizer')
    # parser.add_argument('--weight_decay', type=float, default=cfg.Train.weight_decay, help='weight decay')
    parser.add_argument('--mom', type=float, default=cfg.Train.momentum, help='momentum')
    parser.add_argument('--batch_size', type=int, default=cfg.Train.batch_size, help='batch size in each context')
    parser.add_argument('--store_epoch', default=cfg.Train.store_epoch, help='store the model epoch')
    parser.add_argument('--test_epoch', default=cfg.Valid.step, help='test the model step')
    parser.add_argument('--store_path', default=cfg.Train.store_path, help='the model stored path')
    parser.add_argument('--fiqa_model', default=cfg.Train.fiqa, help='the Face Image Quality Model')
    parser.add_argument('--emb_size', type=int, default=512, help='embedding length')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    args = parser.parse_args()
    return args


def set_logger(args):
    # today = time.strftime("%Y_%m_%d", time.localtime(time.time()))
    global today
    logger_path = f"..\\logging_train\\{args.method}\\{today}_{cfg.Train.dataset}"
    if sys.platform == "linux":
        logger_path = logger_path.replace("\\", "/")
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)
    fh = logging.FileHandler(os.path.join(logger_path, "Train_details.log"))
    formats = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formats)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)


def get_dataset_infotmation():
    '''
    获得数据集相关的信息，比如数据集图片的根路径、图片id、图片分数等
    :return:
    '''

    imgs_root_path = getattr(cfg.Train.imgs_root, cfg.Train.dataset)
    imgs_root_path = imgs_root_path.replace("\\", "/") if sys.platform == "linux" else imgs_root_path

    faces_id = generate_face_id(imgs_root_path)
    faces_images_paths = generate_face_images_paths(imgs_root_path, bound=float('inf'))

    # 如果保存好了人脸分数，直接加载
    faces_quality_score = dict(generate_face_quality(imgs_root_path, cfg.Train.dataset, \
                                                     faces_images_paths, store_path=cfg.fiqa_score_path, \
                                                     fiqa_model=cfg.Train.fiqa)) if cfg.Train.face_score else None

    return faces_images_paths, faces_id, faces_quality_score


def get_train_test(faces_imgs_paths, test_ratio=0.2, train_gap=1):
    '''
    获取训练测试集的图片路径列表
    :param faces_imgs_paths: 所有的图片列表
    :param test_ratio: 测试集比例
    :param train_gap: 训练集图片的采样gap（针对一次数据过多）
    :return:
    '''
    faces_number = len(faces_imgs_paths)
    if test_ratio != 0:
        gap = int(1 // test_ratio)
        test_index = [i for i in range(faces_number) if i % gap == 0]
        train_index = [i for i in range(faces_number) if i not in test_index]
        train_imgs_paths = [faces_imgs_paths[i] for i in train_index]
        test_imgs_paths = [faces_imgs_paths[i] for i in test_index]
    else:
        train_imgs_paths = [faces_imgs_paths[i] for i in range(faces_number) if i % train_gap == 0]
        test_imgs_paths = None
    return train_imgs_paths, test_imgs_paths


def class_accuracy(predict, target):
    '''
    计算预测的准确度
    :param predict: the predict tensor of the trained model 预测类别
    :param target:  the true label of the face image 实际标签
    :return: classification accuracy of the model 分类准确率
    '''

    predict_label = torch.argmax(predict, dim=1)
    predict_label = torch.squeeze(predict_label).int()
    target = torch.squeeze(target).int()
    result = predict_label.eq(target).bool()
    result = torch.squeeze(result)
    accuracy = torch.sum(result).float() / result.shape[0]
    return accuracy


def load_data(args):
    '''
    加载训练的数据集
    :param args:
    :return: 数据loader
    '''
    mask_info = False if args.method != "MFaceSphereface_v2" else True
    # *-----------------------数据准备-----------------------*
    logger.info("\n" + "-" * 8 + "开始加载训练人脸的数据" + "-" * 8)
    faces_images_paths, faces_id, faces_quality_score = get_dataset_infotmation()

    train_images_paths, test_images_paths = get_train_test(faces_images_paths, test_ratio=cfg.Train.test_ratio)
    # *----------------------划分训练集和测试集----------------*
    train_dataset = Face_Dataset(getattr(cfg.Train.imgs_root, cfg.Train.dataset), train_images_paths,
                                 faces_id, faces_quality_score, seed=1, mask_flag=mask_info,
                                 img_input=cfg.Train.img_input)

    train_iterations = int(len(train_images_paths) / args.batch_size)
    test_iterations = int(len(test_images_paths) / args.batch_size) if test_images_paths != None else 0
    test_dataset = Face_Dataset(getattr(cfg.Train.imgs_root, cfg.Train.dataset), test_images_paths,
                                faces_id, faces_quality_score, seed=1, mask_flag=mask_info,
                                img_input=cfg.Train.img_input) if test_images_paths != None else None

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, **cfg.Train.dataloader)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True) if test_dataset != None else None

    test_num, test_id = 0, 0 if test_iterations == 0 else len(test_dataset), test_dataset.face_id_numbers

    logger.info("\n" + "本次的人脸数据集一共有{0}张人脸图像".format(len(faces_images_paths)) + \
                "\n" + f"其中训练人脸图像为{len(train_dataset)}张" + \
                "\n" + f"训练人脸ID数目为{train_dataset.face_id_numbers}" + \
                "\n" + f"测试人脸图像为{test_num}张" + \
                "\n" + f"测试人脸ID数目为{test_id}")
    # *-----------------------加载模型-----------------------*

    train_info = {"face_classes": train_dataset.face_id_numbers,
                  "face_nums": len(train_dataset),
                  "iterations": train_iterations}

    test_info = {"face_classes": test_dataset.face_id_numbers,
                 "face_nums": len(test_dataset),
                 "iterations": test_iterations} if test_dataset != None else None

    return train_loader, test_loader, train_info, test_info


def load_model(args, num_classes):
    '''
    :param args:
    :param num_classes: 人脸的类别
    :return: 网络模型
    '''
    if args.pretrained != None:
        backbone = torch.load(args.pretrained[0])
        metricbone = torch.load(args.pretrained[1]) if args.pretrained[1] != None else None
        logger.info("\n" + "-" * 8 + "从上一次训练好的模型加载" + "-" * 8)
        logger.info(f"模型路径为: {args.pretrained[0]}")
        print("加载上一次的模型")
    else:
        logger.info(f"\n本次使用的方法为{args.method}")
        backbone = load_backbone(cfg.Train.backbone, **{'use_se': False, "num_layerss": 50, "drop_ratio": 0.6})
        metricbone = load_metricsbone(cfg.Train.metric,
                                      **{"in_features": cfg.face_dimension, "out_features": num_classes, 's': 30.0,
                                         'm': 0.50, "easy_margin": False})

    return backbone, metricbone


def model_change(backbone, metrics_bone, current_epoch, linux_flag):
    backbone_, metrics_bone_ = backbone.module, metrics_bone.module if hasattr(backbone, 'module') else \
        backbone, metrics_bone

    for param in backbone_.parameters():
        param.requires_grad = False
        param.requires_grad = False
    if current_epoch >= cfg.Train.backbone_epoch:
        for param in backbone_.parameters():
            param.requires_grad = True
    if current_epoch >= cfg.Train.metricbone_epoch and metrics_bone != None:
        for param in metrics_bone_.parameters():
            param.requires_grad = True
    # for x in backbone_.parameters():
    #     if x.requires_grad == False:
    #         print("fatal")


def lr_update(optimizer, backbone_ldecay, metricbone_ldecay, qualalign_ldecay):
    # if sys.platform == "linux":
    #   optimizer_ = optimizer.module()
    # else:
    #    optimizer_ = optimizer
    for p in optimizer.param_groups:
        if p["name"] == "Backbone":
            p['lr'] *= backbone_ldecay
        elif p['name'] == 'Qualityalign':
            p['lr'] *= qualalign_ldecay
        else:
            p['lr'] *= metricbone_ldecay

    return optimizer


def build_opt_lr(backbone, metrics_bone, quality_align):
    backbone_ = backbone
    metrics_bone_ = metrics_bone if metrics_bone != None else None
    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad, backbone_.parameters()), \
                          'lr': cfg.Train.backbone_lr,
                          'weight_decay': cfg.Train.backbone_wdecay,
                          'name': "Backbone"}]
    if metrics_bone != None:
        trainable_params += [{'params': filter(lambda x: x.requires_grad, metrics_bone_.parameters()), \
                              'lr': cfg.Train.metricbone_lr,
                              'weight_decay': cfg.Train.metricbone_wdecay,
                              'name': "Metricbone"}]
    if quality_align != None:
        trainable_params += [{'params': filter(lambda x: x.requires_grad, quality_align.align_layer.parameters()), \
                              'lr': cfg.Train.qualalign_lr,
                              'weight_decay': cfg.Train.qualalign_wdecay,
                              'name': "Qualityalign"}]
    optimizer = torch.optim.Adam(trainable_params) if cfg.Train.optimizer == "adam" else torch.optim.SGD(
        trainable_params, momentum=cfg.Train.momentum)
    return optimizer


def load_eval_data():
    '''
    加载数据集的测试图像对
    :return: (图像的根路径, 图像对)
    '''
    imgs_pairs_path = getattr(cfg.Eval.imgs_pairs, cfg.Eval.dataset)
    imgs_root_path = getattr(cfg.Eval.imgs_root, cfg.Eval.dataset)
    if sys.platform == "linux":
        imgs_pairs_path = imgs_pairs_path.replace("\\", "/")
        imgs_root_path = imgs_root_path.replace("\\", "/")
    imgs_pairs = np.load(imgs_pairs_path, allow_pickle=True)
    return imgs_root_path, imgs_pairs


def eval(model, mfiqa_model=None):
    model_ = model.module if hasattr(model, 'module') else model
    model_.eval()
    imgs_root, imgs_pairs = load_eval_data()
    preprocess = Preprocess(input_size=cfg.Train.img_input)
    eval_result = []
    for name1, idx1, name2, idx2 in tqdm(imgs_pairs):
        img1_path = os.path.join(imgs_root, name1, '%s' % (idx1))
        img2_path = os.path.join(imgs_root, name2, '%s' % (idx2))
        imgs_tensor = preprocess.arcface(img1=img1_path, img2=img2_path)
        with torch.no_grad():
            output = model_(imgs_tensor.cuda())

            if cfg.Train.feature_align:
                output2 = mfiqa_model(imgs_tensor.cuda())

                output = torch.cat([output, output2], dim=1)
        f1, f1_flip, f2, f2_flip = output[0], output[1], output[2], output[3]
        feature_1 = torch.cat([f1, f1_flip]).view(-1)
        feature_2 = torch.cat([f2, f2_flip]).view(-1)
        similarity = feature_1.dot(feature_2) / (feature_1.norm() * feature_2.norm() + 1e-5)
        eval_result.append((similarity.item(), name1 == name2))
    threshold_list = [i / 100.0 for i in range(-100, 100, 1)]
    best_acc = 0
    best_thr = 0
    for thresh in threshold_list:
        accuracy = get_Acc(eval_result, thresh)
        if accuracy > best_acc:
            best_thr = thresh
            best_acc = accuracy
    return best_acc, best_thr
    # precision = get_Precision(eval_result, thresh)
    # recall = get_Recall(eval_result, thresh)
    # f_score = get_FScore(eval_result, thresh)
    # false_accept = get_FAR(eval_result, thresh)
    # false_reject = get_FRR(eval_result, thresh)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def mode_change(*nets, mode = "Train"):
    for net in nets:
        if not net: continue
        if hasattr(net, 'module'):
            if mode == "Train": net.module.train()
            else: net.module.eval()
        else:
            if mode == "Train": net.train()
            else: net.eval()
def train(args):
    logger.info('Config information: ')
    logger.info(str(cfg))
    train_loader, test_loader, train_info, test_info = load_data(args)
    backbone, metrics_bone = load_model(args, num_classes=train_info["face_classes"])
    # 辅佐层设置
    quality_align = QualiIndicLoss(classify_layer=metrics_bone) if cfg.Train.quality_loss else None
    # 质量分数指导下的损失
    if cfg.Train.quality_constraint:
        quality_constraint = Uq_constrloss(classify_layer=metrics_bone.module, margin=0.7, threshold=0.3) if hasattr(
            metrics_bone, "module") \
            else Uq_constrloss(classify_layer=metrics_bone)
    else:
        quality_constraint = None
    # 是否融入人脸质量MFIQA特征
    if cfg.Train.feature_fusion:
        if sys.platform == "linux":
            cfg.MFIQ_path = cfg.MFIQ_path.replace("\\", "/")
        mfiqa_model = torch.load(cfg.MFIQ_path).module.backbone
        freeze_model(mfiqa_model)
    else:
        mfiqa_model = None

    optimizer = build_opt_lr(backbone, metrics_bone, quality_align)
    device_ids = [0, 1]
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    # 损失函数的设置
    if cfg.Train.loss == "focal_loss":
        criterion = FocalLoss(gamma=2)
    elif cfg.Train.loss == "angle_loss":  # sphereface20a对应的损失函数，函数的输入需要是个二元的元组，(cos,phi) + label
        criterion = AngleLoss(gamma=0)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    criterion.to(device)
    # 并行数据设置
    if hasattr(backbone, 'module'):
        optimizer = nn.DataParallel(optimizer, device_ids=device_ids).module
    else:
        backbone = nn.DataParallel(backbone, device_ids=device_ids)
        metrics_bone = nn.DataParallel(metrics_bone, device_ids=device_ids) if metrics_bone != None else None
        optimizer = nn.DataParallel(optimizer, device_ids=device_ids).module
    # *-----------------------开始训练-----------------------*#
    logger.info("\n" + "-" * 8 + "开始训练人脸的模型" + "-" * 8)
    mini_loss = float("inf")
    backbone.to(device)
    metrics_bone.to(device)
    if quality_align != None:
        quality_align.to(device)
    index = 0
    # epoch_decay = cfg.Train.backbone_ldecay
    logger.info("\n" + "-" * 8 + "开始加载训练人脸的模型" + "-" * 8)
    for epoch in tqdm(range(args.epochs)):
        # time.sleep(0.01)
        mode_change([backbone, metrics_bone, quality_align], "Train")

        # if epoch==cfg.Train.backbone_epoch_start or epoch==cfg.Train.metricbone_epoch_start:
        #     model_change(backbone, metrics_bone, epoch, sys.platform=="linux")
        total_loss = 0

        accuracy = 0
        batch_number = 0
        try:
            for data_batch in tqdm(train_loader):

                index += 1


                data_x = data_batch["face_img"].float()
                data_y = data_batch["face_id"]
                # data_x_name = data_batch["name"
                data_x = Variable(data_x)
                data_y = Variable(data_y)
                if data_x.shape[0] == 1:
                    continue
                data_score = data_batch["face_score"].float() if data_batch["face_score"] != None else None

                face_feature = backbone(data_x.to(device))

                if cfg.Train.feature_align:
                    data_x_qualifea = mfiqa_model(data_x.cuda())
                    face_feature = torch.cat([face_feature, data_x_qualifea], dim=1)

                output = metrics_bone(face_feature, label=data_y.to(device))  # 这个的face_feature是输出前的特征
                quality_loss = quality_align(face_feature, data_y.cuda(), data_score.cuda()) \
                    if quality_align != None else 0
                quality_constraint_loss = quality_constraint(face_feature, data_y.cuda(),
                                                             data_score.cuda()) if cfg.Train.quality_constraint else 0
                batch_loss = criterion(output,
                                           data_y.to(device).view(-1)).float() + quality_loss + quality_constraint_loss



                optimizer.zero_grad()
                batch_loss.backward(batch_loss.clone().detach())
                optimizer.step()
                # print(type(output))
                if len(output) == 2:
                    output = output[0]  # 只取第一个cos值

                batch_accuracy = class_accuracy(output, data_y.to(device))

                accuracy += batch_accuracy

                batch_number += 1

                total_loss += batch_loss
            accuracy = accuracy / (batch_number + 1e-5)
            logger.info(
                "\n" + "-" * 10 + f"第{epoch}次的训练损失为 {total_loss / (batch_number + 1e-5)} 识别准确率为 {accuracy}" + "-" * 10)
            print("-" * 10 + f"第{epoch}次的训练损失为  {total_loss / (batch_number + 1e-5)} 识别准确率为 {accuracy}" + "-" * 10)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            accuracy = accuracy / (batch_number + 1e-5)
            logger.info(
                "\n" + "-" * 10 + f"第{epoch}次的训练损失为 {total_loss / (batch_number + 1e-5)} 识别准确率为 {accuracy}" + "-" * 10)
            print("-" * 10 + f"第{epoch}次的训练损失为  {total_loss / (batch_number + 1e-5)} 识别准确率为 {accuracy}" + "-" * 10)
        else:
            print("Training successfully!")
        if epoch in cfg.Train.backbone_lstep:
            optimizer = lr_update(optimizer, backbone_ldecay=cfg.Train.backbone_ldecay, \
                                  metricbone_ldecay=cfg.Train.metricbone_ldecay, \
                                  qualalign_ldecay=cfg.Train.qualalign_ldecay)

        if epoch % args.test_epoch == 0 and test_loader != None:
            mode_change([backbone, metrics_bone, quality_align], "Eval")

            total_test_loss = 0
            accuracy = 0
            batch_number = 0
            for index, data_batch in enumerate(test_loader):
                with torch.no_grad():
                    data_x = data_batch["face_img"]
                    data_y = data_batch["face_id"]
                    if data_x.shape[0] == 1:
                        continue
                    data_score = data_batch["face_score"]
                    face_feature = backbone(data_x.to(device))


                    output = metrics_bone(face_feature, label=data_y.to(device))  # 这个的face_feature是输出前的特征

                    quality_loss = quality_align(face_feature, data_y.cuda(), data_score.cuda()) \
                        if quality_align != None else 0
                    batch_loss = criterion(output, data_y.to(device).view(-1)) + quality_loss

                    total_test_loss += batch_loss

                    batch_accuracy = class_accuracy(output, data_batch["face_id"].to(device))
                    accuracy += batch_accuracy
                    batch_number += 1
            accuracy = accuracy / (batch_number + 1e-5)
            logger.info(
                "\n" + "-" * 10 + f"第{epoch}次的测试损失为 {total_test_loss / test_info['iterations']} 识别准确率为 {accuracy}" + "-" * 10)
            print(
                "-" * 10 + f"第{epoch}次的测试损失为   {total_test_loss / test_info['iterations']} 识别准确率为 {accuracy}" + "-" * 10)
        global today

        if total_loss < mini_loss:
            path = cfg.Train.store_path + f"\\{cfg.Train.method}\\{today}_{cfg.Train.dataset}\\"
            mini_loss = total_loss
            backbone_store_path = cfg.Train.store_path + f"\\{cfg.Train.method}\\{today}_{cfg.Train.dataset}\\{cfg.Train.backbone}_{epoch}.ckpt"
            metrics_bone_path = cfg.Train.store_path + f"\\{cfg.Train.method}\\{today}_{cfg.Train.dataset}\\{cfg.Train.backbone}_{cfg.Train.metric}_{epoch}.ckpt"
            if sys.platform == "linux":
                path = path.replace("\\", "/")
                backbone_store_path = backbone_store_path.replace("\\", "/")
                metrics_bone_path = metrics_bone_path.replace("\\", "/")
            if not os.path.exists(path):
                os.makedirs(path)
            if cfg.Train.backbone in ["resnet50_IR_SE", "resnet50_IR"]:
                torch.save(backbone.state_dict(), backbone_store_path.split('.')[0] + '.pth')
            else:
                torch.save(backbone, backbone_store_path)
            if metrics_bone != None:
                torch.save(metrics_bone, metrics_bone_path)
        if epoch % 2 == 0:
            best_acc, best_thr = eval(backbone, mfiqa_model=mfiqa_model)
            logger.info("\n" + "-" * 10 + f"第{epoch}次在{cfg.Eval.dataset}数据集上 acc {best_acc} 阈值为 {best_thr}" + "-" * 10)
            print("\n" + "-" * 10 + f"第{epoch}次在{cfg.Eval.dataset}数据集上 acc {best_acc} 阈值为 {best_thr}" + "-" * 10)


def main():
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    set_logger(args)
    train(args)


if __name__ == "__main__":
    main()
