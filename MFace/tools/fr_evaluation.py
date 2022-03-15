import os
import random
import sys
project_path = "\\".join(os.getcwd().split("\\")[:-2]) if not sys.platform=="linux"\
    else "/".join(os.getcwd().split("/")[:-2])
sys.path.append(project_path)
if sys.platform=="linux":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm
from torchvision import transforms
from MFace.Conf.config import cfg
from MFace.utils.Eva_criterion import *
from MFace.net_struture.net_sphere_ori import sphere20a
from MFace.net_struture.backbone.ir_resnet_se import Backbone
from MFace.net_struture.backbone.resnet import *
import argparse
import logging
import time

logger = logging.getLogger('global')
today = time.strftime("%Y_%m_%d_%H", time.localtime(time.time()))
def parse():
    parser = argparse.ArgumentParser(description="Face Recognition Evaluation")
    parser.add_argument('--seed', type=int, default=123456,
                        help='random seed')
    # parser.add_argument('--config', type=str, default=cfg.Config_path,
    #                     help="the config file path")
    args = parser.parse_args()
    return args
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
def set_logger():
    global today
    file_path = cfg.Eval.store_root + f"{cfg.Eval.face_recoginition}\\{today}_{cfg.Eval.dataset}"
    if sys.platform == "linux":
        file_path = file_path.replace("\\", "/")
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    fh_path = os.path.join(file_path, "Eval_details.log")
    fh = logging.FileHandler(fh_path)
    formats = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formats)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)
def load_eval_data():
    '''
    加载数据集的测试图像对
    :return: (图像的根路径, 图像对)
    '''
    imgs_pairs_path = getattr(cfg.Eval.imgs_pairs, cfg.Eval.dataset)
    imgs_root_path = getattr(cfg.Eval.imgs_root, cfg.Eval.dataset)
    if sys.platform == "linux":
        imgs_pairs_path= imgs_pairs_path.replace("\\", "/")
        imgs_root_path  = imgs_root_path.replace("\\", "/")
    imgs_pairs = np.load(imgs_pairs_path, allow_pickle=True)
    return imgs_root_path, imgs_pairs
def load_model():
    parallel_module = ['Sphereface', 'MFaceSphereface','Arcface','Sphereface_Qualign',"Sphereface_two_fealign", 'Arcface_quali1']
    model_path = cfg.Eval.model_path
    if sys.platform == "linux":
        model_path = model_path.replace("\\", "/")
    if cfg.Eval.model_name in parallel_module:
        model = torch.load(model_path)
        model = model.cuda() if torch.cuda.is_available() else model
        model.module.eval()
        if hasattr(model.module, 'feature'): model.module.feature = True
    elif cfg.Eval.model_name == "Sphere_ori":
        model = sphere20a(feature=True).cuda()
        model.load_state_dict(torch.load(model_path))
    elif cfg.Eval.model_name == "resnet50_IR_SE":
        model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
        # model = nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    elif cfg.Eval.model_name == "resnet_face18":
        model = resnet_face18(cfg.Train.use_se)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    return model
# def get_similarity(model, img1, img2):
#     model.feature = True
class Preprocess(object):
    def __init__(self):
        self.scale = cfg.Eval.size
        self.transform = transforms.Compose(
            [transforms.Resize(self.scale[0:2]),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
    def Sphere_ori(self, img1_path, img2_path):
        numpy_list = []
        img1 = cv2.imread(img1_path)
        img1_flip = cv2.flip(img1, 1)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            return None
        img2_flip = cv2.flip(img2, 1)
        img1_resize = cv2.resize(img1, dsize=(96, 112)).transpose(2, 0, 1)
        img1_flip_resize = cv2.resize(img1_flip, dsize=(96, 112)).transpose(2, 0, 1)

        img1_numpy = np.array(img1_resize).reshape(3, 112, 96)
        img1_numpy = (img1_numpy - 127.5) / 128.0
        img1_flip_numpy = np.array(img1_flip_resize).reshape(3, 112, 96)
        img1_flip_numpy = (img1_flip_numpy - 127.5) / 128.0

        img2_resize = cv2.resize(img2, dsize=(96, 112)).transpose(2, 0, 1)
        img2_flip_resize = cv2.resize(img2_flip, dsize=(96, 112)).transpose(2, 0, 1)
        img2_numpy = np.array(img2_resize).reshape(3, 112, 96)
        img2_numpy = (img2_numpy - 127.5) / 128.0
        img2_flip_numpy = np.array(img2_flip_resize).reshape(3, 112, 96)
        img2_flip_numpy = (img2_flip_numpy - 127.5) / 128.0
        numpy_list.append(img1_numpy)
        numpy_list.append(img1_flip_numpy)
        numpy_list.append(img2_numpy)
        numpy_list.append(img2_flip_numpy)
        img_v = np.stack(numpy_list)
        img = Variable(torch.from_numpy(img_v).float(), volatile=True)
        return img
    def Sphereface(self, img1_path, img2_path):
        if not os.path.exists(img1_path) or not os.path.join(img2_path):
            return None
        img1 = Image.open(img1_path).convert("RGB")
        img1_flip = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = Image.open(img2_path).convert("RGB")
        img2_flip = img2.transpose(Image.FLIP_LEFT_RIGHT)
        if self.scale[2] == 1:
            img1 = img1.convert('L')
            img1_flip = img1_flip.convert('L')
            img2 = img2.convert('L')
            img2_flip = img2_flip.convert('L')
        img1 = self.transform(img1).view(-1, self.scale[2], self.scale[0], self.scale[1])
        img1_flip = self.transform(img1_flip).view(-1, self.scale[2], self.scale[0], self.scale[1])
        #img1 = torch.cat((img1, img1_flip), dim=0)
        img2 = self.transform(img2).view(-1, self.scale[2], self.scale[0], self.scale[1])
        img2_flip = self.transform(img2_flip).view(-1, self.scale[2], self.scale[0], self.scale[1])
        img = torch.cat([img1, img1_flip, img2, img2_flip], dim=0).view(-1, self.scale[2], self.scale[0], self.scale[1])
        return img
	#numpy_list = []
        #img1 = cv2.imread(img1_path)
        #img2 = cv2.imread(img2_path)
        #img1_resize = cv2.resize(img1, dsize=(96, 112)).transpose(2, 0, 1)
        #img1_numpy = np.array(img1_resize).reshape(3, 112, 96)
        #img1_numpy = (img1_numpy - 127.5) / 128.0

        #img2_resize = cv2.resize(img2, dsize=(96, 112)).transpose(2, 0, 1)
        #img2_numpy = np.array(img2_resize).reshape(3, 112, 96)
        #img2_numpy = (img2_numpy - 127.5) / 128.0

        #numpy_list.append(img1_numpy)
        #numpy_list.append(img2_numpy)
        #img_v = np.stack(numpy_list)
        #img = Variable(torch.from_numpy(img_v).float(), volatile=True)
        return img
    def Insighface(self, img1_path, img2_path):
        transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img1 = cv2.resize(cv2.imread(img1_path), dsize=(self.scale[0], self.scale[1]))
        img1 = img1[..., ::-1]  # RGB
        img1 = Image.fromarray(img1, 'RGB')  # RGB
        img1 = transform(img1).reshape(-1, 3, self.scale[0], self.scale[1])
        img2 = cv2.resize(cv2.imread(img2_path), dsize=(self.scale[0], self.scale[1]))
        img2 = img2[..., ::-1]  # RGB
        img2 = Image.fromarray(img2, 'RGB')  # RGB
        img2 = transform(img2).reshape(-1, 3, self.scale[0], self.scale[1])
        return torch.vstack(img1, img2)
    def MFaceSphereface(self, img1_path, img2_path):
        return self.Sphereface(img1_path, img2_path)
    def Arcface(self, img1_path, img2_path):
        return self.Sphereface(img1_path, img2_path)
    def Arcface_quali1(self, img1_path, img2_path):
        return self.Sphereface(img1_path, img2_path)
    def Sphereface_Qualign(self, img1_path, img2_path):
        return self.Sphereface(img1_path, img2_path)
    def Sphereface_two_fealign(self, img1_path, img2_path):
        return self.Sphereface(img1_path, img2_path)
    def resnet50_IR_SE(self, img1_path, img2_path):
        return self.Sphereface(img1_path, img2_path)
    def resnet_face18(self, img1_path, img2_path):
        return self.Sphereface(img1_path, img2_path)
    # def __call__(self, *args, **kwargs):

def eval():
    if cfg.Eval.feature_align:
        if sys.platform == "linux":
            cfg.MFIQ_path = cfg.MFIQ_path.replace("\\", "/")
        mfiqa_model = torch.load(cfg.MFIQ_path).module.backbone.eval()
    else:
        mfiqa_model = None
    #model = face_recognition(model_name = cfg.Eval.model_name)
    model = load_model()
    model.cuda()
    #model.module.eval()
    #print(model.module.feature)
    #for child in model.module.children():
    #    print(child)
    #print(model.module.children()[-1])
    preprocess = Preprocess()
    imgs_root, imgs_pairs = load_eval_data()
    eval_result = []
    recognition_result = {}
    eval_path = cfg.Eval.store_path + "\\recognition_performance"
    if sys.platform =="linux":
        eval_path = eval_path.replace("\\", "/")
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    model_time = cfg.Eval.model_path.split('\\')[-2]
    eval_model_path = eval_path + f"\\{cfg.Eval.model_name}_{cfg.Eval.dataset}\\{model_time}"
    if sys.platform =="linux":
        eval_model_path = eval_model_path.replace("\\", "/")
    if not os.path.exists(eval_model_path):
        os.makedirs(eval_model_path)
    model_name = cfg.Eval.model_path.split('\\')[-1].split('.')[0]
    eval_model_path = eval_path + f"\\{cfg.Eval.model_name}_{cfg.Eval.dataset}\\{model_time}\\{model_name}"
    if sys.platform =="linux":
        eval_model_path = eval_model_path.replace("\\", "/")
    if not os.path.exists(eval_model_path):
        os.makedirs(eval_model_path)
    record_file = eval_model_path + "\\record.txt"
    record_file = record_file.replace("\\", "/")  if sys.platform == "linux" else record_file
    recognition_npy_file = eval_model_path +"\\pairs_similarity.npy"
    recognition_npy_file = recognition_npy_file.replace("\\", "/")  if sys.platform == "linux" else recognition_npy_file
    recognition_txt_file = eval_model_path +"\\pairs_similarity.txt"
    recognition_txt_file = recognition_txt_file.replace("\\", "/")  if sys.platform == "linux" else recognition_txt_file
    if os.path.exists(recognition_npy_file):
        recognition_npy = np.load(recognition_npy_file, allow_pickle=True).item()
        for pairs in recognition_npy.keys():
            similarity = recognition_npy[pairs]
            eval_result.append((similarity.item(), pairs[0] == pairs[2]))
    else:
        for name1, idx1, name2, idx2 in tqdm(imgs_pairs):
            img1_path = os.path.join(imgs_root, name1, '%s' % (idx1))
            img2_path = os.path.join(imgs_root, name2, '%s' % (idx2))
            imgs_tensor = preprocess.__getattribute__(cfg.Eval.model_name)(img1_path, img2_path)
            if imgs_tensor == None:
                continue
            with torch.no_grad():
                output = model(imgs_tensor.cuda())
                if cfg.Eval.feature_align:
                    output2 = mfiqa_model(imgs_tensor.cuda())
                    output = torch.cat([output, output2], dim=1)
        #print(output.shape)
        #print(output[0][0].shape)
            f1, f1_flip, f2, f2_flip = output[0], output[1], output[2], output[3]
        #print(f1.shape)
            f1_ = torch.cat((f1, f1_flip), dim=0).view(-1)
            print(f1_.shape)
            f2_ = torch.cat((f2, f2_flip), dim=0).view(-1)
            similarity = f1_.dot(f2_) / (f1_.norm() * f2_.norm() + 1e-5)
            #dis = torch.sqrt(torch.square(f1-f2).sum())
            with open(recognition_txt_file, "a+") as f:
                f.write(f"{name1},{idx1},{name2},{idx2},{similarity}\n")
            recognition_result[(name1, idx1, name2, idx2)] = similarity
            eval_result.append((similarity.item(), name1 == name2))
        np.save(recognition_npy_file , recognition_result)
    threshold_list = [i / 100.0 for i in range(-100, 100, 1)]
    for thresh in threshold_list:
        accuracy = get_Acc(eval_result, thresh)
        precision = get_Precision(eval_result, thresh)
        recall = get_Recall(eval_result, thresh)
        f_score = get_FScore(eval_result, thresh)
        false_accept = get_FAR(eval_result, thresh)
        false_reject = get_FRR(eval_result, thresh)
        with open(record_file, "a+") as f:
            f.write(f"{thresh} ---- accuracy: {accuracy}\n"
                    f"---- precision: {precision} \n"
                    f"---- recall: {recall} \n"
                    f"---- f_score:{f_score}\n"
                    f"---- false accept rate:{false_accept}\n"
                    f"---- false reject rate:{false_reject}\n")
if __name__ == "__main__":
    eval()

