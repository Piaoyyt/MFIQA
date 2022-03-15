import sys

import torch
from torch.utils.data import  Dataset
from PIL import Image
import torchvision.transforms as transforms
import os,random
from pyIUA.dataset.mask_img_utils import gen_mask_img
import matplotlib.pyplot as plt
# import torch.nn as nn
# import numpy as np
class VariableError(BaseException):
    # print("类型错误！请输字典型")
    pass
class Face_Dataset(Dataset):

    def __init__(self, root_path, face_images_paths, face_id, face_quality,seed=1, mask_flag = False, img_input = [112, 112]):
        """
        :param root_path:The root path of the Face Images
        :param face_images_paths: The all face images paths
        :param face_labels: The all face images labels(dict) such {"lht":0,"yyt"：1}
        :param face_quality: The all face image quality score
        """
        super(Face_Dataset, self).__init__()
        self.mask_flag = mask_flag
        if sys.platform=="linux":
            root_path = root_path.replace("\\", "/")
        self.root_path = root_path
        if not isinstance(face_images_paths,list):
            face_images_paths = list(face_images_paths)
  
        if not isinstance(face_id, dict):
            raise VariableError()
        #if not isinstance(face_quality, dict):
        #    raise VariableError()
        # assert(type(face_id)=="<class 'dict'>")
        # assert(type(face_quality)=="<class 'dict'>")
        # face_id = set()
        self.face_images_paths = face_images_paths
        #self.face_labels = face_labels
        self.face_quality = face_quality
        # for face_img_path in self.face_images_paths:
        #     face_id = face_img_path.split("//")[0]
        #     face_id.add(face_id)
        self.face_id = face_id
        self.face_id_numbers = len(self.face_id.keys())
        self.face_numbers = len(face_images_paths)
        #shuffle
        #self.face_images_paths = np.array(self.face_images_paths)
        # np.random.seed(seed)
        # np.random.shuffle(self.face_images_paths)
        #transformer setting
        self.train_transform = transforms.Compose(
            [
                transforms.Resize([img_input[0], img_input[1]]),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    def __len__(self):
        return len(self.face_images_paths)
    def __getitem__(self, index):
        data = {}
        face_path = str(self.face_images_paths[index])
        if sys.platform == "linux":
            face_id = self.face_id[face_path.split("/")[0]]
        else:
            face_id = self.face_id[face_path.split("\\")[0]]
        face_img = Image.open(os.path.join(self.root_path, self.face_images_paths[index])).convert("RGB")
        #plt.imshow(face_img)
        #splt.show()
        if self.mask_flag:
            #img_mask = face_img.copy
            img_mask = gen_mask_img(face_img)
            img_mask = self.train_transform(img_mask)
        if random.random() > 0.5 and not self.mask_flag:
            face_img = face_img.transpose(Image.FLIP_LEFT_RIGHT)
        # plt.imshow(face_img)
        # if face_img.shape[0]==1:
        face_img = self.train_transform(face_img)
        if sys.platform == "linux":
            # face_path = '/'.join(face_path.split("/"))
            pass
        else:
            face_path = '/'.join(face_path.split("\\"))
        face_score = float(self.face_quality[face_path]) if self.face_quality !=None  else 0
        data["face_img"] = face_img
        data["face_id"] = torch.LongTensor([int(face_id)])
        data["face_score"] = face_score
        data["name"] = face_path
        if self.mask_flag:
            data["face_img_mask"] = img_mask
        return data






