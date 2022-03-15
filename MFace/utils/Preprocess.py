import cv2, torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms


class Preprocess(object):
    def __init__(self, input_size = [112, 112]):
        self.size = input_size
        self.transform = transforms.Compose(
            [transforms.Resize([input_size[0], input_size[1]]),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
             ])
    def arcface(self, **kwargs):
        img_list = []
        for img_path in kwargs.keys():
            img_p = kwargs[img_path]
            img = Image.open(img_p).convert("RGB")
            img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_flip = self.transform(img_flip).reshape(-1, 3, self.size[0], self.size[1])
            img = self.transform(img).reshape(-1, 3, self.size[0], self.size[1])
            img_list.append(img)
            img_list.append(img_flip)
        return torch.vstack([*img_list])

    def Insighface(self, img1_path, img2_path):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img1 = cv2.resize(cv2.imread(img1_path), dsize=(112, 112))
        img1 = img1[..., ::-1]  # RGB
        img1 = Image.fromarray(img1, 'RGB')  # RGB
        img1 = transform(img1).reshape(-1, 3, 112, 112)
        img2 = cv2.resize(cv2.imread(img2_path), dsize=(112, 112))
        img2 = img2[..., ::-1]  # RGB
        img2 = Image.fromarray(img2, 'RGB')  # RGB
        img2 = transform(img2).reshape(-1, 3, 112, 112)
        return torch.vstack(img1, img2)

    def MFaceSphereface(self, img1_path, img2_path):
        return self.Sphereface(img1_path, img2_path)