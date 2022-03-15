from torch.utils.data import  Dataset
import torchvision.transforms as transforms
import os,sys
import torch
import numpy as np
from PIL import Image
import random, cv2
#from ..utils.feature  import Hog_descriptor
#from ..utils.brisque.brisque import BRISQUE
from .mask_img_utils import gen_mask_img
import matplotlib.pyplot as plt
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ImgDatabase(Dataset):
    def __init__(self, img_paths, annotation_path, annotation_method):
        super(ImgDatabase, self).__init__()
        # self.device = device
        # self.mtcnn = face_detection
        self.img_paths = img_paths
        self.annotation = np.load(annotation_path,allow_pickle=True).item()
        self.annotation_method = annotation_method
        train_transform = transforms.Compose(
            [#transforms.Resize(size=(160, 160)),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
             ])
        eval_transform = transforms.Compose(
            [transforms.Resize(size=(160, 160)),
             transforms.ToTensor(),

             ])
        self.transform = train_transform  # if mode =="train" else eval_transform
        self.platform = "linux" if sys.platform=="linux" else "win10"
        #self.brisque =BRISQUE()
        # self.landmark_path = landmark_path
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index, label_type="continuous"):
        image_path = self.img_paths[index]
        image_name = "\\".join(image_path.split('\\')[-2:]) if self.platform=="win10" else\
            "\\".join(image_path.split('/')[-2:])#img_name  (person_name\img_name)
        img = Image.open(image_path)
        # plt.imshow(img)
        # plt.pause(5)
        # plt.ioff()
        # plt.clf()
        # plt.close()
        #brisque_feature = self.brisque.get_feature(image_path)
        #brisque_feature = torch.from_numpy(np.squeeze(brisque_feature).reshape(-1)).float()
        #生成掩膜图像
        img_mask = img.copy
        img_mask = gen_mask_img(img)
        # plt.imshow(img_mask)
        # plt.pause(5)
        # plt.ioff()
        # plt.clf()
        # plt.close()
        # X = self.mtcnn(img)
        X = self.transform(img)
        #X__mask = torch.randn([3,160,160])#
        X__mask = self.transform(img_mask)
        # X = torch.reshape(X, [3, 160, 160])
        #X__mask =torch.reshape(X__mask,[3, 160, 160])
        # X_ = torch.stack((X,X__mask),0)
        if label_type == "continuous":
            Y = self.annotation[image_name][self.annotation_method]
        elif label_type == "binary":
            Y = self.binary_label[image_name][self.annotation_method]
        data = {}
        data["input"] = X
        #data["brisque"] = #brisque_feature
        data["input_mask"] = X__mask
        data["quality_label"] = Y

        return data
    def get_landmark(self):
        landmark_dict = {}
        with open(self.landmark_path,"r") as f:
            for line in f.readlines():
                split_list = line.split("*")
                image_name = split_list[0].split("/")[-1]
                landmark_dict[image_name] =[]
                landmark_dict[image_name].append(split_list[2:6])#left eye box
                landmark_dict[image_name].append(split_list[7:11])#right eye box
                landmark_dict[image_name].append(split_list[12:16])#nose box
                landmark_dict[image_name].append(split_list[17:21])#mouth box
        self.landmark_result = landmark_dict
    # def get_ori_mask_pari
    def getbatch(self, indices, label_type="continuous"):
        '''
        :param indices: the image path index
        :param label_type:  the type of the label optional ["continuous", "binary"]
        :return: image tensor[n, channels, width, height], image label[n, 1]
        '''
        img_list = []
        img_mask_list = []
        labels_list = []
        brisque_list = []
        for i in indices:
            data = self.__getitem__(i, label_type=label_type)
            # if image ==None:
            #     print(self.img_paths[i])
            #     continue
            img_list.append(data["input"])
            #brisque_list.append(data["brisque"])
            img_mask_list.append(data["input_mask"])
            labels_list.append(data["quality_label"])
        return torch.stack(img_list), torch.stack(img_mask_list) , None, torch.tensor(labels_list).float()

    # def get_hog_batch(self, indices):
    #     hog_list = []
    #     for i in indices:
    #         img_path = self.img_paths[i]
    #         image = cv2.imread(img_path)
    #         image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         hog2 = Hog_descriptor(image_gray, cell_size=16, bin_size=8)
    #         hog_feature, img = hog2.extract()
    #         hog_feature = torch.from_numpy(np.squeeze(np.array(hog_feature).reshape(1, -1))).float()
    #         hog_list.append(hog_feature)
    #     # print(hog_feature.shape)
    #     # print(len(hog_list))
    #     return torch.stack(hog_list)
    def get_brisque_batch(self, indices):
        brisque_list =[]
        brisque =BRISQUE()
        for i in indices:
            img_path = self.img_paths[i]
            brisque_feature = brisque.get_feature(img_path)
            brisque_feature = torch.from_numpy(np.squeeze(brisque_feature).reshape(1,-1)).float()
            brisque_list.append(brisque_feature)
        return  torch.squeeze(torch.stack(brisque_list))
    def filter(self, choice="1", thresh=0):
        '''

        :param choice: 1-过滤掉基准和类基准图像，2-过滤掉基准类基准和thresh分数之下的图像，\
            3-过滤基准图像和thresh下的图像， 4-只过滤thresh之下的图像
        :param thresh: 用于过滤
        :return:
        '''

        img_paths = self.img_paths.copy()
        # print(len(img_paths))
        for path in img_paths:
            img_name = path.split('\\')[-1]
            pic_name_split_list = img_name.split('_')
            if choice == "1":
                if "0001" in pic_name_split_list or "0001.jpg" in pic_name_split_list:
                    self.img_paths.remove(path)
            elif choice == "2":
                if "0001" in pic_name_split_list or "0001.jpg" in pic_name_split_list \
                        or self.annotation[img_name][self.annotation_method] < thresh:
                    self.img_paths.remove(path)
            elif choice =="3":
                if "0001.jpg" in pic_name_split_list or self.annotation[img_name][self.annotation_method] < thresh:
                    self.img_paths.remove(path)
            elif choice =="4":
                if self.annotation[img_name][self.annotation_method] < thresh:
                    self.img_paths.remove(path)
        # print(count)
        # if choice == "1":
        #     for path in self.img_paths:

        print(f"images number after filter:{len(self.img_paths)}--thresh:{thresh}")

    def get_binary_classification_label(self, annotation_method, thresh):
        '''

        :param annotation_method: optional -[deep_face_score, facenet_score, sphereface_score, insightface_score, insight_mobilenet_score]
        :param thresh: compared to this value to generate according label[0, 1]
        :return:
        '''
        self.binary_label = {}
        posi_sample = 0
        nega_sample = 0
        for image_path in self.img_paths:
            image_name = image_path.split('\\')[-1]
            self.binary_label[image_name] = {}
            self.binary_label[image_name][annotation_method] = 1 if self.annotation[image_name][
                                                                        annotation_method] >= thresh else 0
            if self.binary_label[image_name][annotation_method] == 1.0:
                posi_sample += 1
            elif self.binary_label[image_name][annotation_method] == 0:
                nega_sample += 1
        print(
            f"When thresh ={thresh} the positive sample contains {posi_sample} negative sample contains {nega_sample}")

    def add_image(self, paths):
        for path in paths:
            self.img_paths.append(path)
    def shuffle(self):
        total_num = len(self.img_paths)
        index = [i for i in range(total_num)]
        random.shuffle(index)
        shuffle_paths = [self.img_paths[i] for i in index ]
        self.img_paths = shuffle_paths
        print("shuffle completely!")
def get_image_path(root_path, valid_ratio=0.1):
    train_paths = []
    valid_paths = []
    test_paths = []
    human_number = len(os.listdir(root_path))
    train_number = int(human_number *(1-2*valid_ratio))
    paths = os.listdir(root_path)
    index = 0
    for file_name in paths[0:train_number]:
        file_path = os.path.join(root_path, file_name)
        for img_name in os.listdir(file_path):
            img_path = os.path.join(file_path, img_name)
            train_paths.append(img_path)
    for file_name in paths[train_number:human_number]:
        file_path = os.path.join(root_path, file_name)
        for img_name in os.listdir(file_path):
            img_path = os.path.join(file_path, img_name)
            if index%2==0:
                valid_paths.append(img_path)
            else:
                test_paths.append(img_path)
            index+=1
    # length = len(img_paths)
    # gap = int(1 / valid_ratio)
    # valid_index = [i for i in range(98232, length, 2)]
    # test_index = [i for i in range(98233, length, 2)]
    # train_index = [i for i in range(0, 98232)]
    # train_path = [img_paths[i] for i in train_index]
    # valid_path = [img_paths[i] for i in valid_index]
    # test_path = [img_paths[i] for i in test_index]
    print("train_number:", len(train_paths))
    print("valid_number:", len(valid_paths))
    print("test_number:", len(test_paths))
    return train_paths, valid_paths, test_paths
def get_all_images(root_path):
    '''
    :param root_path:图片的根路径
    :return: 保存所有图片的路径列表
    '''
    imgs_paths = []
    for file in os.listdir(root_path):

        for img_name in os.listdir(os.path.join(root_path,file)):
            if not(img_name.endswith(".jpg") or img_name.endswith(".png")):
                continue
            else:
                imgs_paths.append(os.path.join(root_path,file,img_name))
    return imgs_paths
def shuffle(ori_list):
    length = len(ori_list)
    index = [i for i in range(length)]
    random.shuffle(index)
    shuffle_list = [ori_list[i] for i in index]
    return shuffle_list


def filter_base_image(paths):
    base_path = []
    for imag_path in paths:
        last = imag_path.split("\\")[-1].split(".")[0].split("_")[-1]
        if last == "0001":
            base_path.append(imag_path)
            # print(imag_path)
    print(f"base images contain{len(base_path)}")
    return base_path
if __name__ =="__main__":
    brisq = BRISQUE()
    path1 =r'D:\Yinyangtao\Image utility research of low-quality human face\Face_recognition\deepface-master-1\samples\human_face\Aaron_Peirsol\Aaron_Peirsol_0001.jpg'
    path = r'D:\Yinyangtao\Image utility research of low-quality human face\Face_recognition\deepface-master-1\samples\human_face\Aaron_Peirsol\Aaron_Peirsol_0001_OEandUE_t12_l4.jpg'
    fea  = brisq.get_feature(path)
    fea1 =brisq.get_feature(path1)
    print(fea)
    print(fea1)
