# 单独训练content，用ava训练
# 单独训练attribute，用aadb训练

from __future__ import print_function, division
import os
import torch
from torch import nn
import pandas as pd
from skimage import transform
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from PIL import Image
import time
import math
import copy

import torch.optim as optim
from torch.autograd import Variable
from torchvision import models

import warnings
warnings.filterwarnings("ignore")
import random
# import cv2
from scipy.stats import spearmanr
use_gpu = True
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 构建Style的测试集数据集
class ImageLabelStyleDataset(Dataset):
    """Images Style dataset."""

    def __init__(self, csv_file, root_dir, flag, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            第1列：图片ID，第2列 label：多标签，属性14个
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images_frame = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform
        self.flag = flag

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        try:
            if self.flag == 1:
                img_name = str(os.path.join(self.root_dir, (str(self.images_frame.iloc[idx, 0])+'.jpg')))  # .iloc[行：列]
            else:
                img_name = str(os.path.join(self.root_dir, str(self.images_frame.iloc[idx, 0])))
            im = Image.open(img_name).convert('RGB')
            if im.mode == 'P':
                im = im.convert('RGB')
            image = np.asarray(im)
            #image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
            label = self.images_frame.iloc[idx, 1:]  # 第1列之后所有列，内容和属性的训练集，1个标签，而内容测试集1个标签，但是属性测试集第二列是14元素的列表
            sample = {'image': image, 'label': label}

            if self.transform:
                sample = self.transform(sample)
            return sample
        except Exception as e:
            pass  # print("数据集__getitem__ is wrong!")


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        return {'image': image, 'label': label}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, 
                left: left + new_w]

        return {'image': image, 'label': label}


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() < self.p:
            image = np.flip(image, 1)
            # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        im = image /1.0  # / 255
        im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        im[:, :, 1] = (image[:, :, 1] - self.means[1]) / self.stds[1]
        im[:, :, 2] = (image[:, :, 2] - self.means[2]) / self.stds[2]
        image = im
        return {'image': image, 'label': label}



# 属性模型
class StyleModel(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):  # 1

        super(StyleModel, self).__init__()

        self.fc1_1 = nn.Linear(inputsize, 2048)
        self.bn1_1 = nn.BatchNorm1d(2048)
        self.drop_prob = (1 - keep_probability)
        self.relu1_1 = nn.PReLU()
        self.drop1_1 = nn.Dropout(self.drop_prob)
        self.fc2_1 = nn.Linear(2048, 1024)
        self.bn2_1 = nn.BatchNorm1d(1024)
        self.relu2_1 = nn.PReLU()
        self.drop2_1 = nn.Dropout(p=self.drop_prob)
        self.fc3_1 = nn.Linear(1024, num_classes)
        self.bn3_1 = nn.BatchNorm1d(num_classes)
        self.tanh = nn.Tanh()
        self.soft = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out_p = self.fc1_1(x)
        out_p = self.bn1_1(out_p)
        out_p = self.relu1_1(out_p)
        out_p = self.drop1_1(out_p)
        out_p = self.fc2_1(out_p)
        out_p = self.bn2_1(out_p)
        out_p = self.relu2_1(out_p)
        out_p = self.drop2_1(out_p)
        out_p = self.fc3_1(out_p)
        # out_p = self.bn3_1(out_p)
        # out_p = self.tanh(out_p)

        return out_p

# 属性模型
class StyleModel_lessFC(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):  # 1

        super(StyleModel_lessFC, self).__init__()

        self.fc1_1 = nn.Linear(inputsize, 1024)
        self.bn1_1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1_1 = nn.PReLU()
        self.drop1_1 = nn.Dropout(self.drop_prob)
        self.fc3_1 = nn.Linear(1024, num_classes)
        self.bn3_1 = nn.BatchNorm1d(num_classes)
        self.tanh = nn.Tanh()
        self.soft = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out_p = self.fc1_1(x)
        out_p = self.bn1_1(out_p)
        out_p = self.relu1_1(out_p)
        out_p = self.drop1_1(out_p)
        out_p = self.fc3_1(out_p)
        out_p = self.bn3_1(out_p)
        out_p = self.tanh(out_p)

        return out_p


# 内容属性联合网络
class Associate_Style_Model(nn.Module):
    #constructor
    def __init__(self, resnet, style_net):
        super(Associate_Style_Model, self).__init__()
        #defining layers in convnet
        self.resnet = resnet
        self.StyleNet = style_net

    def forward(self, x):
        x = self.resnet(x)
        x = self.StyleNet(x)
        return x

# 求SPP相关性
def computeSpearman(dataloader_valid, model):
    ratings = []
    predictions = []
    with torch.no_grad():
        cum_loss = 0
        for batch_idx, data in enumerate(dataloader_valid):
            inputs = data['image']
            batch_size1 = inputs.size()[0]
            labels = data['label'].view(batch_size1, -1)
            if use_gpu:
                try:
                    inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                except:
                    print(inputs, labels)
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs_p = model(inputs)  # 同时算俩
            #sheet = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8])  # ？需要调整，跟咱们就没有关系，操！！！
            #ppp = outputs_p.mul(Variable(sheet.cuda()))
            #out = torch.sum(ppp, 1).view(batch_size1, -1)
            ratings.append(labels.float())
            predictions.append(outputs_p.float())

    ratings = [t.cpu().numpy() for t in ratings]
    predictions = [t.cpu().numpy() for t in predictions]
    ratings_i = np.vstack(ratings)  # 按垂直方向（行顺序）堆叠数组构成一个新的数组
    predictions_i = np.vstack(predictions)

    res = []
    for i in range(ratings_i.shape[1]):
        a = ratings_i[:, i]  # 行求一个个求
        b = predictions_i[:, i]
        sp = spearmanr(a, b)[0]
        res.append(sp)
    sp_mean = np.mean(res)
    return sp_mean

# 训练过程，测试过程
def train_model(model, criterion2, optimizer, lr_scheduler, dataloader_train_style, dataloader_valid_style, num_epochs, best_spp):
    since = time.time()
    train_loss_average = []
    test_loss = []
    best_model = model
    best_loss = 100
    criterion2.cuda()
    model.cuda()
    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # Each epoch has a training and validation phase
        for phase in [ 'val']:  # 'train',
            total_size = 0
            running_loss_s = 0.0
            running_acc_s = 0.0
            if phase == 'train':
                mode = 'train'
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()  # Set model to training mode

                dataloader = dataloader_train_style

                counter = 0
                for idx, data_style in enumerate(dataloader):
                    inputs_s = data_style['image']
                    batch_size1 = inputs_s.size()[0]
                    labels_s = data_style['label'].view(batch_size1, -1)
                    #labels_s = data_style['label']
                    if use_gpu:
                        try:
                            inputs_s, labels_s = Variable(inputs_s.float().cuda()), Variable(labels_s.float().cuda())
                        except:
                            print(inputs_s, labels_s)
                    else:
                        inputs_s, labels_s = Variable(inputs_s), Variable(labels_s)

                    # 同一个模型，灌两次数据，loss回传两次
                    optimizer.zero_grad()
                    outputs_ss = model(inputs_s)  # 属性前向和反向
                    #labels_s = labels_s.squeeze().long()
                    loss2 = criterion2(outputs_ss, labels_s)  # 11--11 求差值
                    prec_s = 0

                    loss2.backward()
                    optimizer.step()

                    if counter % 20 == 0:
                        print("---Epoch is %d, Iteration is %d/%d, Attribute Loss is %f, Attribute acc is %f---" % (
                        epoch, counter, len(dataloader), loss2.item(), prec_s))
                    counter += 1
                    try:
                        running_loss_s += loss2.item()
                        running_acc_s += prec_s
                    except:
                        print('unexpected error, could not calculate loss or do a sum.')
                # print('trying epoch loss')
                epoch_loss_s = running_loss_s / len(dataloader)
                running_acc_s = running_acc_s / len(dataloader)
                print('average {} Loss: {:.4f} '.format(phase, epoch_loss_s))
                print('average {} Attribute acc: {:.4f} '.format(phase, running_acc_s))
            else:
                model.eval()
                mode = 'val'
                dataloader = dataloader_valid_style
                spp = computeSpearman(dataloader, model)
                print(spp)

                counter = 0
                # Iterate over data.
                for idx, data_style in enumerate(dataloader):
                    inputs_s = data_style['image']
                    batch_size1 = inputs_s.size()[0]
                    labels_s = data_style['label'].view(batch_size1, -1)
                    #labels_s = data_style['label']
                    if use_gpu:
                        try:
                            inputs_s, labels_s = Variable(inputs_s.float().cuda()), Variable(labels_s.float().cuda())
                        except:
                            print(inputs_s, labels_s)
                    else:
                        inputs_s, labels_s = Variable(inputs_s), Variable(labels_s)


                    # Set gradient to zero to delete history of computations in previous epoch. Track operations so that differentiation can be done automatically.
                    optimizer.zero_grad()
                    outputs_ss = model(inputs_s)  # 属性前向和反向# 标签得是0-13【14个分类】
                    #labels_s = labels_s.squeeze().long()
                    loss_s = criterion2(outputs_ss, labels_s)  # 11--11 求差值
                    prec_s = 0

                    if counter % 20 == 0:
                        print("---Epoch is %d, Iteration is %d/%d, Attribute Loss is %f, Attribute acc is %f---" % (
                        epoch, counter, len(dataloader), loss_s.item(), prec_s))
                    counter += 1
                    try:
                        running_loss_s += loss_s.item()
                        running_acc_s += prec_s
                    except:
                        print('unexpected error, could not calculate loss or do a sum.')

                # print('trying epoch loss')
                epoch_loss_s = running_loss_s / len(dataloader)
                running_acc_s = running_acc_s / len(dataloader)
                print('average {} Attribute Loss: {:.4f} '.format(phase, epoch_loss_s))
                print('average {} Attribute acc: {:.4f} '.format(phase, running_acc_s))
                # deep copy the model
                if phase == 'val':
                    test_loss.append(epoch_loss_s)
                    spp = computeSpearman(dataloader, model)  # p-value就是我们要的结果
                    print('test spp = %f' % spp)
                    if epoch_loss_s < best_loss:
                        best_loss = epoch_loss_s
                        print('new best loss = %f' % epoch_loss_s)
                    if spp > best_spp:
                        best_spp = spp
                        best_model = copy.deepcopy(model)
                        print('new best Attribute spp = %f' % spp)
                    print('best loss: %f, best spp: %f' % (epoch_loss_s, best_spp))
    # Save model
    torch.save(model.cuda(), 'Attribute_Single_Model_AADB/Attribute_Single_Model_AADB_normalized_inception.pt')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    print('returning and looping back')
    return best_model.cuda(), train_loss_average, test_loss, best_spp

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).double(), 
                'label': torch.from_numpy(np.float64([label])).double()}

def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=1):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate =  0.5**(epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer

def my_collate(batch):
    batch = list(filter (lambda x:x is not None, batch))
    return default_collate(batch)

def calc_auto(num, channels):
    lst = [1, 2, 4, 8, 16, 32]
    return sum(map(lambda x: x ** 2, lst[:num])) * channels

# main 主函数
if __name__ == '__main__':
    model_path = 'Attribute_Single_Model_AADB'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
     # 和服务器联调-------------------
    AADB_style_data_path = r'D:\AWYY\aesthetics\dataset\AADB'
    AADB_style_image_path = r'D:\AWYY\aesthetics\dataset\AADB\datasetImages_originalSize'
    AVA_content_data_path = r'D:\AWYY\aesthetics\dataset\AVA\AVA_dataset\aesthetics_content_lists'
    AVA_content_image_path = r'D:\AWYY\aesthetics\dataset\AVA\images'  # 与 style公用一个文件夹，不过数据集不同

    # AADB_style_data_path = r'/home/wyy2/WYY/AADB'
    # AADB_style_image_path = r'/home/wyy2/WYY/AADB/datasetImages_originalSize'
    # AVA_content_data_path = r'/home/wyy2/WYY/AVA/AVA_dataset/aesthetics_content_lists'
    # AVA_content_image_path = r'/home/wyy2/WYY/AVA/images'  # 与 style公用一个文件夹，不过数据集不同

    # 属性
    transformed_dataset_train_style = ImageLabelStyleDataset(csv_file=os.path.join(AADB_style_data_path, 'imgListTrainRegression.csv'),
                                                    root_dir=AADB_style_image_path,
                                                      flag=0,
                                               transform=transforms.Compose([Rescale(output_size=(350, 350)),
                                                                            RandomCrop(output_size=(299, 299)), 
                                                                              Normalize(), 
                                                                              ToTensor()]))  # Personality.csv

    transformed_dataset_valid_style = ImageLabelStyleDataset(csv_file=os.path.join(AADB_style_data_path, 'imgListTestNewRegression.csv'),
                                                    root_dir=AADB_style_image_path,
                                                    flag=0,
                                               transform=transforms.Compose([Rescale(output_size=(299, 299)),
                                                                             Normalize(),
                                                                            ToTensor()]))  # style_test_label.csv
    bsize = 16  # 和服务器联调-------------------

    dataloader_train_style = DataLoader(transformed_dataset_train_style, batch_size=bsize,
                            shuffle=True, num_workers=0, collate_fn=my_collate, drop_last=True)
    dataloader_valid_style = DataLoader(transformed_dataset_valid_style, batch_size=int(bsize),
                            shuffle=True, num_workers=0, collate_fn=my_collate, drop_last=True)  # 构建时就已经shuffle了


    best_spp = -1
    num_epochs = 20
    for i in range(20):
        # model_ft = models.densenet121(pretrained=True)
        #model_ft = models.inception_v3(pretrained=True)  # 主干预训练模型
        model_ft = models.resnet34(pretrained=True)  # 主干预训练模型
        model_ft.aux_logits = False
        num_ftrs = model_ft.fc.out_features
        net2 = StyleModel(1, 0.5, num_ftrs)  # 美学模型
        net3 = Associate_Style_Model(resnet=model_ft, style_net=net2)
        # net_2 = (torch.load('Personality_Model/FlickrAES_Personality_normalized_resnet18.pt'))

        #device = cuda
        criterion2 = nn.MSELoss()
        criterion2.cuda()
        if torch.cuda.device_count() > 1:
            print("We have", torch.cuda.device_count(), 'gpus，use all，out to gpu 0')

         # 和服务器联调-------------------
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 设置主卡为1，主卡必须参与计算
        # model_ft = nn.DataParallel(net3, device_ids=[0, 3,], output_device=0)  # 用03块gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置本机主卡为0
        model_ft = nn.DataParallel(net3, device_ids=[0,], output_device=0)  # 用本机的0号

        ignored_params = list(map(id, net2.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, 
                             model_ft.parameters())
        optimizer = optim.Adam([
            {'params': base_params}, 
            {'params': net2.parameters(), 'lr': 1e-3}
        ], lr=1e-5)  # , weight_decay=0.00001)


        print('repeat: %d, best acc: %f' %(i, best_spp))
        best_model, train_loss, test_loss, best_spp = train_model(model_ft, criterion2, optimizer, exp_lr_scheduler,
                                                                    dataloader_train_style, dataloader_valid_style,
                                                               num_epochs, best_spp)
        torch.save(best_model.cuda(), 'Attribute_Single_Model_AADB/Attribute_Single_Model_AADB_normalized_inception_best.pt')
