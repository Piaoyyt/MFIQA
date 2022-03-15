from abc import ABC

import torch
import torch.nn as nn
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1


class Block1(nn.Module):
    def __init__(self, in_planes, inter_media_planes, out_planes, kernel_sizes, strides, padding=0):
        super(Block1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, inter_media_planes,
                                             kernel_size=kernel_sizes[0], stride=strides[0],
                                             padding=padding, bias=False),
                                   nn.BatchNorm2d(num_features=inter_media_planes), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(inter_media_planes, out_planes,
                                             kernel_size=kernel_sizes[1], stride=strides[1],
                                             padding=padding, bias=False), nn.BatchNorm2d(out_planes), nn.ReLU(True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        return x


class low_level_net(nn.Module):
    def __init__(self, extracted_layer=1):
        super(low_level_net, self).__init__()

        self.feature_layer = extracted_layer
        if self.feature_layer == 1:
            self.net_backbone = Block1(32, 16, 1, [5, 3], [2, 2])
        elif self.feature_layer == 2:
            self.net_backbone = Block1(32, 16, 1, [3, 3], [2, 2])
        elif self.feature_layer == 3:
            self.net_backbone = Block1(64, 16, 1, [3, 3], [2, 2])

    def forward(self, x):
        x = self.net_backbone(x)
        return x


class ensemble_net(nn.Module):
    def __init__(self, layer=[1, 2, 3], full_1=972, full_2=512, dropout=0):
        super(ensemble_net, self).__init__()
        self.net1 = low_level_net(layer[0])
        self.net2 = low_level_net(layer[1])
        self.net3 = low_level_net(layer[2])
        self.full1 = nn.Sequential(nn.Linear(full_1, full_2), nn.ReLU(True), nn.BatchNorm1d(num_features=full_2),
                                   nn.Dropout(dropout))
        self.full2 = nn.Sequential(nn.Linear(full_2, 1), nn.ReLU(True))

    def forward(self, layer1_feature, layer2_feature, layer3_feature, choice="final_score"):
        layer1_flatten = self.net1(layer1_feature)
        layer2_flatten = self.net2(layer2_feature)
        layer3_flatten = self.net3(layer3_feature)
        concatenated_feature = torch.cat((layer1_flatten, layer2_flatten, layer3_flatten), axis=1)
        if choice == "low_level_feature_1":
            return concatenated_feature
        x = self.full1(concatenated_feature)
        if choice == "low_level_feature_2":
            return x
        x = self.full2(x)
        x = 100 * torch.sigmoid(x)
        if choice == "final_score":
            return x


class two_full_net(nn.Module):
    def __init__(self, input_dim=(1792 + 512), hidden_dim=512, dropout=0):
        super(two_full_net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(True),
                                    nn.BatchNorm1d(num_features=hidden_dim), nn.Dropout(dropout))
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, 1), nn.ReLU(True), nn.Dropout(dropout))

    def forward(self, input_1, input_2):
        input_1 = torch.flatten(input_1, start_dim=1)
        input_2 = torch.flatten(input_2, start_dim=1)
        # print(input_2.shape)
        # print(input_1.shape)
        x = torch.cat((input_1, input_2), axis=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = 100 * torch.sigmoid(x)
        return x


class two_full_net_3(nn.Module):
    def __init__(self, input_dim=(972 + 1792 + 512), hidden_dim=512, dropout=0):
        super(two_full_net_3, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(True),
                                    nn.BatchNorm1d(num_features=hidden_dim), nn.Dropout(dropout))
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, 1), nn.ReLU(True), nn.Dropout(dropout))

    def forward(self, input_1, input_2, input3):
        input_1 = torch.flatten(input_1, start_dim=1)
        input_2 = torch.flatten(input_2, start_dim=1)
        input3 = torch.flatten(input3, start_dim=1)
        # print(input_2.shape)
        # print(input_1.shape)
        x = torch.cat((input_1, input_2, input3), axis=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = 100 * torch.sigmoid(x)
        return x


class high_level_net(nn.Module):
    def __init__(self, input_dim=(1792 + 512), hidden_dim=512, dropout=0):
        super(two_full_net, self).__init__()
        self.layer1 = nn.Sequential(nn.BatchNorm1d(num_features=input_dim), nn.Linear(input_dim, hidden_dim),
                                    nn.ReLU(True), nn.Dropout(dropout))
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, 1), nn.ReLU(True), nn.Dropout(dropout))

    def forward(self, input_1, input_2):
        input_1 = torch.flatten(input_1, start_dim=1)
        input_2 = torch.flatten(input_2, start_dim=1)
        # print(input_2.shape)
        # print(input_1.shape)
        x = torch.cat((input_1, input_2), axis=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = 100 * torch.sigmoid(x)
        return x


class two_full_net_1(nn.Module):
    def __init__(self, input_dim=2592, hidden_dim=512, dropout=0):
        super(two_full_net_1, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(True),
                                    nn.Dropout(dropout))
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, 1), nn.ReLU(True), nn.Dropout(dropout))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.layer1(x)
        x = self.layer2(x)

        x = 100 * torch.sigmoid(x)
        return x


class fine_tune_net(nn.Module):
    def __init__(self, net_type="vgg_resnet_pretrained", full_connection=[512, 1], pre_dropout=0, full_dropout=0):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(fine_tune_net, self).__init__()
        self.pre_dropout = pre_dropout
        self.full_dropout = full_dropout
        if net_type == "vgg_resnet_pretrained":
            self.backbone = InceptionResnetV1(pretrained="vggface2", dropout_prob=self.pre_dropout)
        elif net_type == "vgg_resnet":
            self.backbone = InceptionResnetV1(dropout_prob=self.pre_dropout)
        elif net_type =="vgg_resnet_low_3":
            self.backbone = InceptionResnetV1(pretrained="vggface2", dropout_prob=self.pre_dropout)
            self.net1 = low_level_net(1)
            self.net2 = low_level_net(2)
            self.net3 = low_level_net(3)

        self.full_layer = nn.Sequential(nn.Linear(full_connection[0], full_connection[1]),
                                        nn.Dropout(self.full_dropout))

    def forward(self, x, feature_layer="final", label_type="continuous",prior=None,quality_feature =None):
        if feature_layer == "final":
            x = self.backbone(x)
        elif feature_layer == 15:
            x = nn.Sequential(*list(self.backbone.children())[0:14])(x)
            x = torch.squeeze(x)
        elif feature_layer =="high_integrated":
            high_1 = torch.squeeze(nn.Sequential(*list(self.backbone.children())[0:14])(x))
            high_2 = self.backbone(x)
            x = torch.cat((high_1, high_2), axis =1)
        elif feature_layer == "low_integrated":
            low_1 = nn.Sequential(list(self.backbone.children())[0])(x)
            low_2 = nn.Sequential(*list(self.backbone.children())[:2])(x)
            low_3 = nn.Sequential(*list(self.backbone.children())[:3])(x)
            low_1_flatten = self.net1(low_1)
            low_2_flatten = self.net2(low_2)
            low_3_flatten = self.net3(low_3)
            x = torch.cat((low_1_flatten, low_2_flatten, low_3_flatten), axis=1)
            # self.full_layer = nn.Sequential(nn.Linear(1792, 1), nn.Dropout(self.dropout))
        elif feature_layer =="low3_high_1":#前三低层经过卷积得到的特征拼接与高层特征级联
            low_1 = nn.Sequential(list(self.backbone.children())[0])(x)
            low_2 = nn.Sequential(*list(self.backbone.children())[:2])(x)
            low_3 = nn.Sequential(*list(self.backbone.children())[:3])(x)
            high_1 = nn.Sequential(*list(self.backbone.children())[0:14])(x)
            high_1 = torch.squeeze(high_1)
            low_1_flatten = self.net1(low_1)
            low_2_flatten = self.net2(low_2)
            low_3_flatten = self.net3(low_3)
            low_feature = torch.cat((low_1_flatten, low_2_flatten, low_3_flatten), axis=1)
            x = torch.cat((low_feature, high_1), axis=1)
        elif feature_layer =="low2_high_1":#前两低层经过卷积得到的特征拼接与高层级联
            low_1 = nn.Sequential(list(self.backbone.children())[0])(x)
            low_2 = nn.Sequential(*list(self.backbone.children())[:2])(x)
            high_1 = nn.Sequential(*list(self.backbone.children())[0:14])(x)
            high_1 = torch.squeeze(high_1)
            low_1_flatten = self.net1(low_1)
            low_2_flatten = self.net2(low_2)
            x = torch.cat((low_1_flatten, low_2_flatten, high_1), axis =1)
        #全连接层
        if prior =="brisque":
            # quality_feature.cuda()
            x= torch.cat((x,quality_feature),1)
        x = self.full_layer(x)
        if label_type == "continuous":
            x = 100 * torch.sigmoid(x)
        elif label_type == "binary":
            x = torch.sigmoid(x)
        return x


class fine_tune_net_2(nn.Module):
    def __init__(self, net_type="vgg_resnet_pretrained", full_connection=[512, 32, 1], pre_dropout=0, full_dropout=0):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(fine_tune_net_2, self).__init__()
        self.pre_dropout = pre_dropout
        self.full_dropout = full_dropout
        # 判断所使用的网络模型
        #预训练的vgg-resnet
        if net_type == "vgg_resnet_pretrained":
            self.backbone = InceptionResnetV1(pretrained="vggface2", dropout_prob=self.pre_dropout)
            self.full_layer = nn.Sequential(nn.Linear(full_connection[0], full_connection[1]), \
                                            nn.BatchNorm1d(full_connection[1]), nn.ReLU(True),
                                            nn.Dropout(self.full_dropout))
            self.full_layer2 = nn.Sequential(nn.Linear(full_connection[1], full_connection[2]), nn.ReLU(True))
        #没有预训练的vgg-resnet
        elif net_type == "vgg_resnet":
            self.backbone = InceptionResnetV1(dropout_prob=self.pre_dropout)
            self.full_layer = nn.Sequential(nn.Linear(full_connection[0], full_connection[1]), \
                                            nn.BatchNorm1d(full_connection[1]), nn.ReLU(True),
                                            nn.Dropout(self.full_dropout))
            self.full_layer2 = nn.Sequential(nn.Linear(full_connection[1], full_connection[2]), nn.ReLU(True))
        #vgg-resnet前面低层的网络
        elif net_type =="vgg_resnet_low_3":
            self.backbone = InceptionResnetV1(pretrained="vggface2", dropout_prob=self.pre_dropout)
            self.net1 = low_level_net(1)
            self.net2 = low_level_net(2)
            self.net3 = low_level_net(3)
            self.full1 = nn.Sequential(nn.Linear(972, 1), nn.ReLU(True),
                                       nn.Dropout(full_dropout))
            # self.full2 = nn.Sequential(nn.Linear(full_2, 1), nn.ReLU(True))



    def forward(self, x, feature_layer="final", label_type="continuous"):
        if feature_layer == "final":
            x = self.backbone(x)
        elif feature_layer == 15:
            x = nn.Sequential(*list(self.backbone.children())[0:14])(x)
            x = torch.squeeze(x)
            # self.full_layer = nn.Sequential(nn.Linear(1792, 1), nn.Dropout(self.dropout))
        elif feature_layer =="high_integrated":
            high_1 = torch.squeeze(nn.Sequential(*list(self.backbone.children())[0:14])(x))
            high_2 = self.backbone(x)
            x = torch.cat((high_1, high_2), axis =1)
        elif feature_layer =="low_integrated":
            low_1 = nn.Sequential(list(self.backbone.children())[0])(x)
            low_2 = nn.Sequential(*list(self.backbone.children())[:2])(x)
            low_3 = nn.Sequential(*list(self.backbone.children())[:3])(x)
            low_1_flatten = self.net1(low_1)
            low_2_flatten = self.net2(low_2)
            low_3_flatten = self.net3(low_3)
            x = torch.cat((low_1_flatten, low_2_flatten, low_3_flatten), axis=1)

        elif feature_layer =="low3_high_1":
            low_1 = nn.Sequential(list(self.backbone.children())[0])(x)
            low_2 = nn.Sequential(*list(self.backbone.children())[:2])(x)
            low_3 = nn.Sequential(*list(self.backbone.children())[:3])(x)
            high_1 = nn.Sequential(*list(self.backbone.children())[0:14])(x)
            high_1 = torch.squeeze(high_1)
            low_1_flatten = self.net1(low_1)
            low_2_flatten = self.net2(low_2)
            low_3_flatten = self.net3(low_3)
            low_feature = torch.cat((low_1_flatten, low_2_flatten, low_3_flatten), axis=1)
            x = torch.cat((low_feature, high_1), axis=1)
        elif feature_layer =="low2_high_1":#前两低层经过卷积得到的特征拼接与高层级联
            low_1 = nn.Sequential(list(self.backbone.children())[0])(x)
            low_2 = nn.Sequential(*list(self.backbone.children())[:2])(x)
            high_1 = nn.Sequential(*list(self.backbone.children())[0:14])(x)
            high_1 = torch.squeeze(high_1)
            low_1_flatten = self.net1(low_1)
            low_2_flatten = self.net2(low_2)
            x = torch.cat((low_1_flatten, low_2_flatten, high_1), axis =1)
        #全连接层


        x = self.full_layer(x)
        x = self.full_layer2(x)
        if label_type == "continuous":
            x = 100 * torch.sigmoid(x)
        elif label_type == "binary":
            x = torch.sigmoid(x)
        return x
    # def load_pretrained(self, net_type ="vgg_resnet"):
    #     if net_type == "vgg_resnet":
    #         self.backbone = InceptionResnetV1(pretrained="vggface2")

# class binary_prediction(nn.Module):
#     def __init__(self, net_type="vgg_resnet_pretrained", full_connection=[512, 1], dropout=0):
#         super(binary_prediction, self).__init__()
# class brisque_feature():
#     def __init__(self):
