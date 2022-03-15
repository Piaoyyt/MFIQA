# from .backbone.net_structure import *
# import torch.nn as nn
from .backbone import  get_backbone
from ..core.config import cfg
from .backbone.net_structure import *
import torch.nn.functional as F
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
# from torch.autograd import Variable
# class Quality_predict(nn.Module):
#     def __init__(self):
#         super(Quality_predict,self).__init__()
#     def forward(self,output1,output2):
#         distance = F.pairwise_distance(output1, output2)
#         return distance
#         # output1 = output1.view(output1.shape[0], -1)
#         # output2 = output1.view(output2.shape[0], -1)
#         # output1 = F.normalize(output1)
#         # output2 = F.normalize(output2)
#         # quality_label = 1 - output1.mm(output2.t())  # 余弦值，真实标签越大，说明原图像应该越偏离掩膜图像，即余弦值越小
#         # return quality_label
class ModelBuilder(nn.Module):

    def __init__(self):
        super().__init__()
#       self.backbone_branch_function = {"Inception-Resnet":self.get_incpetion_resnet_branch_net}#主干网络下的分支网络
        #self.backbone_get_feature ={"Inception-Resnet":self.get_inception_resnet_final_feature}#
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,**cfg.BACKBONE.KWARGS)
#        self.backbone_branch_function[cfg.BACKBONE.TYPE]()#生成分支网络模型
        self.tailnet =[]#nn.Sequential(nn.Linear(cfg.Tailnet.layer_set[0], cfg.Tailnet.layer_set[1],bias=False))#[]#最后尾部的全连接层网络
        #self.tail_quality =  Quality_predict()
        for i in range(cfg.Tailnet.layer_number):
            #print("The tailor full connections network constructing...")
            self.tailnet.append(nn.Linear(cfg.Tailnet.layer_set[i], cfg.Tailnet.layer_set[i+1]))
            self.tailnet.append((nn.Dropout(cfg.TRAIN.full_dropout)))
            #self.tailnet.append(nn.ReLU(inplace=True))
            #self.tailnet.append(nn.BatchNorm1d(cfg.Tailnet.layer_set[i+1]))
            # if len(cfg.Tailnet.layer_set)==2 :#and cfg.Dataset.label_normalize
            #     print(f"{i}----Sigmoid constructing...")
            #     #self.tailnet.append(nn.Sigmoid())
            # else:
            #     print(f"{i}----ReLU")
            #     self.tailnet.append(nn.ReLU(True))
        #     #self.tailnet.append(nn.Dropout(cfg.TRAIN.full_dropout))
        self.tailnet =nn.Sequential(*self.tailnet)
        self.f2 = nn.Sequential(nn.Linear(32 , 1) )
        self.sigmoid = nn.Sigmoid()
        #self.full_dropout = cfg.TRAIN.full_dropout
    # def get_feature_distance(self,output1,output2):
    #     output1 = output1.view(output1.shape[0], -1)
    #     output2 = output2.view(output2.shape[0], -1)
    #     output1 = F.normalize(output1)
    #     output2 = F.normalize(output2)
    #     distance = 1-torch.sum(output1.mul(output2), dim=1)
    #     # loss_contrastive = self.mse(distance, label)
    #     return distance
    # def get_incpetion_resnet_branch_net(self):
    #     self.low_level1 = Block1(32, 16, 1, [5, 3], [2, 2])#for inut image 160x160 ,each get 324 dimension vector
    #     self.low_level2 = Block1(32, 16, 1, [3, 3], [2, 2])
    #     self.low_level3 = Block1(64, 16, 1, [3, 3], [2, 2])
    # def get_inception_resnet_final_feature(self, x, feature_layer="final"):
    #     if feature_layer == "final":
    #         x = self.backbone(x)#512
    #     elif feature_layer == 15:
    #         x = nn.Sequential(*list(self.backbone.children())[0:14])(x)
    #         x = torch.squeeze(x)#1792
    #     elif feature_layer =="high_integrated":
    #         high_1 = torch.squeeze(nn.Sequential(*list(self.backbone.children())[0:14])(x))#get 1792 dimension vector
    #         high_2 = self.backbone(x)#get 512 dimensions vector
    #         x = torch.cat((high_1, high_2), axis =1)#1792+512
    #     elif feature_layer == "low_integrated":
    #         low_1 = nn.Sequential(list(self.backbone.children())[0])(x)
    #         low_2 = nn.Sequential(*list(self.backbone.children())[:2])(x)
    #         low_3 = nn.Sequential(*list(self.backbone.children())[:3])(x)
    #         low_1_flatten = self.low_level1(low_1)
    #         low_2_flatten = self.low_level2(low_2)
    #         low_3_flatten = self.low_level3(low_3)
    #         x = torch.cat((low_1_flatten, low_2_flatten, low_3_flatten), axis=1)#get 972dimensions vector
    #     elif feature_layer =="low3_high_1":#前三低层经过卷积得到的特征拼接与高层特征级联
    #         low_1 = nn.Sequential(list(self.backbone.children())[0])(x)
    #         low_2 = nn.Sequential(*list(self.backbone.children())[:2])(x)
    #         low_3 = nn.Sequential(*list(self.backbone.children())[:3])(x)
    #         high_1 = nn.Sequential(*list(self.backbone.children())[0:14])(x)
    #         high_1 = torch.squeeze(high_1)
    #         low_1_flatten = self.low_level1(low_1)
    #         low_2_flatten = self.low_level2(low_2)
    #         low_3_flatten = self.low_level3(low_3)
    #         low_feature = torch.cat((low_1_flatten, low_2_flatten, low_3_flatten), axis=1)
    #         x = torch.cat((low_feature, high_1), axis=1)#972+1792
    #     elif feature_layer =="low2_high_1":#前两低层经过卷积得到的特征拼接与高层级联
    #         low_1 = nn.Sequential(list(self.backbone.children())[0])(x)
    #         low_2 = nn.Sequential(*list(self.backbone.children())[:2])(x)
    #         high_1 = nn.Sequential(*list(self.backbone.children())[0:14])(x)
    #         high_1 = torch.squeeze(high_1)
    #         low_1_flatten = self.low_level1(low_1)
    #         low_2_flatten = self.low_level2(low_2)
    #         x = torch.cat((low_1_flatten, low_2_flatten, high_1), axis =1)#648+1792
    #     return  x
    # def show_model_details(self):
    #     print("本次选取的主干网络是：",cfg.BACKBONE.TYPE)
    #     details_dic={"Inception-Resnet":{"low_integrated":'低三层特征融合→972维度',
    #                                      "low3_high_1":'低三层和高层1792维度特征融合→972+1792维度',
    #                                      "low2_high_1":'低前两层和高层1792维度特征融合→648+1792维度',
    #                                      "high_integrated":'高两层特征融合→1792+512维度',
    #                                      15:'高层1792维度',
    #                                      "final":'最后一层512*2维度'}}
    #     print(details_dic[cfg.BACKBONE.TYPE][cfg.TRAIN.layer])
    #     if cfg.TRAIN.prior !=None:
    #         print("加入手工特征：{0},其维度为{1}".format(cfg.TRAIN.prior,cfg.TRAIN.prior_dimension))
    #     for index,layer in enumerate(cfg.Tailnet.layer_set):
    #         print(f"the {index}th full connection layer nodes number: {layer}\n")
    # def forward_once(self,x,feature_layer="final",
    #             label_type="continuous",prior=None,
    #             quality_feature =None):
    #     x = self.backbone_get_feature[cfg.BACKBONE.TYPE](x=x, feature_layer=feature_layer)
    #     return x
    def forward(self, x, x_mask, x_brisque=None):
                # feature_layer="final",
                # label_type="continuous",prior=None,
                # quality_feature =None):
        # x = torch.Variable(x,requires_grad=True)
        # x = torch.Variable(x_mask, requires_grad=True)
        x = self.backbone(x)
        #x = self.backbone_get_feature[cfg.BACKBONE.TYPE](x=x, feature_layer=feature_layer)
        x_mask = self.backbone(x_mask)
        # #return x,x_mask
        # score = self.get_feature_distance(x, x_mask)
        x_out  = torch.cat([x,x_mask],-1)
        #x_out = self.get_feature_distance(x,x_mask)
        x  = self.tailnet(x_out)
        #x = torch.cat([x, x_brisque], -1)
        x = self.f2(x)
        # # x.requires_grad = True
       # print("融合人脸图像和掩膜图像的特征后的维度为",x_out.shape)
        # #全连接层
        # if prior =="brisque":
        #     # quality_feature.cuda()
        #     x= torch.cat((x_out,quality_feature),1)
        # x_out = self.tailnet(x_out)
        # x_out = torch.sigmoid(x_out)
        # if label_type == "continuous" and cfg.Dataset.label_normalize:
        #     x = 100 * x
        # elif label_type=="continuous" and not cfg.Dataset.label_normalize:
        #     x= x
        # elif label_type == "binary":
        #     x = torch.sigmoid(x)
        return x

