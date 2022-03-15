import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
class QualiIndicLoss(nn.Module):
    def __init__(self, classify_layer, align_dim=512+512):
        super(QualiIndicLoss, self).__init__()
        self.classify_layer = classify_layer
        self.align_layer = nn.Sequential(nn.Linear(align_dim, 1), nn.Sigmoid())
        self.mse = nn.MSELoss()
    def forward(self, face_feature, face_id, face_score):
        weight_feature = self.classify_layer.weight[:, face_id].view(face_id.shape[0], -1)
        face_feature = face_feature.view(face_id.shape[0], -1)
        feature_concate = torch.cat([face_feature, weight_feature], dim=1)
        align_score = self.align_layer(feature_concate)
        align_loss = self.mse(align_score, face_score).float()
        #print('align_loss',align_loss)
        return align_loss
class Uq_constrloss(nn.Module):
    def __init__(self,  classify_layer:nn.Module, margin=0.1, threshold = 0.6):
        '''
        :param classify_layer: 人脸识别模型训练最后的分类层
        :param margin: 针对低质量图像而设置的约束度
        :param threshold: 高低质量图像的阈值设置，凡是高于阈值的认为是高质量图像，低于则认为是低质量
        margin > 1- threshold
        '''
        super(Uq_constrloss, self).__init__()
        self.thresh = threshold
        self.classify_layer = classify_layer
        self.margin = margin
        self.lambda_function = lambda q: 1 if q > self.thresh else 1/(q + self.margin)
    def forward(self, face_feature, face_id, face_score):
        '''
        :param face_feature: 人脸的特征
        :param face_id: 人脸的真实类别
        :param face_score: 人脸的可用性质量分数
        :return:质量约束下的损失
        '''
        #self.classify_layer.cuda()
        weight_feature = F.normalize(self.classify_layer.weight[face_id, :].view(face_id.shape[0], -1))
        #print(weight_feature.shape)
        face_feature = F.normalize(face_feature.view(face_id.shape[0], -1))
        f_w_mul = weight_feature * face_feature
        #print(f_w_mul.shape)
        cos_theta = torch.sum(f_w_mul, dim=1).view(-1, 1)
        lambda_value = list(map(self.lambda_function, face_score))
        #print(lambda_value)
        lambda_tensor = torch.Tensor(lambda_value).view(-1, 1).cuda()
        loss = torch.square(1 - torch.mul(cos_theta, lambda_tensor)).sum()/face_id.shape[0]
        return loss
if __name__ == "__main__":
    file_path = r"I:\My_project\Evaluation\FIQA_score_store\FIQA_mine_v2_WDFace.npy"
    file_path_2 = r"I:\My_project\Evaluation\FIQA_score_store\FIQA_mine_v2_WDFace_Train.npy"
    file_1 = np.load(file_path, allow_pickle=True)
    file_2 = np.load(file_path_2, allow_pickle=True)
    file = np.concatenate([file_1, file_2],axis=0)
    np.save(r"I:\My_project\Evaluation\FIQA_score_store\FIQA_mine_v2_DDFace.npy", file)
