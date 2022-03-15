# import  torch.nn as nn
#
# from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
import torch
import numpy as np
import gc
import torch.nn as nn
import torch.nn.functional as F
# from feature_process import channel_sum
from  pyIUA.core.config import cfg

def batch_sum_loss(dataloader, model, data_num=12556, batch=256, layer_name="final", mse_lambda=1, rank_lambda=1,label_type = "continuous"):
    criterion = nn.MSELoss()
    predict_result = []
    _,_, _, y_label = dataloader.getbatch(range(0, data_num), label_type = label_type)
    batch_time = data_num // batch
    # y_label  =None
    for j in range(batch_time):
    # for batch in dataloader:
        # x_batch = batch["input"]
        # x_batch_mask = batch["input_mask"]
        # if y_label == None:
        #     y_label = batch["quality_label"]
        # else:
        #     y_label = torch.cat([y_label, batch["quality_label"]], dim=0)
        if j == batch_time - 1:
            # index = [i for i in range(j * batch, data_num)]
            x_batch,x_batch_mask, x_batch_brisque, _ = dataloader.getbatch(range(j * batch, data_num), label_type=label_type)
        else:
            # index = [i for i in range(j * batch, (j + 1) * batch)]
            x_batch,x_batch_mask, x_batch_brisque, _ = dataloader.getbatch(range(j * batch, (j + 1) * batch), label_type=label_type)

        #y_label = batch["quality_label"]
        with torch.no_grad():
            # prior_feature =None
            # if cfg.TRAIN.prior =="brisque":
            #     prior_feature = database.get_brisque_batch(index)
            predict_output = model(x_batch.cuda(), x_batch_mask.cuda())#,x_batch_mask
                                   # feature_layer=layer_name, label_type=label_type,\
                                   # prior =cfg.TRAIN.prior, quality_feature = prior_feature.cuda() if prior_feature!=None else None)
            predict_result = predict_result + list(predict_output.cpu())
            # gc.collect()
    predict_result = torch.reshape(torch.from_numpy(np.squeeze(predict_result).astype(float)), [-1, 1])
    y_label = torch.reshape(y_label, [-1, 1])
    # print("predicted.shape:", predict_result.shape)
    if label_type =="continuous":
        mse_loss = mse_lambda * criterion(predict_result, y_label) ** 0.5
        rank_Loss = rank_lambda * rank_loss(predict_result, y_label)
        total_loss = mse_loss + rank_Loss
    elif label_type =="binary":
        total_loss = criterion(predict_result, y_label)
        mse_loss = None
        rank_Loss = None
    return mse_loss, total_loss
def compare(former, latter):
    return -1 if former>=latter else 1
def rank_loss( target, predict):
    """

    :param target: tensor [number, 1]
    :param predict: tensor[number, 1]
    :return:
    """
    constant = 1e-6
    relu = nn.ReLU(inplace= True)
    number = target.shape[0]
    temp_loss = 0
    for i in range(number-1):
        temp_loss += relu(compare(target[i],target[i+1]) * (predict[i] - predict[i+1]-constant))
    temp_loss = temp_loss/number
    # print("rank loss:", temp_loss)
    return temp_loss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.mse = nn.MSELoss()
    def forward(self, output1, output2, label):
        output1 = output1.view(output1.shape[0],-1)
        output2 = output2.view(output2.shape[0], -1)
        output1 = F.normalize(output1)
        output2 = F.normalize(output2)
        distance = torch.sum(output1.mul(output2),dim=1)
        loss_contrastive = self.mse(distance,label)
        # euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        # loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
        #                               (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive
