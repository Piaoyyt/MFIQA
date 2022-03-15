import torch
import numpy as np
from torch.autograd import Variable
from torch.nn.functional import cross_entropy,nll_loss
import torch.nn.functional as F
import torch.nn as nn
import math
def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)
def mlambda(x):
    return  8*x**4-8*x**2+1
class MFaceLinear(torch.nn.Module):
    '''
    The Linear layer that reflect the Face Feature on the Class Space
    '''
    def __init__(self,input_features,out_features, m=4, phiflag=True) :
        '''
        :param input_features:The Face Feature Dimension 'd'
        :param out_features: The Face ID Categories 'C'
        '''
        super(MFaceLinear,self).__init__()
        self.input_features = input_features
        self.out_features = out_features
        self.weight =torch.nn.Parameter(torch.Tensor(input_features,out_features))
        self.weight.data.uniform_(-1,1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m
        self.mlambda = [None,
                        None,
                        None,
                        None,
                        mlambda
                        ]
        self.phiflag = phiflag
    def forward(self, input):
        """
        :param input:The Human Face Feature Batch [B,d]
        :return:
        """
        x = input
        w = self.weight
        w_norm = w.renorm(2,1,1e-5).mul(1e5)#Normalize along the dim=1
        x_len = x.pow(2).sum(1).pow(0.5)
        w_len = w_norm.pow(2).sum(0).pow(0.5)
        cos_theta  =x.mm(w_norm)
        cos_theta = cos_theta/x_len.view(-1,1)#/w_len.view(-1,1)
        cos_theta =cos_theta.clamp(-1,1)
        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * x_len.view(-1,1)
        phi_theta = phi_theta * x_len.view(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)
       # return cos_theta

class MFaceLoss(torch.nn.Module):

    def __init__(self,fia_model=None):
        super(MFaceLoss,self).__init__()
        self.gamma = 0
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self,input, target, quality):
        '''
        :param input: [B,C]--tensor, the Predicted the Cosine Theta Matrix
        :param target: [B,1]--tensor,the Face ID,[[0],[4],[8],...[2]]
        :param quality:[B,1]--tensor,the Face Qualtiy Score predicted by the MFIQ
        :return:
        '''
        quality = quality.float()
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)
        #quality_indicator = 1-quality
        #output[index] += quality_indicator
        logpt = F.log_softmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        loss_quality = torch.square(output[index] - quality).sum() / output.shape[0]
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean() + loss_quality
        #input[id_index] = input[id_index].cos()
        #logpt = log_softmax(input)
        # logpt = logpt.gather(1,target)
        # logpt = logpt.view(-1)#the log(probablity) the Face belonging the true ID
        # pt = Variable(logpt.data.exp())#the probablity
        # loss = - logpt
        # loss = loss.mean()
        #soft_max = nn.Softmax(dim=1)
        #input = torch.log(soft_max(input))
        #target = torch.squeeze(target)
        #loss = nll_loss(input,target,reduction="mean")
        return loss
if __name__=="__main__":
    m = MFaceLoss()
    np.random.seed(1)
    in_ = torch.from_numpy(np.random.rand(6,5))
    in_2 = torch.from_numpy(np.random.rand(6,5))
    target_ = torch.tensor([1,2,4,0,3,2])
    quality_ = torch.tensor([0.3,0.1,1,0.9,0.1,0.6])
    loss = m((in_, in_2),target_,quality_)
    print(loss)
