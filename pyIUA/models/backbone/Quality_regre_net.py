import torch.nn as nn
from pyIUA.models.backbone import  model_mobilefaceNet
class Quality_net(nn.Module):

    def __init__(self):
        super(Quality_net, self).__init__()
        self.backbone = model_mobilefaceNet.MobileFaceNet([112, 112], 512, \
                    output_name = 'GDC', use_type = "Qua")

    def forward(self, x, x_mask=None, x_bri=None):
        x = self.backbone(x)

        return x

