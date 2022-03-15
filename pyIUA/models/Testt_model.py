import torch.nn as nn
import torch


class Model_test(nn.Module):
    def __init__(self):
        super(Model_test,self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, 2)#10*79*79
        self.conv2 = nn.Conv2d(10,20,3,2)#20*39*39
        self.conv3 = nn.Conv2d(20,30,3, 4)#30*10*10
        self.fc = nn.Linear(3000, 1)
    def forward(self,x,x_):
        x= self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return x


