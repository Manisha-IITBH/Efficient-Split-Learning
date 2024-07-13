from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F

"""
PARAMETER COUNTS
----------------
client front: 28,768
server center_front: 1,592,576
server center_back: 9,574,400
client back: 293,640
frozen: 1,621,344
trainable: 9,868,040
total: 11,489,384
"""

def resnet18():
    m = models.resnet18(pretrained=True)
    return m


class front(nn.Module):
    def __init__(self,):
        super().__init__()
        m = resnet18()
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.act1 = m.relu
        self.pool = m.maxpool

        for p in self.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        # print("Front size 1:",x.size())
        x = self.conv1(x)
        # print("Front size 2:",x.size())
        x = self.bn1(x)
        # print("Front size 3:",x.size())
        x = self.act1(x)
        # print("Front size 4:",x.size())
        x = self.pool(x)
        # print("Front size 5:",x.size())
        return x
    

class center_front(nn.Module):
    def __init__(self,):
        super().__init__()
        m = resnet18()
        self.l1 = m.layer1
        self.l2 = m.layer2

        for p in self.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        # print("Center Front size 1:",x.size())
        x = self.l1(x)
        # print("Center Front size 2:",x.size())
        x = self.l2(x)
        # print("Center Front size 3:",x.size())
        return x
    

class center_back(nn.Module):
    def __init__(self,):
        super().__init__()
        m = resnet18()
        self.l3 = m.layer3
        self.l4 = m.layer4
        self.l5 = m.avgpool
        self.head = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256)
        )

    def freeze(self,epoch,pretrained):
        for p in self.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        # print("Center back size 1:",x.size())
        x = self.l3(x)
        # print("Center back size 2:",x.size())
        x = self.l4(x)
        # print("Center back size 3:",x.size())
        x = self.l5(x)
        # print("Center back size 4:",x.size())
        feat = torch.flatten(x, start_dim=1)
        # print("Center back size 5:", feat.size())
        feat = self.head(feat)
        # print("Center back size 6:", feat.size())
        feat = F.normalize(feat, dim=1)
        # print("Center back size 7:", feat.size())
        return feat
    

class back(nn.Module):
    def __init__(self,):
        super().__init__()
        self.fl = nn.Flatten()
        self.fc = nn.Linear(512, 10)
        
    def forward(self, x):
        # x = self.fl(x)
        # x = self.fc(x)
        # print("Client Back: ",x.size())
        return x
    
# Instantiate the model and print its architecture

m = resnet18()
print(m)

front_model = front()
print("Front Model:")
print(front_model)

center_front_model = center_front()
print("\nCenter Front Model:")
print(center_front_model)

center_back_model = center_back()
print("\nCenter Back Model:")
print(center_back_model)

back_model = back()
print("\nBack Model:")
print(back_model)