import torch
import torch.nn as nn 
import torch.nn.functional as F

class Backbone(nn.Module):
    
    def __init__(self,embedding_size):
        super(Backbone,self).__init__()
        self.conv1 = nn.Conv2d(1,64,3,padding=1)
        self.bn1   = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64,128,3,padding=1)
        self.bn2   = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128,128,3,padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.fc    = nn.Linear(1152,embedding_size)
 
    def forward(self,x):
        
        # layer 1
        x = self.bn1(F.relu(self.conv1(x)))
        x = F.max_pool2d(x,2,2)
        
        # layer 2
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x,2,2)
        
        # layer 3
        x = self.bn3(F.relu(self.conv3(x)))
        x = F.max_pool2d(x,2,2)
        
        # fc layer
        x = torch.flatten(x,1) 
        x = self.fc(x)
        
        
        return x
