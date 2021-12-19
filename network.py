import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models


class Backbone_Resnet34(nn.Module):
    def __init__(self):
        super(Backbone_Resnet34, self).__init__()
        self.model = models.resnet34(pretrained=True)
        # self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth'))
        # self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        return x
