import torch
from torch import cuda
import torch.nn as nn
from torchvision import models
from math import sqrt


class Self_Consistency(nn.Module):
    def __init__(self):
        super(Self_Consistency, self).__init__()
        resnet34 = models.resnet34(pretrained=True)
        self.feature_extraction = nn.Sequential(
            resnet34.conv1,
            resnet34.bn1,
            resnet34.relu,
            resnet34.maxpool,
            resnet34.layer1,
            resnet34.layer2,
            resnet34.layer3
        )
        # self.embed1 = nn.Embedding(1,128)
        self.conv1 = nn.Conv2d(256, 128, 1, 1)
        self.conv2 = nn.Conv2d(256, 128, 1, 1)
        self.act1 = nn.Sigmoid()

        channels_in = resnet34.fc.in_features
        binary_num = 2
        

        self.avg = nn.Sequential(
            resnet34.layer4,
            resnet34.avgpool,
        )

        self.fc = nn.Linear(channels_in, binary_num)
        self.sm = nn.Softmax()

    def forward(self, x):
        x = self.feature_extraction(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        score_volumn = torch.zeros((16, 16, 16, 16)).cuda()
        for h1 in range(x.shape[2]):
            for w1 in range(x.shape[3]):
                for h2 in range(x.shape[2]):
                    for w2 in range(x.shape[3]):
                        f1 = x1[:, :, h1, w1].view(1, 128)
                        f2 = x2[:, :, h2, w2].view(128, 1)
                        score = torch.mm(f1, f2)
                        score = score/sqrt(128)
                        score = self.act1(score)
                        score_volumn[h1, w1, h2, w2] = score
        label = self.avg(x)
        label = torch.squeeze(label)
        label = self.fc(label)
        label = self.sm(label)
        return score_volumn, label
