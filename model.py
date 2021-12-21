import torch
from torch import cuda
import torch.nn as nn
from torchvision import models
from math import sqrt


class Self_Consistency(nn.Module):
    def __init__(self, batch_size):
        super(Self_Consistency, self).__init__()
        self.batch_size = batch_size
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
        self.conv1 = nn.Conv2d(256, 256, 1, 1)
        self.conv2 = nn.Conv2d(256, 256, 1, 1)
        self.act1 = nn.Sigmoid()

        channels_in = resnet34.fc.in_features
        binary_num = 1

        self.avg = nn.Sequential(
            resnet34.layer4,
            resnet34.avgpool,
        )

        self.fc = nn.Linear(channels_in, binary_num)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_extraction(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        # score_volumn = torch.zeros((16, 16, 16, 16)).cuda()
        # for h1 in range(x.shape[2]):
        #     for w1 in range(x.shape[3]):
        #         for h2 in range(x.shape[2]):
        #             for w2 in range(x.shape[3]):
        #                 f1 = x1[:, :, h1, w1].view(1, 128)
        #                 f2 = x2[:, :, h2, w2].view(128, 1)
        #                 score = torch.mm(f1, f2)
        #                 score = score/sqrt(128)
        #                 score = self.act1(score)
        #                 score_volumn[h1, w1, h2, w2] = score
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = x_1.view(-1, 16, 16)
        x_2 = x_2.view(-1, 16, 16)
        # print(x_1.shape)
        # print(x_2.shape)
        score_volumn = torch.matmul(x_1, x_2)
        score_volumn = score_volumn.view(
            self.batch_size, 16, 16, 16, 16)
        score_volumn = self.act1(score_volumn)
        label = self.avg(x)
        label = label.squeeze(3).squeeze(2)
        label = self.fc(label)
        label = self.act2(label)
        return score_volumn, label
