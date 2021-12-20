from torch.serialization import save
from model import Self_Consistency
import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

train_set = data.CustomDataset('data.csv')
train_dataset = DataLoader(
    dataset=train_set, num_workers=4, batch_size=64, shuffle=True,drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
batch_size = 64

model = Self_Consistency(batch_size).to(device)

epochs = 200
lr = 5e-5
weight = 10


optimizer = optim.Adam(model.parameters(), lr=lr)

criterion_1 = nn.BCELoss().to(device)  # loss function for PCL
criterion_2 = nn.CrossEntropyLoss().to(device)


def loss_func(loss1, loss2):
    loss = 10*loss1 + loss2
    return loss


def train(epoch):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # print(model)
    for (i, train_data) in enumerate(train_dataset):
        model.zero_grad()
        in_image = train_data[0].to(device)
        volumn = train_data[1].to(device).squeeze().float()
        label = train_data[2].to(device)
        # print(in_image.shape)
        # print(volumn.shape)
        # print(label.shape)

        in_image = Variable(in_image)
        volumn = Variable(volumn)
        label = Variable(label)

        # # print(in_image, in_image.shape)
        volumn_1, label_1 = model(in_image)
        volumn_1 = volumn_1.float()
        # gt = torch.load('GT_Volume/deepfakes/000_003/0000.pt').float().to(device)
        loss1 = criterion_1(volumn_1, volumn)
        loss2 = criterion_2(label_1, label)
        loss = loss_func(loss1, loss2)
        loss.backward()
        optimizer.step()
        print('=> Epoch[{}]({}/{}): PCL_Loss: {:.4f} CLS_Loss: {:.4f} Total_loss: {:.4f}'.format(
            epoch,
            i,
            len(train_dataset),
            loss1.item(),
            loss2.item(),
            loss.item()
        ))
        # print(loss1, label_1)
        # loss1.backward()
        # print(feature_v.shape)
        # print(feature_v)

def save_checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    

if __name__ == '__main__':
    for epoch in tqdm(range(epochs)):
        train(epoch)
        if epoch % 50 == 0:
            save_checkpoint(epoch)
