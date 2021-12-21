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

total_set = data.CustomDataset('data.csv')
train_set,val_set = torch.utils.data.random_split(train_db, [50000, 10000])
train_dataset = DataLoader(
    dataset=train_set, num_workers=4, batch_size=64, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
batch_size = 64

model = Self_Consistency(batch_size).to(device)

epochs = 200
lr = 5e-5
weight = 10  # control loss function

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5)

criterion_1 = nn.BCELoss().to(device)  # loss function for PCL
criterion_2 = nn.BCELoss().to(device)


def loss_func(loss1, loss2):
    loss = weight*loss1 + loss2
    return loss


def train(epoch):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # print(model)
    for (i, train_data) in enumerate(train_dataset):
        # model.zero_grad()
        optimizer.zero_grad()
        in_image = train_data[0].to(device)
        volumn = train_data[1].to(device).squeeze().float()
        label = train_data[2].to(device).float()

        # label = label.reshape(label.shape[0],1)
        print(label)
        in_image = Variable(in_image)
        volumn = Variable(volumn)
        label = Variable(label)
        # print(in_image.shape)
        # print(volumn.shape)
        # print(label.shape)
        # # print(in_image, in_image.shape)
        volumn_1, label_1 = model(in_image)
        label_1 = label_1.squeeze().float()
        volumn_1 = volumn_1.float()
        print(label_1)
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
    scheduler.step()

def save_checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    output_path = 'checkpoint/model_epoch_{}.pth'.format(int(epoch+1))
    torch.save(model, output_path)


if __name__ == '__main__':
    for epoch in tqdm(range(epochs)):
        train(epoch)
        if epoch % 50 == 0 or epoch == epochs-1:
            save_checkpoint(epoch)
