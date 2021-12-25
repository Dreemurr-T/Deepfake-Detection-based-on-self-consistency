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

from torch.utils.tensorboard import SummaryWriter

total_set = data.CustomDataset('data.csv')
# print(len(total_set))
batch_size = 32
train_set, val_set = torch.utils.data.random_split(total_set, [5551, 448])
train_dataset = DataLoader(
    dataset=train_set, num_workers=4, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataset = DataLoader(
    dataset=val_set, num_workers=4, batch_size=batch_size, shuffle=False, drop_last=True)

# print(len(train_dataset),len(val_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)


model = Self_Consistency(batch_size).to(device)

epochs = 150
lr = 5e-5
weight = 10  # control loss function

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

criterion_1 = nn.BCELoss().to(device)  # loss function for PCL
criterion_2 = nn.BCELoss().to(device)  # loss function for CLS

best_acc = 0  # used for saving best model


def loss_func(loss1, loss2):
    loss = weight*loss1 + loss2
    return loss


def train(epoch):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # print(model)
    PCL_Loss = 0
    CLS_Loss = 0
    LOSS = 0
    for (i, train_data) in enumerate(train_dataset):
        # model.zero_grad()
        optimizer.zero_grad()
        in_image = train_data[0].to(device)
        volumn = train_data[1].to(device).squeeze().float()
        label = train_data[2].to(device).float()

        # label = label.reshape(label.shape[0],1)
        # print(label)
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
        # print(label_1)
        loss1 = criterion_1(volumn_1, volumn)
        loss2 = criterion_2(label_1, label)
        loss = loss_func(loss1, loss2)
        PCL_Loss += loss1.item()
        CLS_Loss += loss2.item()
        LOSS += loss.item()
        loss.backward()
        optimizer.step()

    print('=> Epoch[{}]: PCL_Loss: {:.4f} CLS_Loss: {:.4f} train_loss: {:.4f}'.format(
        epoch,
        PCL_Loss/len(train_dataset),
        CLS_Loss/len(train_dataset),
        LOSS/len(train_dataset)
    ))
    scheduler.step()


def validation(epoch):
    val_acc = []
    acc_sum = 0
    model.eval()
    with torch.no_grad():
        for (i, val_data) in enumerate(val_dataset):
            in_image = val_data[0].to(device)
            volumn = val_data[1].to(device).squeeze().float()
            label = val_data[2].to(device).float()
            in_image = Variable(in_image)
            volumn = Variable(volumn)
            label = Variable(label)
            volumn_1, label_1 = model(in_image)
            label_1 = label_1.squeeze().float()
            volumn_1 = volumn_1.float()
            loss1 = criterion_1(volumn_1, volumn)
            loss2 = criterion_2(label_1, label)
            loss = loss_func(loss1, loss2)
            label_1 = torch.round(label_1)
            acc = 0
            for index in range(label_1.shape[0]):
                if label[index] == label_1[index]:
                    acc += 1
            val_acc.append(acc)
        for acc in val_acc:
            acc_sum += acc
        acc_sum /= 448
        print('Validation accuracy: {:.4f}'.format(acc_sum))
    return acc_sum


def save_checkpoint(val_acc):
    global best_acc
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if best_acc < val_acc:
        best_acc = val_acc
        torch.save(model, 'checkpoint/best_acc_model.pt')


if __name__ == '__main__':
    for epoch in tqdm(range(epochs)):
        train(epoch)
        val_acc = validation(epoch)
        save_checkpoint(val_acc)
    print('Best model accuracy is: {:.4f}'.format(best_acc))
