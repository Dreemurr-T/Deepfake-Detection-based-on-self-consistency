from model import Self_Consistency
import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

model = Self_Consistency().to(device)

epochs = 100
lr = 1e-5

optimizer = optim.Adam(model.parameters, lr=lr)

criterion_1 = nn.BCELoss().to(device)  # loss function for PCL
criterion_2 = nn.CrossEntropyLoss().to(device)


def train():

    print(model)
    input_path = 'faces/original/000/0000.png'
    in_image = Image.open(input_path)
    in_image = transforms.ToTensor()(in_image)
    in_image = Variable(torch.unsqueeze(
        in_image, dim=0).float())
    in_image = in_image.to(device)
    # print(in_image, in_image.shape)
    feature_v, label1 = model(in_image)
    feature_v = feature_v.float()
    gt = torch.load('GT_Volume/deepfakes/000_003/0000.pt').float().to(device)
    loss1 = criterion_1(feature_v, gt)
    print(loss1, label1)
    # loss1.backward()
    # print(feature_v.shape)
    # print(feature_v)


if __name__ == '__main__':
    for epoch in tqdm(epochs):
        train()
