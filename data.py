import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self,csv_path):
        self.path_info = pd.read_csv(csv_path)
        self.image_list = self.path_info.loc[:,'image']
        self.vol_list = self.path_info.loc[:,'volumn']
        self.label_list = self.path_info.loc[:,'label']

        self.data_len = len(self.path_info.index) - 1
    
    def __getitem__(self,index):
        img_path = self.image_list[index]
        img = Image.open(img_path)
        img = transforms.ToTensor()(img)

        vol_path = self.vol_list[index]
        volumn = torch.load(vol_path)

        label = self.label_list[index]

        return img,volumn,label

    def __len__(self):
        return self.data_len
