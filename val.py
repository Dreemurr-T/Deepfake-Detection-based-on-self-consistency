import pandas as pd
from imageio import imread,imsave
import torch
import os
import random
import numpy as np
from skimage.transform import resize
# resnet = models.resnet34()
# print(resnet)

# data = pd.read_csv('data.csv')
# print(len(data.index))
# img_path = data.loc[:,'image']

# for img in img_path:
#     print(img)

device = ('cuda' if torch.cuda_is_available() else 'cpu')

model = torch.load('checkpoint/model_epoch_200.pth').to(device)
# print(model)

frame_path = 'frames/'

frame_filenames = []
labels = [0]*1000+[1]*4000
print(labels)

def get_filename():
    global frame_filenames
    for curDir,dirname,files in os.walk(frame_path):
        if len(files)>0:
            filename = random.sample(files,1)
            frame_filenames += [os.path.join(curDir,filename[0])]
    print(len(frame_filenames))

def test():
    for frame in frame_filenames:
        img = imread(frame)
        img = resize(img,(256,256,3))
        img = np.transpose(img,(2,0,1))
        img = torch.from_numpy(img).to(device)
        dssim,label = 


if __name__ == '__main__':
    get_filename()