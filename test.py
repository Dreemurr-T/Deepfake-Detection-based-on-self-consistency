import pandas as pd
from imageio import imread,imsave
import torch
import os
import random
import numpy as np
from skimage.transform import resize
from torch import tensor
from torch.autograd import Variable
from model import Self_Consistency
from sklearn import metrics
from tqdm import tqdm

device = torch.device("cuda")

model = Self_Consistency(batch_size=64).to(device)

model.load_state_dict(torch.load('checkpoint/model_epoch_150.pth'))

model.eval()
# print(model)

frame_path = 'faces/'

frame_filenames = []
labels = [1]*4000 + [0]*1000
predictions = []
# print(labels)

def get_filename():
    global frame_filenames
    for curDir,dirname,files in os.walk(frame_path):
        if len(files)>0:
            filename = random.sample(files,1)
            frame_filenames += [os.path.join(curDir,filename[0])]
    print(len(frame_filenames))

def test():
    global predictions
    with torch.no_grad():
        for frame in tqdm(frame_filenames):
            img = imread(frame)
            img = np.transpose(img,(2,0,1))
            img = torch.from_numpy(img).to(device)
            img = Variable(img)
            # img = torch.unsqueeze(img,dim=0)
            in_img = img
            for i in range(63):
                in_img = torch.cat((in_img,img),dim=0)
            # print(in_img.shape)
            dssim,label_1 = model(in_img)
            label_1 = label_1.squeeze().float()
            label_1 = torch.round(label_1)
            label_1 = label_1.cpu().numpy()
            predictions.append(label_1[0])
        # print(predictions)
    acc_score = metrics.accuracy_score(labels,predictions)
    auc_score = metrics.roc_auc_score(labels,predictions)
    print('Accuracy is {:.4f}, AUC score is {:.4f}'.format(acc_score,auc_score))



if __name__ == '__main__':
    get_filename()
    test()