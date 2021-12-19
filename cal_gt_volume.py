import numpy as np
import os
import cv2
from numpy.lib.npyio import save
import pandas as pd
import torch
from tqdm import tqdm

Mask_Path = 'mask_dssim/'
Save_Path = 'GT_Volume/'
mask_names = []


def load_mask():
    global mask_names
    for curDir, dirs, files in os.walk(Mask_Path):
        for filename in files:
            mask_names += [os.path.join(curDir, filename)]


def cal_volume():
    gt_V = np.zeros((16, 16, 16, 16))
    gt_V = torch.from_numpy(gt_V)
    # print(gt_V,gt_V.shape)
    for mask_path in tqdm(mask_names):
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask / 255).astype(float)
        mask = cv2.resize(mask, (16, 16), interpolation=cv2.INTER_LINEAR)
        # print(mask.shape)
        # print(mask)
    # mask = np.random.randint(0,255,(16,16))
    # mask = mask / 255
        for h_1 in range(16):
            for w_1 in range(16):
                for h_2 in range(16):
                    for w_2 in range(16):
                        gt_V[h_1, w_1, h_2, w_2] = 1 - \
                            abs(mask[h_1][w_1]-mask[h_2][w_2])
        save_path = mask_path.split('/')
        save_path = os.path.join(Save_Path, save_path[1])
        save_dir = save_path.split('\\')
        # print(save_dir)
        save_name = save_dir[2]
        save_name = os.path.splitext(save_name)[0]
        save_name = save_name + '.pt'
        save_dir = os.path.join(save_dir[0] , save_dir[1])
        # print(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir,save_name)
        # print(save_path)
        torch.save(gt_V, save_path)
        # print(gt_V)


if __name__ == '__main__':
    load_mask()
    cal_volume()
