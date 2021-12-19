# randomly select the face crops of the dataset and calculate the mask
import os
import random
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2

ORI_PATH = 'faces/original'
MANI_PATH = 'faces/manipulated'
OUTPUT_PATH = 'mask_dssim/'
original_filenames = [[] for i in range(4)]
fake_filenames = [[] for i in range(4)]


def load_fake_image():
    global fake_filenames
    for curDir, dirs, files in os.walk(MANI_PATH):
        # print(curDir)
        pathname = curDir.split('\\')
        if len(pathname) == 3:
            if pathname[1] == 'deepfakes':
                filename = random.sample(files, 1)
                fake_filenames[0] += [os.path.join(curDir, filename[0])]
            elif pathname[1] == 'face2face':
                filename = random.sample(files, 1)
                fake_filenames[1] += [os.path.join(curDir, filename[0])]
            elif pathname[1] == 'faceswap':
                filename = random.sample(files, 1)
                fake_filenames[2] += [os.path.join(curDir, filename[0])]
            elif pathname[1] == 'neutraltextures':
                filename = random.sample(files, 1)
                fake_filenames[3] += [os.path.join(curDir, filename[0])]
    # for i in range(4):
    #     print(len(fake_filenames[i]))
    #     print(fake_filenames[i])


def load_original_image():
    global original_filenames
    count = 0
    for curDir, dirs, files in os.walk(ORI_PATH):
        # print(curDir)
        pathname = curDir.split('\\')
        if len(pathname) == 2:
            for i in range(4):
                filename = fake_filenames[i][count].split('\\')[-1]
                original_filenames[i] += [os.path.join(curDir, filename)]
            count = count + 1
    # for i in range(4):
    #     print(len(original_filenames[i]))
    #     print(original_filenames[i])


def calculate_dssim():
    # diff = np.zeros((256, 256))
    for i in range(4):
        for j in range(2):
            original_img = cv2.imread(original_filenames[i][j])
            fake_img = cv2.imread(fake_filenames[i][j])
            # cv2.imshow('origin',original_img)
            # cv2.waitKey(0)
            # cv2.imshow('fake',fake_img)
            # cv2.waitKey(0)
            grayA = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(fake_img, cv2.COLOR_BGR2GRAY)
            # grayA = cv2.blur(grayA, (3,3))
            # grayB = cv2.blur(grayB, (3,3))
            # cv2.imshow('gray1',grayA)
            # cv2.waitKey(0)
            # cv2.imshow('gray1',grayB)
            # cv2.waitKey(0)
            score,diff = ssim(grayA, grayB,full=True,win_size=33)
            diff = (diff*255).astype("uint8")
            diff = cv2.GaussianBlur(diff, (5, 5), 0)
            # print(diff)
            # cv2.imshow('fake', diff)
            # cv2.waitKey(0)
            # # print(diff.shape)
            # # cv2.imshow('diff',diff)
            # # cv2.waitKey(0)
            # min_threshold = score*255
            dst = cv2.threshold(
                diff,0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
            
            # cv2.imshow('threshold', dst)
            # cv2.waitKey(0)
            # dst = cv2.resize(dst,(16,16),interpolation=cv2.INTER_LINEAR)
            # dst = dst / 255
            # print(dst)
            # cv2.imshow('threshold', dst)
            # cv2.waitKey(0)


if __name__ == '__main__':
    load_fake_image()
    load_original_image()
    calculate_dssim()
