# randomly select the face crops of the dataset and calculate the mask
import os
import random
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
from tqdm import tqdm

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
                fake_filenames[0] += [os.path.join(curDir, filename[i])
                                      for i in range(1)]
            elif pathname[1] == 'face2face':
                filename = random.sample(files, 1)
                fake_filenames[1] += [os.path.join(curDir, filename[i])
                                      for i in range(1)]
            elif pathname[1] == 'faceswap':
                filename = random.sample(files, 1)
                fake_filenames[2] += [os.path.join(curDir, filename[i])
                                      for i in range(1)]
            elif pathname[1] == 'neutraltextures':
                filename = random.sample(files,1)
                fake_filenames[3] += [os.path.join(curDir, filename[i])
                                      for i in range(1)]
    # for i in range(4):
    #     print(len(fake_filenames[i]))
    #     print(fake_filenames[i])


def load_original_image():
    global original_filenames
    for i in range(4):
        for j in range(1000):
            pathname = fake_filenames[i][j].split('\\')
            filename = pathname[-1]
            pathname[0] = ORI_PATH
            pathname[1] = pathname[2][0:3]
            original_filenames[i] += [os.path.join(
                pathname[0], pathname[1], filename)]
    # for i in range(4):
    #     print(len(original_filenames[i]))
    #     print(original_filenames[i])


def calculate_dssim():
    # diff = np.zeros((256, 256))
    for i in range(4):
        for j in tqdm(range(1000)):
            original_img = cv2.imread(original_filenames[i][j])
            fake_img = cv2.imread(fake_filenames[i][j])
            # original_img = cv2.imread('E:/Self_consistency/faces/original/000/0000.png')
            # fake_img = cv2.imread('E:/Self_consistency/faces/manipulated/deepfakes/000_003/0000.png')
            # cv2.imshow('origin',original_img)
            # cv2.waitKey(0)
            # cv2.imshow('fake',fake_img)
            # cv2.waitKey(0)
            grayA = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(fake_img, cv2.COLOR_BGR2GRAY)
            score, diff = ssim(grayA, grayB, full=True, win_size=31)
            dssim_score = (1-score)/2
            diff = (diff*255).astype("uint8")
            diff = cv2.bitwise_not(diff)
            diff = cv2.GaussianBlur(diff, (3, 3), 0)
            # print(diff)
            # cv2.imshow('fake', diff)
            # cv2.waitKey(0)
            dst = cv2.threshold(
                diff, dssim_score*255, 255, cv2.THRESH_BINARY)[1]
            # cv2.imshow('threshold', dst)
            # cv2.waitKey(0)
            tail = fake_filenames[i][j].split('\\')
            output_dir = os.path.join(OUTPUT_PATH, tail[-3], tail[-2])
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, tail[-1])
            cv2.imwrite(output_path, dst)


if __name__ == '__main__':
    load_fake_image()
    load_original_image()
    calculate_dssim()
