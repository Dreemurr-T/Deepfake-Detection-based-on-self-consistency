import os
import random
from PIL import Image
import PIL
import torch
from tqdm import tqdm
import csv

ORI_PATH = 'faces/original'
MASK_PATH = 'mask_dssim/'
FAKE_PATH = 'faces/manipulated'
GT_PATH = 'GT_Volume/'
train_path = 'train_set/'

original_filenames = []
fake_filenames = []
gt_filenames = []
f = open('data.csv', 'w+', newline='')


def load_original_image():
    global original_filenames
    for curDir, dirs, files in os.walk(ORI_PATH):
        # print(curDir)
        pathname = curDir.split('\\')
        if len(pathname) == 2:
            if len(files) > 1:
                filename = random.sample(files, 2)
                original_filenames += [os.path.join(curDir, filename[i])
                                       for i in range(2)]
            else:
                filename = random.sample(files, 1)
                original_filenames += [os.path.join(curDir, filename[0])]


def load_fake_image():
    global fake_filenames
    for curDir, dirs, files in os.walk(MASK_PATH):
        for filename in files:
            name = os.path.join(curDir, filename)
            # print(name)
            filename_split = name.split('/')
            filename_split[0] = FAKE_PATH
            name = os.path.join(filename_split[0], filename_split[1])
            fake_filenames += [name]


def load_gt():
    global gt_filenames
    for filename in fake_filenames:
        filename_split = filename.split('\\')
        filename_split[0] = GT_PATH
        filename_split[3] = filename_split[3][0:4]+'.pt'
        gt_filenames += [os.path.join(filename_split[0],
                                      filename_split[1], filename_split[2], filename_split[3])]


if __name__ == '__main__':
    load_original_image()
    load_fake_image()
    load_gt()
    cnt = 0
    writer = csv.writer(f)
    header = ['image', 'volumn', 'label']
    writer.writerow(header)
    for filename in tqdm(original_filenames):
        original_img = Image.open(filename)
        output_dir = os.path.join(train_path, str(cnt))
        # print(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_path_img = output_dir + '\\' + str(cnt) + '.png'
        # print(output_path)
        original_img.save(output_path_img)
        output_path_gt = output_dir + '\\' + str(cnt) + '.pt'
        gt = torch.ones((16, 16, 16, 16))
        torch.save(gt, output_path_gt)
        data = [output_path_img, output_path_gt, 0]
        writer.writerow(data)
        cnt += 1

    for i in tqdm(range(4000)):
        fake_img_path = fake_filenames[i]
        gt_path = gt_filenames[i]
        fake_img = Image.open(fake_img_path)
        output_dir = os.path.join(train_path, str(cnt))
        os.makedirs(output_dir, exist_ok=True)
        output_path_img = output_dir + '\\' + str(cnt) + '.png'
        fake_img.save(output_path_img)
        output_path_gt = output_dir + '\\' + str(cnt) + '.pt'
        gt = torch.load(gt_path)
        torch.save(gt,output_path_gt)
        data = [output_path_img, output_path_gt, 1]
        writer.writerow(data)
        cnt += 1

    # print(len(original_filenames))
    # print(len(fake_filenames))
    # print(original_filenames)
    # print(fake_filenames)
    # print(len(mask_filenames))
