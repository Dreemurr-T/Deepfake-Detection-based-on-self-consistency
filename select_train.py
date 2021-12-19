import os
import random

ORI_PATH = 'faces/original'
MANI_PATH = 'faces/manipulated'
TRAIN_PATH = 'train_set/'

def copy_original():
     for curDir, dirs, files in os.walk(ORI_PATH):
        # print(curDir)
        pathname = curDir.split('\\')
