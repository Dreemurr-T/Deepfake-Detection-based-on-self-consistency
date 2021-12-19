import os
from tqdm import tqdm
import cv2
import numpy as np
import face_alignment
from skimage import io
import ssl
import torch
import dlib
from PIL import Image
from autocrop import Cropper

print(torch.cuda.is_available())
cropper = Cropper()

# ssl._create_default_https_context = ssl._create_unverified_context

Frame_Path = {
    'original': 'frames/original',
    'deepfakes': 'frames/manipulated/deepfakes',
    'face2face': 'frames/manipulated/face2face',
    'faceswap': 'frames/manipulated/faceswap',
    'neutraltextures': 'frames/manipulated/neutraltextures'
}

OUTPUT_PATH = {
    'original': 'faces/original',
    'deepfakes': 'faces/manipulated/deepfakes',
    'face2face': 'faces/manipulated/face2face',
    'faceswap': 'faces/manipulated/faceswap',
    'neutraltextures': 'faces/manipulated/neutraltextures'
}

# face_landmark = []

def load_data(dataset):
    path = Frame_Path[dataset]
    output_path = OUTPUT_PATH[dataset]
    pathnames = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            pathnames += [os.path.join(dirpath, filename)]
    return pathnames,output_path

def face_crop(pathnames,output_path):
    for pathname in tqdm(pathnames):
        # frame_input = cv2.imread(pathname)
        # frame_input = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
        # get_landmark(frame_input)
        # face_image = dlib.get_face_chips(frame_input, face_landmark[0], size=256)
        # face_image.show()
        # Get a Numpy array of the cropped image
        cropped_array = cropper.crop(pathname)
        # Save the cropped image with PIL if a face was detected:
        if cropped_array is not None and cropped_array.any():
            cropped_image = Image.fromarray(cropped_array)
            cropped_image = cv2.cvtColor(np.asarray(cropped_image), cv2.COLOR_RGB2BGR)
            path_split = pathname.split('\\')
            output_path_new = os.path.join(output_path,path_split[-2])
            os.makedirs(output_path_new, exist_ok=True)
            output_path_new = os.path.join(output_path_new,path_split[-1])
            # print(output_path_new)
            # f = open(output_path_new,'w')
            cv2.imwrite(output_path_new,cropped_image)

# def get_landmark(frame_input):
#     global face_landmark
#     fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
#     face_landmark = fa.get_landmarks(frame_input)

if __name__ == '__main__':
    for dataset in Frame_Path.keys():
        pathnames,output_path = load_data(dataset)
        face_crop(pathnames,output_path)