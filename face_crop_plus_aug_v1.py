import cv2
import dlib
import os
import numpy as np
from tqdm import tqdm
from torchvision import transforms

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

predictor_model = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()  # dlib人脸检测器
predictor = dlib.shape_predictor(predictor_model)

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

def load_data(dataset):
    path = Frame_Path[dataset]
    output_path = OUTPUT_PATH[dataset]
    pathnames = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            pathnames += [os.path.join(dirpath, filename)]
    return pathnames, output_path


def face_crop(pathnames, output_path):
    for pathname in tqdm(pathnames):
        path_split = pathname.split('\\')
        img = cv2.imread(pathname)
        # Save the cropped image with PIL if a face was detected:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rects = detector(img, 0)
        if len(rects) ==0:
            continue
        faces = dlib.full_object_detections()
        for i in range(len(rects)):
            faces.append(predictor(img, rects[i]))
        face_images = dlib.get_face_chips(img, faces, size=256)
        output_path_new = os.path.join(output_path, path_split[-2])
        os.makedirs(output_path_new, exist_ok=True)
        output_path_new = os.path.join(output_path_new, path_split[-1])
        for img in face_images:
            img = image_normalization(img,normalize=__imagenet_stats)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cv2.imshow('1',img)
            # cv2.waitKey()
            # cv2.imwrite(output_path_new, img)

def image_normalization(img,normalize=__imagenet_stats):
    # img = img.transpose(2,0,1)  #h,w,c to c,h,w
    transform_norm = transforms.ToTensor()
    img_tr = transform_norm(img) # get tensor image
    # mean, std = img_tr.mean([1,2]), img_tr.std([1,2]) 
    transform_norm = transforms.Normalize(**normalize)
    img_normalized = transform_norm(img_tr)
    img_normalized = np.array(img_normalized)
    # img = img_normalized.transpose(1, 2, 0)
    return img

if __name__ == '__main__':
    for dataset in Frame_Path.keys():
        pathnames, output_path = load_data(dataset)
        face_crop(pathnames, output_path)
        # image_normalization(img)
