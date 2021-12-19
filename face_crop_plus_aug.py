import cv2
import dlib
import os
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from tqdm import tqdm

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

# predictor_model = 'shape_predictor_68_face_landmarks.dat'
# detector = dlib.get_frontal_face_detector()  # dlib人脸检测器
# predictor = dlib.shape_predictor(predictor_model)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

pathnames = [[] for i in range(5)]
output_path = [[] for i in range(5)]
index = 0
pos = [0, 0, 0, 0, 0]


def load_data(dataset):
    global pathnames
    global output_path
    global index
    path = Frame_Path[dataset]
    # output_path = OUTPUT_PATH[dataset]
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            pathnames[index] += [os.path.join(dirpath, filename)]
            tail = dirpath.split('\\')
            output_path[index] += [os.path.join(
                OUTPUT_PATH[dataset], tail[-1], filename)]
    print(len(pathnames[index]))


def face_crop():
    global pos
    for j in tqdm(range(len(pathnames[0]))):
        img = cv2.imread(pathnames[0][j])
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            img_gray, 1.1, 5, cv2.CASCADE_DO_CANNY_PRUNING)
        if len(faces) > 0:
            for i in range(5):
                # print(len(faces))
                bound_x, bound_y, bound_w, bound_h = 0, 0, 0, 0
                img = cv2.imread(pathnames[i][j])
                output_dir = output_path[i][j].split('\\')
                output_dir = os.path.join(
                    output_dir[-3], output_dir[-2])
                os.makedirs(output_dir, exist_ok=True)
                for (x, y, w, h) in faces:
                    if w*h > bound_w*bound_h:
                        bound_w = w
                        bound_h = h
                        bound_x = x
                        bound_y = y
                img = img[bound_y:bound_y + int(1.1*bound_h),
                          bound_x:bound_x + int(1.1*bound_w)]
                if img.shape[0] > 0 and img.shape[1] > 0:
                    img = cv2.resize(img, (256, 256))
                    img = image_normalization(
                        img, normalize=__imagenet_stats)
                    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    # cv2.imshow('1', img)
                    # cv2.waitKey(0)
                    cv2.imwrite(output_path[i][j], img)


def image_normalization(img, normalize=__imagenet_stats):
    # img = img.transpose(2,0,1)  #h,w,c to c,h,w
    transform_norm = transforms.ToTensor()
    img_tr = transform_norm(img)  # get tensor image
    # mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
    transform_norm = transforms.Normalize(**normalize)
    img_normalized = transform_norm(img_tr)
    img_normalized = np.array(img_normalized)
    # img = img_normalized.transpose(1, 2, 0)
    return img


if __name__ == '__main__':
    for dataset in Frame_Path.keys():
        load_data(dataset)
        index += 1
    face_crop()
