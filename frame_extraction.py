import os
from os.path import join
import cv2
from tqdm import tqdm

DATASET_PATHS = {
    'original': 'original_sequences/youtube/c23/videos',
    'Deepfakes': 'manipulated_sequences/Deepfakes/c23/videos',
    'Face2Face': 'manipulated_sequences/Face2Face/c23/videos',
    'FaceSwap': 'manipulated_sequences/FaceSwap/c23/videos',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures/c23/videos'
}

OUTPUT_PATHS = {
    'original': 'frames/original',
    'Deepfakes': 'frames/manipulated/deepfakes',
    'Face2Face': 'frames/manipulated/face2face',
    'FaceSwap': 'frames/manipulated/faceswap',
    'NeuralTextures': 'frames/manipulated/neutraltextures'
}


def extract_videos(data_path, dataset):
    videos_path = DATASET_PATHS[dataset]
    images_path = OUTPUT_PATHS[dataset]
    for video in tqdm(os.listdir(videos_path)):
        image_folder = video.split('.')[0]
        extract_frames(join(videos_path, video),
                       join(images_path, image_folder))

def extract_frames(data_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    reader = cv2.VideoCapture(data_path)
    frame_num = 0
    while reader.isOpened():
        success, image = reader.read()
        if not success:
            break
        if frame_num <= 200:
            if frame_num % 50 == 0:
                cv2.imwrite(join(output_path, '{:04d}.png'.format(frame_num)),
                            image)
        else:
            break
        frame_num += 1
    reader.release()

if __name__ == '__main__':
    for dataset in DATASET_PATHS.keys():
        extract_videos(DATASET_PATHS[dataset],dataset)
