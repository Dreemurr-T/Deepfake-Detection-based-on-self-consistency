from moviepy.editor import *
import os
import cv2
from tqdm import tqdm

DATA_PATH = 'E:/Self_consistency/manipulated_sequences/Deepfakes/c23/videos'
OUTPUT_PATH = 'E:/Self_consistency/extracted_audios'

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

for video in tqdm(os.listdir(DATA_PATH)):
    videoclip = VideoFileClip(os.path.join(DATA_PATH,video))
    audioclip = videoclip.audio
    # print(videoclip.reader.infos)
    # audio_clip = AudioFileClip()
    audioclip.write_audiofile(os.path.join(OUTPUT_PATH,video))