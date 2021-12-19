import numpy as np
from imageio import imread,imsave
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(filepath):
    image = imread(filepath)
    image = np.transpose(image,(2,0,1))
    image = torch.from_numpy(image)
    return image

def is_image(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg"])