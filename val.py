from torchvision import models
import pandas as pd
from PIL import Image
import PIL

# resnet = models.resnet34()
# print(resnet)

data = pd.read_csv('data.csv')
print(len(data.index))
img_path = data.loc[:,'image']

# for img in img_path:
#     print(img)