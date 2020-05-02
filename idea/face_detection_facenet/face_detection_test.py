from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import multiprocessing as mp
import os
import cv2

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


thread_hold = 0.4
dataPath = 'data'
testPath ='test'
os.chdir(dataPath)
dir_name = [n for n in os.listdir() if os.path.isdir(n) and not n.startswith('.')]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

#Feature extraction:
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709)
#Crop face from image:
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
test_images =[]
os.chdir('..')
list_test_images = [testPath+"/"+n for n in os.listdir(testPath) if not n.startswith('.')]
print(list_test_images)
# os.chdir(testPath)
for image in list_test_images:
    
    im = cv2.imread(image)
    if isinstance(im, type(None)):
        continue
    im_crop = mtcnn(im)
    im_crop = im_crop.to(device)
    im_extract = resnet(im_crop.unsqueeze(0))
    im_extract = im_extract.detach().cpu().numpy()
    im_extract = im_extract.reshape(512)
    test_images.append(im_extract)


model = Sequential([
    Dense(64, activation='relu', input_shape=(512,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(dir_name), activation='softmax'),
])

# Load the model's saved weights.
model.load_weights('model.h5')

test_images = np.asarray(test_images)

#predict the images:
predictions = model.predict(test_images[:])
# print(predictions[2])
resProb = np.amax(predictions, axis=1)
res = np.argmax(predictions, axis=1)
print(res)

count = 0
# for i in res:

#     print(list_test_images[count], ": ", dir_name[i])
#     count+=1

for i in range(0, len(res)):
    if(resProb[i] >= thread_hold):
        print(list_test_images[i], ":", "res: ", dir_name[res[i]])
    else:
        print(list_test_images[i], ":", "Unknown")
