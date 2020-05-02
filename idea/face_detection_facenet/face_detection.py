
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

dataPath = 'data'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

#Feature extraction:
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709)
#Crop face from image:
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

os.chdir(dataPath)
dir_name = [n for n in os.listdir() if os.path.isdir(n) and not n.startswith('.')]

print(dir_name)
train_images = []
train_labels = []

count=0
for person in dir_name:
    images_list = [i for i in os.listdir(person) if not i.startswith('.')]
    
    for path in images_list:
        path = person+"/"+path
        # print(path)
        im = cv2.imread(path)
        if isinstance(im, type(None)):
            continue
        im_crop = mtcnn(im)
        if isinstance(im_crop, type(None)):
            continue
        # print(type(im_crop))
        im_crop = im_crop.to(device)
        # print(type(im_crop))
        im_extract = resnet(im_crop.unsqueeze(0))
        im_extract = im_extract.detach().cpu().numpy()
        im_extract = im_extract.reshape(512)
        train_images.append(im_extract)
        train_labels.append(count)
        print("done.")
    count+=1

train_images = np.asarray(train_images)

model = Sequential([
    Dense(64, activation='relu', input_shape=(512,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(dir_name), activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=8,
    batch_size=32,
)
os.chdir('..')
model.save_weights('model.h5')