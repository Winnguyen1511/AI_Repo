from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import pandas as pd
import multiprocessing as mp
import os
import cv2
import tensorflow as tf
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

thread_hold = 0.5
dataPath = 'data'
dir_name = []

def sys_init():
    # set gpu memory growth True
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
      except RuntimeError as e:
        print(e)

    global dir_name
    os.chdir(dataPath)
    dir_name = [n for n in os.listdir() if os.path.isdir(n) and not n.startswith('.')]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('Running on device: {}'.format(device))

    #Feature extraction:
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, keep_all=True, device=device)

    #Crop face from image:
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    os.chdir('..')

    model = Sequential([
        Dense(64, activation='relu', input_shape=(512,)),
        Dense(64, activation='relu'),
        Dense(len(dir_name), activation='softmax'),
    ])

    # Load the model's saved weights.
    model.load_weights('model.h5')

    return mtcnn,resnet,device,model

def recognize_faces(mtcnn, resnet, device, model) :
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    #start_time = time.time()

    while True :
        _, image = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q') : # exit on q
              break
        box_start = time.time()
        boxes, _ = mtcnn.detect(image)
        box_stop = time.time()
        #print("\rt-box: {:.2f}".format((box_stop - box_start)))
        
        if not isinstance(boxes, type(None)):
            for (x, y, w, h) in boxes:
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                cv2.rectangle(image, (x, y), (w,h), (255, 0, 0), 2)

                im_crop = image[y:h, x:w]
                im_crop = cv2.resize(im_crop, (160, 160), interpolation = cv2.INTER_CUBIC)
                im_crop = ToTensor()(im_crop)
                im_crop = im_crop.to(device)
                
                resnet_start = time.time()
                im_extract = resnet(im_crop.unsqueeze(0))
                im_extract = im_extract.detach().cpu().numpy()
                im_extract = im_extract.reshape(512)
                resnet_stop = time.time()
                #print("\rFPS extract: {:.2f}".format((time.time() - start_time)))


                predictions = model.predict(np.asarray([im_extract]))
                predict_stop = time.time()
                resProb = np.amax(predictions)
                res = np.argmax(predictions)

                #print("\rFPS pred: {:.2f}".format((time.time() - start_time)))

                if(resProb >= thread_hold):
                    cv2.putText(image, '{0}: {1:.4f}'.format(dir_name[res], resProb), (x+5,y-5), font, 1, (0,0,255), 2)
                else:
                    cv2.putText(image, "Unknown", (x+5,y-5), font, 1, (0,0,255), 2)

            cv2.imshow("Face Recognizer", image)
        else:
            cv2.imshow("Face Recognizer", image)
            continue

        # cv2.imshow("Face Recognizer", image)
        print("> t-detect= %.4f"%(box_stop - box_start))
        print("> t-extract= %.4f"%(resnet_stop - resnet_start))
        print("> t-predict= %.5f"%(predict_stop - resnet_stop))
        #print("\rFPS all: {:.2f}".format((time.time() - start_time)))
        
        stop_time = time.time()
        print(">> FPS: {:.2f}".format(1 / (stop_time - box_start)))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    mtcnn, resnet, device, model = sys_init()
    recognize_faces(mtcnn,resnet,device, model) 