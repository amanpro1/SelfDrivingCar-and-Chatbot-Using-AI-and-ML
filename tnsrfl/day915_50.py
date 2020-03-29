# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:58:34 2019

@author: amanv
"""

from os import listdir
from os.path import isfile,join
import time
import cv2
import numpy as np

data_path = "C:\\Users\\amanv\\Documents\\ai&mlclass\\Datasets\\Face\\"
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data,Labels = [],[]
for i,files in enumerate(onlyfiles):
    image_path = data_path +onlyfiles[i]
    images = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images,dtype=np.uint8))
    Labels.append(i)
Labels = np.asarray(Labels,dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data),np.asarray(Labels))
print("Model Training Complete")

face_classifier = cv2.CascadeClassifier(r"C:\Users\amanv\Documents\ai&mlclass\Datasets\data\haarcascades\haarcascade_frontalface_default.xml")
def face_detector(img,size=0.5):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return img,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h,x:x+w]
        roi = cv2.resize(roi,(200,200))
    return img,roi
cap = cv2.VideoCapture(0)
while True:
    flag=False
    ret,frame = cap.read()
    image,face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result = model.predict(face)
        if result[i]<500:
            confidence=int(100*(i-(result[i])/500))
            display_string = str(confidence)+ '% confidence it is user'
            cv2.putText(image,display_string,(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
        if confidence>80:
            cv2.putText(image,"Unlocked",(10,400),cv2.FONT_HERSHEY_COMPLEX,2,(0,120,0),2)
            cv2.imshow('Face cropper',image)
            cv2.waitKey(20)
        else:
            cv2.putText(image,"Locked",(10,400),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
            cv2.imshow('Face cropper',image)
    except:
        cv2.imshow('Face Cropper',image)
        pass
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

        