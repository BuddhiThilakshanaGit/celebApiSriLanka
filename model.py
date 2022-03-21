import base64
import joblib 
import json
import cv2
from wavelet import w2d
import numpy as np 
from matplotlib import pyplot as plt

# loading model
model=joblib.load('assets/celebrityModel.joblib') 

# loades facecascade
face_cascade=cv2.CascadeClassifier('C:/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('C:/haarcascades/haarcascade_eye.xml')

def getCroppedImage(img): 
    gray=cv2.cvtColor(img
    ,cv2.COLOR_BGR2GRAY) 
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    face_arr=[]
    for (x,y,w,h) in faces:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        face_arr.append(roi_color)
    return face_arr



def getCelebrities():
    f=open(r'assets/celebrities.json','r')
    return json.loads(f.read())


def get_cv2_image_from_Base64(image_base64_data):
    image_base64_data=image_base64_data.replace('data:image/jpeg;base64,','')
    nparr=np.frombuffer(base64.b64decode(image_base64_data),np.uint8)
    img=cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    return img

def get_base64_string(path):
    return open(path, 'r').read() 

def getCombinedImage(img):
    scaled_raw_img=cv2.resize(img,(32,32))
    img_har=w2d(img,'db1',5)
    scaled_img_har=cv2.resize(img_har,(32,32))
    combined_img=np.vstack((scaled_raw_img.reshape(32*32*3,1),scaled_img_har.reshape(32*32,1))) 
    len_image_array=32*32*3+32*32
    return combined_img.reshape(1,len_image_array).astype(float) 



def predictCeleb(base64_img):
    img_arr=get_cv2_image_from_Base64(base64_img) 
    face_arr=getCroppedImage(img_arr)
    celebrities=[]
    for face in face_arr:
        combined_img=getCombinedImage(face)
        prediction=model.predict_proba(combined_img)
        if max(prediction[0])>0.3:
            celebrities.append(getCelebrities()[np.argmax(prediction[0])].replace('_',' ').title())
        else:
            celebrities.append('Unknown Person')
    return celebrities

if __name__=='__main__':
    print(predictCeleb(get_base64_string('b64.txt') ))
