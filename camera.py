import cv2
import numpy as np
import pandas as pd
import sys

from tensorflow.keras.models import Sequential #for initializing
from tensorflow.keras.layers import Dense  #adding layers
from tensorflow.keras.layers import Conv2D  #adding convolution layer
from tensorflow.keras.layers import MaxPooling2D  #max pooling
from tensorflow.keras.layers import Flatten,Dropout,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}


#emoji_dist={0:"./emojis/angry.png",1:"./emojis/disgusted.png",2:"./emojis/fearful.png",3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surpriced.png"}
emoji_dist_male={0:"./emojis/mangry.png",1:"./emojis/mdisgusted.png",2:"./emojis/mfearful.png",3:"./emojis/mhappy.png",4:"./emojis/mneutral.png",5:"./emojis/msad.png",6:"./emojis/msurpriced.png"}
emoji_dist_female={0:"./emojis/fangry.png",1:"./emojis/fdisgusted.png",2:"./emojis/ffearful.png",3:"./emojis/fhappy.png",4:"./emojis/fneutral.png",5:"./emojis/fsad.png",6:"./emojis/fsurpriced.png"}

faceCascade=cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')

ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)
emoji = cv2.imread('./emojis/neutral.png')
cartoon=cv2.imread('./emojis/neutral.png')
num_down = 2
num_bilateral = 7
model = load_model('gender_detection.model')

class VideoCamera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.emotion_model = Sequential()

        self.emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        self.emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.emotion_model.add(Dropout(0.25))

        self.emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.emotion_model.add(Dropout(0.25))

        self.emotion_model.add(Flatten())
        self.emotion_model.add(Dense(1024, activation='relu'))
        self.emotion_model.add(Dropout(0.5))
        self.emotion_model.add(Dense(7, activation='softmax'))
        self.emotion_model.load_weights('emotion_model.h5')

    def __del__(self):
        self.cap.release()


    def get_frame(self):
        success, img = self.cap.read()
        cv2.ocl.setUseOpenCL(False)

        while True:
            success,img=self.cap.read()
            img=cv2.flip(img,1)
            ig=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=faceCascade.detectMultiScale(ig,1.3,5)
            global cartoon
            cartoon=img
            img_color= cartoon
            for _ in range(num_down):
                img_color = cv2.pyrDown(img_color)
            for _ in range(num_bilateral):
                img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9,  sigmaSpace=7)
            for _ in range(num_down):
                img_color = cv2.pyrUp(img_color)
            # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_blur = cv2.medianBlur(ig, 7)
            img_edge = cv2.adaptiveThreshold(img_blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,  blockSize=9, C=2)
            img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
            img_cartoon = cv2.bitwise_and(img_color, img_edge)
            cartoon=img_cartoon

            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray_frame = ig[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                prediction = self.emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))

                face = img[y:y + h, x:x + w].copy()
                # gblob = cv2.dnn.blobFromImage(face, 1.5, (227, 227), MODEL_MEAN_VALUES, swapRB=True)
                # genderNet.setInput(gblob)
                # genderPreds=genderNet.forward()
                # gender=genderList[genderPreds[0].argmax()]
                face_crop = cv2.resize(face, (96,96))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)
                conf = model.predict(face_crop)[0]
                idx = np.argmax(conf)
                gender = genderList[idx]

                global emoji
                if gender=='Male':
                    emoji=cv2.imread(emoji_dist_male[maxindex])
                elif gender=='Female':
                    emoji=cv2.imread(emoji_dist_female[maxindex])

                ablob = cv2.dnn.blobFromImage(face, 1.5, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                ageNet.setInput(ablob)
                agePreds=ageNet.forward()
                age=ageList[agePreds[0].argmax()]

                cv2.putText(img, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img, f'{gender}, {age}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

            ret, jpeg = cv2.imencode('.jpg', img)
            return jpeg.tobytes()

    def  get_emoji(self):
        global emoji
        ret, emj = cv2.imencode('.jpg', emoji)
        return emj.tobytes()

    def get_cartoon(self):
        global cartoon
        ret, car = cv2.imencode('.jpg', cartoon)
        return car.tobytes()
