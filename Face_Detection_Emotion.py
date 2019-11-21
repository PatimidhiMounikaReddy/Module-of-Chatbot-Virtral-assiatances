#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils


# In[2]:


pwd


# In[3]:


df =  pd.read_csv('fer2013.csv')
df


# In[4]:


print(df.info())


# In[5]:


X_train,train_y,X_test,test_y=[],[],[],[]


for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
           X_train.append(np.array(val,'float32'))
           train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
           X_test.append(np.array(val,'float32'))
           test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")
        
        
num_features = 64
num_labels = 7
batch_size = 64
epochs = 4
width, height = 48, 48


X_train = np.array(X_train,'float32')
train_y = np.array(train_y,'float32')
X_test = np.array(X_test,'float32')
test_y = np.array(test_y,'float32')

train_y=np_utils.to_categorical(train_y, num_classes=num_labels)
test_y=np_utils.to_categorical(test_y, num_classes=num_labels)

X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)


# In[6]:


model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))


model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))


model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))


model.add(Flatten())


model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))


model.add(Dense(num_labels, activation='softmax'))


model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])


model.fit(X_train, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, test_y),
          shuffle=True)


fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")


# In[7]:





# In[7]:


import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


from keras.models import model_from_json
model = model_from_json(open("fer.json", "r").read())
model.load_weights('fer.h5') 

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


cap = cv2.VideoCapture(0) 

frame = 0

while(True):
    ret, img = cap.read()

    img = cv2.resize(img, (640, 360))
    img = img[0:308,:]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        if w > 130:
            cv2.rectangle(img,(x,y),(x+w,y+h),(64,64,64),2)

            detected_face = img[int(y):int(y+h), int(x):int(x+w)] 
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) 
            detected_face = cv2.resize(detected_face, (48, 48)) 
            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255 
            predictions = model.predict(img_pixels) 
            max_index = np.argmax(predictions[0])
            overlay = img.copy()
            opacity = 0.4
            cv2.rectangle(img,(x+w+10,y-25),(x+w+150,y+115),(64,64,64),cv2.FILLED)
            cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
            cv2.line(img,(int((x+x+w)/2),y+15),(x+w,y-20),(255,255,255),1)
            cv2.line(img,(x+w,y-20),(x+w+10,y-20),(255,255,255),1)
            emotion = ""
            for i in range(len(predictions[0])):
                emotion = "%s %s%s" % (emotions[i], round(predictions[0][i]*100, 2), '%')
                """if i != max_index:
                color = (255,0,0)"""
                color = (0,0,0)
                cv2.putText(img, emotion, (int(x+w+15), int(y-12+i*20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imshow('img',img)

    frame = frame + 1
    #print(frame)

    if frame > 227:
        break

    if cv2.waitKey(480) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




