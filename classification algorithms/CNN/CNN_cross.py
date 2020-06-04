import os
from os import path
import cv2
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import to_categorical
from tqdm import tqdm
from sklearn.model_selection import KFold
from keras.models import model_from_json


source_folder=[]

source_folder_orange= '/home/bala/Documents/BTP/ROI/Orange'   #class0
source_folder_green= '/home/bala/Documents/BTP/ROI/Green'     #class1
source_folder_yellow= '/home/bala/Documents/BTP/ROI/Yellow'   #class2
source_folder_broken= '/home/bala/Documents/BTP/ROI/Broken'   #class3
source_folder_toyellow='/home/bala/Documents/BTP/ROI/Green_to_yellow' #class 4




source_folder.append(source_folder_orange)
source_folder.append(source_folder_green)
source_folder.append(source_folder_yellow)
source_folder.append(source_folder_broken)
source_folder.append(source_folder_toyellow)




train_image = []

for directory in source_folder:
    for filename in os.listdir(directory):
        for path, dir, files in os.walk(directory + '/' + filename):
            print(directory,filename)
            if files:
                for file in files:
                    img = image.load_img(directory + '/' + filename+'/'+file, target_size=(28, 28, 1),grayscale=False)
                    img = image.img_to_array(img)
                    img = img / 255
                    train_image.append(img)

#print(len(train_image))

X = np.array(train_image)
Y=[]
for k in tqdm(range(591)):
  Y.append(0)
for n in tqdm(range(465)):
  Y.append(1)
for a in tqdm(range(232)):
  Y.append(2)
for b in tqdm(range(16)):
  Y.append(3)
for b in tqdm(range(90)):
  Y.append(4)


Y=np.array(Y)
Y= to_categorical(Y)

print(len(X))
print(len(Y))

[X,Y]=shuffle(X,Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.2)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=40, batch_size=10, validation_data=(X_test, Y_test))
print(model.summary())

test_image=[]
test_img=image.load_img('/home/bala/PycharmProjects/untitled/23.jpg', target_size=(28, 28, 1),grayscale=False)

test_img = image.img_to_array(test_img)
test_img = test_img / 255
test_image.append(test_img)
X_testing = np.array(test_image)
Y_testing=model.predict(X_testing)
print(Y_testing[0])




model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelcnn.h5")
print("Saved model to disk")
