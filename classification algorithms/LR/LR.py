
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
from sklearn.linear_model import LogisticRegression #LR
from sklearn.externals import joblib

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
            #print(directory,filename)
            if files:
                for file in files:
                    img = image.load_img(directory + '/' + filename+'/'+file, target_size=(28, 28, 1),grayscale=False)
                    img = image.img_to_array(img)
                    img = img / 255
                    train_image.append(img.flatten())

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
#Y= to_categorical(Y)


[X,Y]=shuffle(X,Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.2)


# print(X.shape)
# print(Y.shape)
#
model=LogisticRegression(solver='lbfgs',max_iter=2000)
model.fit(X,Y)
# model = OneVsRestClassifier(LinearSVC())
# model.fit(X,Y)
Y_test=np.array(Y_test)
#Y_testing= to_categorical(Y_testing)
print(X_test.shape)
print(Y_test.shape)

prediction = model.predict(X_test)
accuracy = model.score(X_test,Y_test)
print(accuracy)

# original=Y_testing.tolist()
# prediction=prediction
# print('AuC score on test data:',roc_auc_score(original,prediction))

# Save to file in the current working directory
joblib_file = "joblib_model_LR.pkl"
joblib.dump(model, joblib_file)

print("Saved model to disk")