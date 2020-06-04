import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


target_folder= '/home/bala/Documents/BTP/ROI/DAY4/yellow'
source_folder= '/home/bala/Documents/BTP/Day4/Yellow'

for path,dir,files in os.walk(source_folder):
    if files:
        for file in files:
            img = cv2.imread(os.path.join(source_folder,file))
            # cv2.imshow('nut', img)

            result = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  ## convert to hsv

            lower = np.array([5, 80, 25])  # Filtering (Thresholding)
            upper = np.array([100, 255, 180])  # lower and higher hsv values of interested region

            result = cv2.inRange(result, lower, upper)
            # cv2.imshow('result1', result)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            result = cv2.dilate(result, kernel)
            # cv2.imshow('result', result)  # its similar to mask

            # Contours
            contours, hierarchy = cv2.findContours(result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

            if len(contours) != 0:
                idx = 0
                for (i, c) in enumerate(contours):
                    area = cv2.contourArea(c)
                    if area > 8000:
                        print(area)
                        idx += 1
                        # cv2.drawContours(img, c, -1, (0,0,0),0)
                        x, y, w, h = cv2.boundingRect(c)
                        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),0)
                        cropped = img[y:y + h, x:x + w]
                        # cv2.imwrite('43_' + str(idx) + '.jpg', cropped)
                        cv2.imwrite(os.path.join(target_folder, file), cropped)

            # Stack results
            result = np.vstack((result, img))
            resultOrig = result.copy()

            max_dimension = float(max(result.shape))
            scale = 900 / max_dimension
            result = cv2.resize(result, None, fx=scale, fy=scale)
            # cv2.imshow('res', result)
            # cv2.imwrite('43f.jpg', result)



            # if not os.path.isfile(target_folder + file):
            #     os.rename(path+'/'+file,target_folder+'/'+file)