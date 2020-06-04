import cv2
import numpy
import sys
import datetime
import os
import time
import math

wtLogFileName="yellow_mean_variance.txt"

wtFile = open(wtLogFileName,"a")

source_folder=[]

source_folder_orange_day1= '/home/bala/Documents/BTP/ROI/Yellow/Day1'
source_folder_orange_day2= '/home/bala/Documents/BTP/ROI/Yellow/Day2'
source_folder_orange_day3= '/home/bala/Documents/BTP/ROI/Yellow/Day3'
source_folder_orange_day4= '/home/bala/Documents/BTP/ROI/DAY4/yellow'

source_folder.append(source_folder_orange_day1)
source_folder.append(source_folder_orange_day2)
source_folder.append(source_folder_orange_day3)
source_folder.append(source_folder_orange_day4)

for i in range(1,61):
    for directory in source_folder:
        if os.path.isfile(directory+'/'+str(i)+'.jpg'):
            start_time=time.time()
            img=cv2.imread(directory+'/'+str(i)+'.jpg')
            avg_colour_per_row=numpy.average(img,axis=0)
            avg_colour=numpy.average(avg_colour_per_row,axis=0)
            var_colour=numpy.var(avg_colour_per_row,axis=0)
            end_time=time.time()
            wtFile.write(str(round(avg_colour[2],2)))
            wtFile.write("   ")
            wtFile.write(str(round(avg_colour[1], 2)))
            wtFile.write("   ")
            wtFile.write(str(round(avg_colour[0], 2)))
            wtFile.write("   ")
            wtFile.write(str(round(var_colour[2],2)))
            wtFile.write("   ")
            wtFile.write(str(round(var_colour[1], 2)))
            wtFile.write("   ")
            wtFile.write(str(round(var_colour[0], 2)))
            wtFile.write("   ")
            time_taken=round(end_time-start_time,4)
            wtFile.write(str(time_taken))
            wtFile.write("           ")
        else:
            wtFile.write("----")
            wtFile.write("   ")
            wtFile.write("----")
            wtFile.write("   ")
            wtFile.write("----")
            wtFile.write("   ")
            wtFile.write("----")
            wtFile.write("   ")
            wtFile.write("----")
            wtFile.write("   ")
            wtFile.write("----")
            wtFile.write("   ")
            wtFile.write("----")
            wtFile.write("   ")
            wtFile.write("           ")

    wtFile.write("\n")

wtFile.close()















