import numpy as np
import cv2
import os

# Paths and Variables
path = "./myData"

images = []
classNo = []
myList = os.listdir(path)
print("Total Number of Classes Detected ",len(myList))
noOfClasses = len(myList)
print("Importing Classes...")

# Adding Images and Class No to arrays
for x in range(noOfClasses):
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        currentImage = cv2.imread(path + "/" + str(x) + "/" + y)
        currentImage = cv2.resize(currentImage,(32,32))
        images.append(currentImage)
        classNo.append(x)
    print(x,end=" ")
print(" ")
print("Total Images in Image List : ",len(images))
print("Total IDs in Class List : ",len(classNo))

images = np.array(images)
classNo = np.array(classNo)

# print(images.shape) #(10160,32,32,3)
# print(classNo.shape) #(10160,)

# Splitting Data
