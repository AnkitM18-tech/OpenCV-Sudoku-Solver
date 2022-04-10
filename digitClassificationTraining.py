import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Paths and Variables
path = "./myData"
testRatio = 0.2
valRatio = 0.2

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
x_train,x_test,y_train,y_test = train_test_split(images,classNo,test_size=testRatio)
x_train,x_validation,y_train,y_validation = train_test_split(x_train,y_train,test_size=valRatio)

# print(x_train.shape)
# print(x_test.shape)
# print(x_validation.shape)

numOfSamples = []
for x in range(noOfClasses):
    # print(len(np.where(y_train == x)[0]))
    numOfSamples.append(len(np.where(y_train == x)[0]))
print(numOfSamples)

# Represent the Number of Samples
plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClasses),numOfSamples)
plt.title("Number Of Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("Number Of Images")
plt.show()