import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle

# Paths and Variables
path = "./myData"
testRatio = 0.2
valRatio = 0.2
imageDimensions = (32,32,3)

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
        currentImage = cv2.resize(currentImage,(imageDimensions[0],imageDimensions[1]))
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

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

# Mapping the function to items in the array
x_train = np.array(list(map(preProcessing,x_train)))
x_test = np.array(list(map(preProcessing,x_test)))
x_validation = np.array(list(map(preProcessing,x_validation)))

# Reshaping the arrays, adding extra depth
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_validation = x_validation.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)

# Augmentation to make the dataset more generic
dataGen = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,shear_range=0.1,rotation_range=10)
dataGen.fit(x_train) #Generating the images as we go along not previously generated image

# One hot encoding of matrices
y_train = to_categorical(y_train,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)

# Defining our model
def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNode = 500

    model = Sequential()
    model.add(Conv2D(noOfFilters,sizeOfFilter1,input_shape=(imageDimensions[0],imageDimensions[1],1),activation="relu"))
    model.add(Conv2D(noOfFilters,sizeOfFilter1,activation="relu"))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Conv2D(noOfFilters//2,sizeOfFilter2,activation="relu"))
    model.add(Conv2D(noOfFilters//2,sizeOfFilter2,activation="relu"))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5)) #handling overfitting
    model.add(Flatten())
    model.add(Dense(noOfNode,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses,activation="softmax"))
    model.compile(Adam(lr=0.001),loss="categorical_crossentropy",metrics=["accuracy"])
    return model