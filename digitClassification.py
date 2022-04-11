import numpy as np
import cv2
import pickle
from keras.models import load_model

width = 640
height = 480
threshold = 0.65

cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

# model = pickle.load(open("./Resources/myModel.pkl","rb"))
model = load_model("./Resources/myModel.h5")
# model = load_model("./myModel.h5")
# model = pickle.load(open("./model_trained_10.p","rb"))

def preProcess(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Convert Image to GrayScale
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1) # Add Gaussian Blur
    imgThreshold = cv2.adaptiveThreshold(imgBlur,255,1,1,11,2) #apply adaptive threshold
    return imgThreshold

while True:
    success,imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(32,32))
    img = preProcess(img)
    # cv2.imshow("Processed Image",img)
    img = img.reshape(1,32,32,1)
    # Predict
    classIndex = model.predict(img).argmax(axis=-1)[0]
    print(classIndex)
    predictions = model.predict(img)
    probVal = np.amax(predictions)
    print(probVal)

    if probVal > threshold:
        cv2.putText(imgOriginal,str(classIndex) + " " + str(round(probVal*100,2)),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)

    cv2.imshow("Original Image",imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break