print("Setting Up")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from utils import *
import sudokuSolver

pathImage = "./Resources/1.png"
heightImg = 360
widthImg = 360
model = initializePredictionModel()  #initialize the cnn model

# Prepare the Image
img = cv2.imread(pathImage)
img = cv2.resize(img,(widthImg, heightImg)) #resize image to make it a square
imgBlank = np.zeros((heightImg,widthImg,3),np.uint8) #create a blank image for testing
imgThreshold = preProcess(img)

# Find All Contours
imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS

# Find the Biggest Contour and use it as Sudoku
biggest, maxArea = biggestContour(contours) # FIND THE BIGGEST CONTOUR
if biggest.size != 0:
    biggest=reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 25) # DRAW THE BIGGEST CONTOUR
    # imgBigContour = drawRectangle(imgBigContour,biggest,2)
    pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

    # Split the image and find each digit available
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    numbers = getPrediction(boxes, model)
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255,0,255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)

    # Find Solution Of the Board
    board = np.array_split(numbers,9)
    try:
        sudokuSolver.solve(board)
    except:
        pass

    flatList = []
    for subList in board:
        for item in subList:
            flatList.append(item)
    solvedNumbers = flatList*posArray
    imgSolvedDigits = displayNumbers(imgSolvedDigits,solvedNumbers)

    # Overlay Solutions
    pts2 = np.float32(biggest) #Prepare points for warp
    pts1 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]]) #Prepare points
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits,matrix,(widthImg,heightImg))
    inv_perspective = cv2.addWeighted(imgInvWarpColored,1,img,0.5,1)
    imgDetectedDigits = drawGrid(imgDetectedDigits)
    imgSolvedDigits = drawGrid(imgSolvedDigits)

    imageArray = ([img,imgThreshold,imgContours,imgBigContour],
                    [imgDetectedDigits,imgSolvedDigits,imgInvWarpColored,inv_perspective])
    stackedImage = stackImages(imageArray,1)
    cv2.imshow("Stacked Images",stackedImage)
else:
    print("No Sudoku Found")
cv2.waitKey(0)