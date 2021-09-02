import cv2
# import mediapipe
# import numpy as np
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
cam.set(cv2.CAP_PROP_FPS,60)

segment = SelfiSegmentation()
imgList = os.listdir("Images")

imList=[]

for imgPath in imgList:
    img = cv2.imread(f'Images/{imgPath}')
    imList.append(img)


fpsReader = cvzone.FPS()

imgIndex = 0

while True:
    success, img = cam.read()
    outputImg= segment.removeBG(img,imList[imgIndex],0.3)

    imgStacked = cvzone.stackImages([img,outputImg],2,1)
    _, imgStacked = fpsReader.update(imgStacked, color=(0,255,0))
    cv2.imshow("Image",imgStacked)
    key = cv2.waitKey(1)
    if key == ord('a'):
        if imgIndex > 0:
            imgIndex -=1
    elif key == ord('f'):
        if imgIndex < len(imList) - 1:
            imgIndex +=1
    elif key == ord('q'):
        break

