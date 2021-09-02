import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe

cam = cv2.VideoCapture(0)
cam.set(3,1280)
cam.set(4,720)
detector = HandDetector(detectionCon=0.7)
cx,cy,w,h = 100,100,200,200
colorW = (255,0,0)

while True:
    success, img = cam.read()
    img = cv2.flip(img, flipCode= 1)
    img = detector.findHands(img)
    llist,_ = detector.findPosition(img)

    if llist:
        l, _,_ = detector.findDistance(8,12,img)
        print(l)
        if l < 40:
            cursorhand = llist[8]
            if cx-w//2 < cursorhand[0] < cx+w//2 and cy-h//2<cursorhand[1] < cy+h//2:
                colorW = (0,255,0)
                cx,cy = cursorhand
            else:
                colorW = (255,0,0)

    cv2.rectangle(img,(cx-w//2,cy-h//2),(cx+w//2,cy+h//2),colorW,cv2.FILLED)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
