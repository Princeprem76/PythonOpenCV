import cv2
import mediapipe as mp
import handtrackingmod as hm
import time

prevTime = 0
crrTime = 0
cam = cv2.VideoCapture(0)
detector =hm.handDetect()
while True:
    success, img = cam.read()
    img = detector.findHands(img)
    lmlist = detector.findpost(img)
    if len(lmlist) != 0:
        print(lmlist[4])
    crrTime = time.time()
    fps = 1 / (crrTime - prevTime)
    prevTime = crrTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

