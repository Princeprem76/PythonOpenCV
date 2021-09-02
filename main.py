import cv2
import mediapipe as mp
import time


cam = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
crrTime = 0


while True:
    success, img = cam.read()
    imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultimg = hands.process(imgRGB)
    print(resultimg.multi_hand_landmarks)

    if resultimg.multi_hand_landmarks:
       for handLnd in resultimg.multi_hand_landmarks:
           for id, lm in enumerate(handLnd.landmark):

               h, w, c = img.shape
               cx, cy = int(lm.x*w), int(lm.y*h)
               print(id, cx, cy)
               if id ==8:
                   cv2.circle(img, (cx,cy), 25,(255,0,0), 2)

           mpDraw.draw_landmarks(img, handLnd, mpHands.HAND_CONNECTIONS)

    crrTime = time.time()
    fps = 1/(crrTime - prevTime)
    prevTime = crrTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),2)
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
