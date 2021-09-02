import cv2
import mediapipe as mp
import time

class handDetect():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.resultimg = self.hands.process(imgRGB)
        #print(resultimg.multi_hand_landmarks)

        if self.resultimg.multi_hand_landmarks:
           for handLnd in self.resultimg.multi_hand_landmarks:
               if draw:
                    self.mpDraw.draw_landmarks(img, handLnd, self.mpHands.HAND_CONNECTIONS)

        return img

    def findpost(self, img, handNo = 0, draw=True):
        lmlist = []
        if self.resultimg.multi_hand_landmarks:
            myhand = self.resultimg.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhand.landmark):

                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                # if id ==8:
                lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 2,(255,0,0), 2)

        return lmlist

def main():
    prevTime = 0
    crrTime = 0
    cam = cv2.VideoCapture(0)
    detector =handDetect()
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

if __name__ == "__main__":
    main()