import cv2
import mediapipe as mp
import time

import numpy as np

import handtrackingmod as hm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wcam, hcam = 480,720

pTime = 0
cam = cv2.VideoCapture(0)
cam.set(3,wcam)
cam.set(4,hcam)
detector = hm.handDetect(detectionCon=0.8)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volumeRan = volume.GetVolumeRange()

minVol = volumeRan[0]
maxVol = volumeRan[1]
vol =0
volb = 400

while True:
    success, img = cam.read()
    img = detector.findHands(img)
    lmlist = detector.findpost(img, draw=False)
    if len(lmlist) != 0:
        # print(lmlist[2])

        x1,y1= lmlist[4][1],lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cx,cy =(x1+x2)//2, (y1 + y2) // 2

        cv2.circle(img,(x1,y1), 15, (255,0,0),cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.circle(img,(cx,cy),15,(255,0,0),3)

        length = math.hypot(x2-x1,y2-y1)
        # print(length)

        vol = np.interp(length,[50,300],[minVol,maxVol])
        volb = np.interp(length,[50,300],[400,150])

        volume.SetMasterVolumeLevel(vol, None)

        if length < 40:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
    cv2.rectangle(img, (50,150), (85,400),(0,255,0),3)
    cv2.rectangle(img, (50, int(volb)), (85, 400), (0, 255, 0), cv2.FILLED)

    Ctime = time.time()
    fps = 1/(Ctime-pTime)
    pTime = Ctime
    cv2.putText(img,str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break