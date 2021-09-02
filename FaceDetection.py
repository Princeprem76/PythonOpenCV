import cv2
import mediapipe as mp
import time


cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS,60)
pTime = 0

mpFace = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetect = mpFace.FaceDetection(0.8)

while True:
    success, img = cam.read()
    imgRBG = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = faceDetect.process(imgRBG)
    # print(result)

    if result.detections:
        for id, detect in enumerate(result.detections):
            # mpDraw.draw_detection(img,detect)
            # print(id,detect)
            # detect.score
            ih, iw, ic = img.shape
            bbox = detect.location_data.relative_bounding_box
            bboxC = int(bbox.xmin * iw),int(bbox.ymin* ih),\
                    int(bbox.width * iw), int(bbox.height*ih)
            cv2.rectangle(img,bboxC,(0,255,0),2)
            cv2.putText(img,f'{int(detect.score[0]*100)}%',
                        (bboxC[0],bboxC[1]-20),cv2.FONT_HERSHEY_PLAIN,
                        2,(0,255,0),2)

    Ctime = time.time()
    fps = 1/(Ctime-pTime)
    pTime = Ctime
    cv2.putText(img, f'Fps: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)

    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break