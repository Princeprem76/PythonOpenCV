import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd= 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

img = cv2.imread('Images/2.png')
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
himg,wimg,_ = imgRGB.shape
boxes = pytesseract.image_to_boxes(imgRGB)
for bo in boxes.splitlines():
    b = bo.split(' ')
    x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
    cv2.rectangle(imgRGB,(x,himg-y),(w,himg-h),(0,255,0),1)
    cv2.putText(imgRGB,b[0],(x,himg-y+30),cv2.FONT_HERSHEY_PLAIN,1,(50,50,0),2)




cv2.imshow("Image",imgRGB)

cv2.waitKey(0)