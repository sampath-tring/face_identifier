import cv2
import numpy as np
import sys


def save_face(name):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

    while True:
      ret, img = cap.read()
      ret1, org = cap.read()
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30, 30))
      #cv2.putText(img,'Tap on the Enter key to feed your photo to our AI engine.',(20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,255,155), 2,cv2.LINE_AA)
      for(x,y,w,h) in faces:
        img1 = cv2.rectangle(img,(x,y),(x+w,y+h),(200,0,200),2)
        #cv2.putText(img,"Save your Pic press s !!!", (x,y), cv2.FONT_HERSHEY_SIMPLEX 2, 255)
        cv2.putText(img,'Press S key to save your pic.',(x-100,y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,255,155), 2, cv2.LINE_AA)        
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img1[y:y+h,x:x+w]

      cv2.imshow('frame',img)  
         
      if cv2.waitKey(1) & 0xFF == ord('S'):
        cv2.imwrite("./pics/"+name+".jpg",org)
        break 
      
      if cv2.waitKey(1) & 0xFF == ord('Q'):
        break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  save_face(sys.argv[1])
