#Github Cascades (from https://github.com/opencv/opencv/tree/master/data/haarcascades)

import cv2
import numpy as np
import sys
##########################################################
frameWidth = 500
frameHeight = 480
FullFaceCascade=cv2.CascadeClassifier("FrontalFace.xml")
minArea = 300
color = (255,255,255)
##########################################################
cap = cv2.VideoCapture(0) #0 is default camera. May need to change to 1 for usb / external webcam.
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,150)

def main():
    frame_counter = 0 #may use a fall_frame counter for delay; lower chance for false positive.
    while True: #Infinite loop, camera feed runs infinitely
        try:
            success,img = cap.read()
            imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            area_of_interest = FullFaceCascade.detectMultiScale(imgGray,1.5,1,minSize=(100,100))
            for (x,y,w,h) in area_of_interest:
                frame_counter += 1 #If area of interest detected, add to counter
                cv2.rectangle(img,(x,y),(x+w, y+h),(255,255,255),2) #draws the rectangle around area of interest
                cv2.putText(img,"AreaOfIterest",(x,y-5),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
            if frame_counter > 50: #50 frames = about 5 seconds. Change as needed
                EndCamera(cap_counter)
            cv2.imshow("Video",img)
            if cv2.waitKey(1)&0xFF == ord('q'): #q key will end camera feed. Can change to optimize.
                sys.exit()
                cv2.destoryAllWindows()
                break
        except:
            print("Face Detected. Video Feed Ended.")
            sys.exit()

def EndCamera(cap_counter):
    print(cap_counter,"frames detected")
    cap.release()
    cv2.destroyAllWindows()      

main()
