"""
Fall Rate Identification, Detection, & Analysis (FRIDA) program
For CSC 450 Course Project at Missouri State University
Contributors: Jonah Falk, Samuel Pete, Normandy River, Niko Robbins, Jacob Schmoll
"""

import cv2
import time

cameraPort = 0  # 0 is system's default webcam. 1 is an external webcam. -1 is auto-detection.
# cameraPort = input("What kind of webcam are you using?\nEnter 0 for a built-in webcam.\nEnter 1 for an external webcam.\n")
# print("Loading...")
camera = cv2.VideoCapture(cameraPort)  # Initiates video stream
videoBrightness = 150
camera.set(10, videoBrightness)
time.sleep(1)  # Gives the camera's auto-focus & auto-saturation time to load
fgbg = cv2.createBackgroundSubtractorMOG2()  # Initiates a background subtraction/foreground detection process
frameCount = 0  # Used to delay fall detection to prevent false positives

if not camera.isOpened():
    # if int(cameraPort) == 0:
    #     try:
    #         camera = cv2.VideoCapture(1)
    #     except:
    #         camera = cv2.VideoCapture(-1)
    #         if not camera.isOpened():
    #             print("WEBCAM NOT DETECTED")
    # elif int(cameraPort) == 1:
    #     try:
    #         camera = cv2.VideoCapture(1)
    #     except:
    #         camera = cv2.VideoCapture(-1)
    #         if not camera.isOpened():
    #             print("WEBCAM NOT DETECTED")
    # else:
    #     print("ERROR. INPUT MUST BE 0 OR 1.")

        print("WEBCAM NOT DETECTED")

else:
    # Returns confirmation to user if valid or error if invalid
    # if int(cameraPort) == 0 or int(cameraPort) == 1:
    #     print("Thank you. Webcam confirmed.")
    # else:
    #     print("ERROR. INPUT MUST BE 0 OR 1.")
    while True:
        ret, frame = camera.read()  # Reads video frames
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = fgbg.apply(gray)
            _, contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours

            if contours:
                areas = []  # A list to hold all the areas

                for contour in contours:
                    ar = cv2.contourArea(contour)  # Calculate the area of each contour
                    areas.append(ar)

                max_area = max(areas, default=0)
                max_area_index = areas.index(max_area)
                cnt = contours[max_area_index]
                M = cv2.moments(cnt)  # Calculates moments of detected binary image
                x, y, w, h = cv2.boundingRect(cnt)  # Calculates an upright bounding rectangle

                cv2.drawContours(fgmask, [cnt], 0, (255, 255, 255), 3, maxLevel=0)

                if h < w:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255),
                                  2)  # Puts red rectangle around fallen object
                    frameCount += 1

                if frameCount > 75:  # After ~2 seconds (at 30 fps) of being fallen down; will need to be optimized
                    print("FALL DETECTED")  # Prints this every time a fall is detected

                if h > w:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),
                                  2)  # Puts green rectangle around detected object
                    frameCount = 0

                cv2.imshow("Video Feed", frame)  # Loads video feed window

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Press 'q' to terminate video feed
                    print("VIDEO FEED TERMINATED")
                    camera.release()
                    cv2.destroyAllWindows()
                    break
        except:  # Can be edited
            break