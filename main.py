"""
Fall Rate Identification, Detection, & Analysis (FRIDA) program
Contributors: Jonah Falk, Samuel Pete, Normandy River, Niko Robbins, Jacob Schmoll
For CSC 450 at Missouri State University
"""

import cv2
import time

cameraPort = 0  # 0 is system's default camera. May need to change to 1 for external webcam.
cap = cv2.VideoCapture(cameraPort)  # Initiates video stream
videoBrightness = 150
cap.set(10, videoBrightness)
time.sleep(1)  # Gives the camera's auto-focus & auto-saturation time to load
fgbg = cv2.createBackgroundSubtractorMOG2()  # Initiates a background subtraction/foreground detection process
frameCount = 0  # Used to delay fall detection to prevent false positives

while True:
    ret, frame = cap.read()  # Reads video frames

    # Convert each frame to grayscale & subtract the background
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours

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
                frameCount += 1

            if frameCount > 75:  # After ~2 seconds (at 30 fps) of being fallen down; will need to be optimized
                print("FALL DETECTED")  # Prints this every time a fall is detected
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Puts red rectangle around fallen object

            if h > w:
                frameCount = 0
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Puts green rectangle around detected object

            cv2.imshow("Video Feed", frame)  # Loads video feed window

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Press 'q' to terminate video feed
                cap.release()
                cv2.destroyAllWindows()
                print("Video feed terminated.")

    except Exception as e:  # Can be edited
        break

