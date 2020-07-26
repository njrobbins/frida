"""
Fall Rate Identification, Detection, & Analysis (FRIDA) program
For CSC 450 Course Project at Missouri State University
Contributors: Jonah Falk, Samuel Pete, Normandy River, Niko Robbins, Jacob Schmoll
"""

import cv2
import time

cameraPort = 0  # 0 is system's default webcam
camera = cv2.VideoCapture(int(cameraPort), cv2.CAP_DSHOW)  # Initiates video stream
videoBrightness = 150
camera.set(10, videoBrightness)
time.sleep(1)  # Gives the camera's auto-focus & auto-saturation time to load
fgbg = cv2.createBackgroundSubtractorMOG2()  # Initiates a background subtraction/foreground detection process
frameCount = 0  # Used to delay fall detection to prevent false positives
detectionTest = 0  # Used to keep track of successful detection
personDetected = 0  # Track amount of frames detected
percentageDetected = 0  # Percentage of frames person is detected
movementCount = 0  # Keeps count of frames for movement updates
movementState = "Not Moving"  # Starts out assuming person is standing still

if not camera.isOpened():
    raise IOError("CANNOT OPEN WEBCAM")

while True:
    ret, frame = camera.read()  # Reads video frames
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
        # _, contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Use with OpenCvVersion < 4

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
                personDetected += 1

            if frameCount > 75:  # After ~2 seconds (at 30 fps) of being fallen down; will need to be optimized
                print("FALL DETECTED")  # Prints this every time a fall is detected

            if h > w:
                start_point = (x, y)  # Gives reference for where person starts
                cv2.rectangle(frame, start_point, (x + w, y + h), (0, 255, 0),
                              2)  # Puts green rectangle around detected object

                if movementCount == 0:  # Sets checkPoint to startPoint to keep track of original location
                    checkPoint = start_point
                frameCount = 0
                personDetected += 1
                movementCount += 1  # Increase MovementCount so it can be tracked each second

                if movementCount > 30 and checkPoint == start_point:
                    movementState = "No Movement"  # Updates state to No Movement if target does not move for one second
                    movementCount = 0

                if movementCount > 30 and checkPoint != start_point:
                    movementState = "Moving"  # Updates state to Moving if target moves during the second
                    movementCount = 0


            cv2.imshow("Video Feed", frame)  # Loads video feed window

            detectionTest += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Press 'q' to terminate video feed
                percentageDetected = personDetected / detectionTest
                print(percentageDetected * 100, "% of frames detected person")
                print("VIDEO FEED TERMINATED")
                print("Movement State: ", movementState)
                camera.release()
                cv2.destroyAllWindows()

    except Exception as e:
        print (e)
        break