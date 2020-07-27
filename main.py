"""
Fall Rate Identification, Detection, & Analysis (FRIDA) program
For CSC 450 Course Project at Missouri State University
Contributors: Jonah Falk, Samuel Pete, Normandy River, Niko Robbins, Jacob Schmoll
License: GNU GPLv3
"""

import cv2
import itertools
import sys
import threading
import time

done = False  # Program hasn't yet loaded video


def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rLoading ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rFRIDA software loaded.\n')


t = threading.Thread(target=animate)  # Ties thread to animate()
t.start()  # Starts threading process
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Initiates live video stream; comment out if using video file option
# camera = cv2.VideoCapture("file.mov")  # Option to instead use video file input; replace "file.mov" with video file
camera.set(10, 150)  # Sets brightness
time.sleep(1)  # Gives the camera's auto-focus & auto-saturation time to load
fgbg = cv2.createBackgroundSubtractorMOG2()  # Initiates a background subtraction/foreground detection process

detectionTest = 0  # Used to keep track of successful detection
firstFrame = None  # Used as the initial background frame
frameCount = 0  # Used to delay fall detection to prevent false positives
HUD = 1  # Heads Up Display; set to 0 to turn off
minArea = 50 * 50  # Minimum to accepted as a person; can be optimized
movementCount = 0  # Keeps count of frames for movement updates
movementState = "Not Moving"  # Starts out assuming person is standing still
personDetected = 0  # Track amount of frames detected
percentageDetected = 0  # Percentage of frames person is detected

if not camera.isOpened():
    raise IOError("CANNOT OPEN WEBCAM")

while True:
    grabbed, frame = camera.read()  # Reads video frames
    if not grabbed:
        break

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)

        # Updates every time light conditions change
        if firstFrame is None:
            time.sleep(1)
            grabbed, frame = camera.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = fgbg.apply(gray)
            firstFrame = gray
            continue

        # Compares difference between the current frame and background frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=30)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
        # _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Use with OpenCV version <4

        detectStatus = "Idle"
        areas = []
        for contour in contours:
            if cv2.contourArea(contour) < minArea:
                continue
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

            if frameCount > 150:  # After ~5 seconds (at 30 fps) of being fallen down; can be optimized
                detectStatus = "FALL DETECTED"
                print(detectStatus)  # Prints this every time a fall is detected

            if h > w:
                start_point = (x, y)  # Gives reference for where person starts
                cv2.rectangle(frame, start_point, (x + w, y + h), (0, 255, 0),
                              2)  # Puts green rectangle around detected object

                if movementCount == 0:  # Sets checkPoint to startPoint to keep track of original location
                    checkPoint = start_point

                frameCount = 0
                personDetected += 1
                movementCount += 1  # Increases movementCount so it can be tracked each second

                if movementCount > 30 and checkPoint == start_point:
                    movementState = "Not Moving"  # Updates state to No Movement if target does not move for one second
                    movementCount = 0

                if movementCount > 30 and checkPoint != start_point:
                    movementState = "Moving"  # Updates state to Moving if target moves during the second
                    movementCount = 0

            if HUD:
                cv2.putText(frame, "Status: {}".format(detectStatus), (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                            (0, 0, 0), 1)

            cv2.imshow("Video Frame", frame)  # Loads video frame window
            done = True  # Video has successfully loaded
            detectionTest += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Press 'q' to terminate video feed
                print("\nVIDEO TERMINATED")
                percentageDetected = personDetected / detectionTest
                print("{0:.2%}".format(percentageDetected), "of frames detected a person")
                print("Final Movement State:", movementState, "\n")
                camera.release()
                cv2.destroyAllWindows()
                break

    except Exception as e:
        print(e)
        break
