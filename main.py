"""
Fall Rate Identification, Detection, & Analysis (FRIDA) program
For CSC 450 Course Project at Missouri State University
Contributors: Jonah Falk, Samuel Pete, Normandy River, Niko Robbins, Jacob Schmoll
License: GNU GPLv3
"""

from sklearn.utils.extmath import softmax
import cv2
import onnx
import onnxruntime as ort
import numpy as np
import numpy.random
import sys
import tensorflow as tf
import time
import torch
import torchvision
import itertools
import threading

# Uncomment for debugging purposes.
# print("onnx version:", onnx.__version__)
# print("onnxruntime version:", ort.__version__)
# print("opencv version:", cv2.__version__)
# print("tensorflow version:", tf.__version__)
# print("torchvision version:", torchvision.__version__)


def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rLoading ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rFRIDA software loaded.\n')


done = False  # Program hasn't yet loaded video.
t = threading.Thread(target=animate)
t.start()

# Only change cameraPort for debugging purposes.
cameraPort = 0  # 0 = system's default webcam (recommended), 1 = external webcam, -1 = auto-detection
# Option 1: Use this for live video feed via a webcam. Press 'q' to terminate.
camera = cv2.VideoCapture(cameraPort)
# Option 2: Use this for a video file. Refer to the adl, fallcam0, & fallcam1 dataset folders.
# The videos are short and will terminate once finished.
# Make sure to comment out option 1 for VideoCapture (above) if used.
# camera = cv2.VideoCapture('fallcam1/fall30cam1.mp4')

camera.set(cv2.CAP_PROP_FPS, 30)  # Sets frames per second (FPS).
videoBrightness = 150
camera.set(10, videoBrightness)
time.sleep(1)  # Gives the camera's auto-focus & auto-saturation time to load.

# Explain this group
onnxModel = onnx.load('model.onnx')
model = cv2.dnn.readNetFromONNX('model.onnx')
sess = ort.InferenceSession('model.onnx')

batch = []
batchSize = 32
color = np.random.randint(0, 255, (100, 3))
condenseBatch = []
countFrame = 0
HUD = 1  # Heads Up Display. Set to 0 to turn off.
inputName = sess.get_inputs()[0].name
prevFrame = None
prev_mhi = None

if not camera.isOpened():
    raise IOError("CANNOT OPEN WEBCAM")

while True:
    grabbed, frame = camera.read()
    if not grabbed:
        break

    try:
        countFrame += 1
        frame_x = frame
        img2 = np.zeros_like(frame)
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY)
        img2[:, :, 0] = frame
        img2[:, :, 1] = frame
        img2[:, :, 2] = frame
        frame = img2
        dim = (224, 256)
        dims = (256, 224, 3)
        mhi_zeros = np.zeros(dims)
        detectStatus = "Idle"

        if countFrame == 1:
            prevFrame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            prev_mhi = mhi_zeros

        else:
            resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            diff = cv2.absdiff(prevFrame, resized)
            binary = (diff >= (.41 * 255)).astype(np.uint8)
            mhi = binary + (binary == 0) * np.maximum(mhi_zeros, (prev_mhi - 1 / 16))
            prev_frame = resized
            prev_mhi = mhi
            img2 = mhi

        img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)
        frames = np.expand_dims(img2, axis=0)
        frames = np.array(frames)
        frames = frames.astype(numpy.float32)
        image = torch.from_numpy(frames)
        image = image.permute(0, 3, 1, 2)

        # This is for the append current frame to next frame
        # if(countFrame == 1):
        # result = np.array(image)
        # elif(countFrame > 1 and countFrame %2 == 0):
        # result = np.array(image)
        # condense = np.array(image)'''

        # This is for normal use current frame then empty and take next frame
        result = np.array(image)

        if len(batch) != 32:
            batch.append(result)
            # (Alternative) Only used for condensed-spaced model:
            # condenseBatch.append(condense)

        if len(batch) > 32:
            batch = batch[:32]
            batch = np.array(batch)

        if len(batch) == 32:
            result_x = np.concatenate(batch, axis=0)
            x = result_x
            res = sess.run(None, {inputName: x})
            norm = softmax(res[0])

            for x in norm:
                fall = x.item(0)
                notFall = x.item(1)
                # FP = Fall Prediction, NFP = Non-Fall Prediction
                print("FP", "{0:.2%}".format(fall),
                      "NFP", "{0:.2%}".format(notFall))
                if fall > notFall:
                    detectStatus = "FALL DETECTED"
                    print(detectStatus)
                    batch = []

            # (Default) Used for the regular current frame, then next frame is its own frame also:
            batch = []
            # (Alternative) Used to test the appending of the current frame to the next frame:
            # batch = condenseBatch[16::]
            # condenseBatch = []

        if HUD:
            if detectStatus == "FALL DETECTED":
                cv2.putText(frame, "Status: {}".format(detectStatus), (10, 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1)
            else:
                cv2.putText(frame, "Status: {}".format(detectStatus), (10, 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 128, 0), 1)

        # (Default) Loads video frame window in grayscale:
        cv2.imshow("Video Feed", frame)
        # (Optional) Loads video frame window using background subtraction:
        # cv2.imshow("Background Subtraction", img2)

        done = True  # Video has successfully loaded.

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nVIDEO FEED TERMINATED\n")
            camera.release()
            cv2.destroyAllWindows()
            break

    except Exception as e:
        print(e)
        break
