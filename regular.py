"""
Fall Rate Identification, Detection, & Analysis (FRIDA) program
For CSC 450 Course Project at Missouri State University
Contributors: Jonah Falk, Samuel Pete, Normandy River, Niko Robbins, Jacob Schmoll
License: GNU GPLv3
Regular Model
"""

# SRS: SIR.1, SIR.2, SIR.3
import cv2
import itertools
import numpy as np
import numpy.random
import onnx
import onnxruntime as ort
from sklearn.utils.extmath import softmax
import sys
import threading
import time
import torch

# Uncomment for debugging purposes:
# print("opencv version:", cv2.__version__)
# print("onnx version:", onnx.__version__)
# print("onnxruntime version:", ort.__version__)
# print("torch version:", torch.__version__)


# SDD:
class LoadModel:
    def __init__(self):
        self.onnx_model = onnx.load('model.onnx')
        self.model = cv2.dnn.readNetFromONNX('model.onnx')
        self.sess = ort.InferenceSession('model.onnx')
        self.input_name = self.sess.get_inputs()[0].name


# (Video Option 1) Use this for live video feed via a webcam.
# Press 'q' to terminate.
# SDD:
# SRS: 3.1.3, UI.2
class CameraSetUpLiveVideo:
    def __init__(self, port=-1):
        self.cameraPort = port
        self.camera = cv2.VideoCapture(self.cameraPort)
        self.camera.set(cv2.CAP_PROP_FPS, 32)  # Sets frames per second (FPS).
        video_brightness = 150
        self.camera.set(10, video_brightness)
        time.sleep(1)  # Gives the camera's auto-focus & auto-saturation time to load.


# (Video Option 2) Use this for video file playback.
# Video files will terminate once finished.
# SDD:
# SRS: 3.1.20, UI.2
class CameraSetUpVideoPlayBack:
    def __init__(self, path_to_video):
        self.path_to_video = path_to_video
        self.camera = cv2.VideoCapture(self.path_to_video)
        self.camera.set(cv2.CAP_PROP_FPS, 32)  # Set frames per second (FPS).
        video_brightness = 150
        self.camera.set(10, video_brightness)
        time.sleep(1)  # Gives the camera's auto-focus & auto-saturation time to load.


# SDD:
# SRS: 3.2.13
class TransformShape:
    def __init__(self, frame):
        self.frame = frame
        self.frame_transform = np.zeros_like(self.frame)


# SDD:
# SRS: 3.2.13
class MotionHistoryTransform:
    dim = (224, 256)

    def __init__(self, frame):
        self.frame = frame
        self.dims = (256, 224, 3)
        self.dim = (224, 256)
        self.mhi_zeros = np.zeros(self.dims)
        self.prev_frame = cv2.resize(self.frame, self.dim, interpolation=cv2.INTER_AREA)


# SDD:
# SRS: 3.2.13
class MotionHistoryDifference:
    def __init__(self, frame):
        self.frame = frame
        self.resized = cv2.resize(self.frame, MotionHistoryTransform.dim, interpolation=cv2.INTER_AREA)


# SDD:
# SRS: 3.2.13
class CreateBatch:
    def __init__(self):
        self.batch_size = 32
        self.batch = []
        self.condense_batch = []


# SRS: CIR.1
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

regularModel = LoadModel()
batchCreate = CreateBatch()

# (Video Option 1) Use this class for live video feed.
# 0 = system's default webcam (recommended), 1 = external webcam, -1 = auto-detection
# Only change argument for debugging purposes.
# camera = CameraSetUpLiveVideo(0)

# (Video Option 2) Use this class for video file playback.
# Change "fallcam0/fall1cam0.mp4" to the video file of your choice.
# Refer to the adl, fallcam0, & fallcam1 dataset folders.
camera = CameraSetUpVideoPlayBack("fallcam0/fall1cam0.mp4")

color = np.random.randint(0, 255, (100, 3))
countFrame = 0
HUD = 1  # Heads Up Display for live video feed. Set to 0 to turn off.
mhi_maker = None
prev_mhi = None

# SRS: 3.1.3, 3.1.20, CIR.1
if not camera.camera.isOpened():
    raise IOError("CANNOT LOAD VIDEO FRAME")

while True:
    # SRS: 3.1.3, 3.1.20
    grabbed, frame = camera.camera.read()
    if not grabbed:
        break

    try:
        countFrame += 1
        frameTransform = TransformShape(frame)
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY)
        frameTransform.frame_transform[:, :, 0] = frame
        frameTransform.frame_transform[:, :, 1] = frame
        frameTransform.frame_transform[:, :, 2] = frame
        frame = frameTransform.frame_transform

        if countFrame == 1:
            mhi_maker = MotionHistoryTransform(frame)
            prev_mhi = mhi_maker.mhi_zeros

        else:
            mhi_difference_maker = MotionHistoryDifference(frame)
            diff = cv2.absdiff(mhi_maker.prev_frame, mhi_difference_maker.resized)
            binary = (diff >= (.41 * 255)).astype(np.uint8)
            mhi = binary + (binary == 0) * np.maximum(mhi_maker.mhi_zeros, (prev_mhi - 1 / 16))
            mhi_maker.prev_frame = mhi_difference_maker.resized
            prev_mhi = mhi
            frameTransform.frame_transform = mhi

        frameTransform.frame_transform = cv2.resize(frameTransform.frame_transform,
                                                    MotionHistoryTransform.dim, interpolation=cv2.INTER_AREA)
        frames = np.expand_dims(frameTransform.frame_transform, axis=0)
        frames = np.array(frames)
        frames = frames.astype(numpy.float32)
        image = torch.from_numpy(frames)
        image = image.permute(0, 3, 1, 2)
        detectStatus = "Idle"

        # Uses the current frame, empties it, & takes the next frame.
        result = np.array(image)

        if len(batchCreate.batch) != 32:
            batchCreate.batch.append(result)

        if len(batchCreate.batch) > 32:
            batchCreate.batch = batchCreate.batch[:32]
            batchCreate.batch = np.array(batchCreate.batch)

        if len(batchCreate.batch) == 32:
            result_x = np.concatenate(batchCreate.batch, axis=0)
            model_input_x = result_x
            res = regularModel.sess.run(None, {regularModel.input_name: model_input_x})
            norm = softmax(res[0])

            for x in norm:
                fall = x.item(0)
                notFall = x.item(1)
                # SRS: CIR.1
                # FP = Fall Prediction, NFP = Non-Fall Prediction
                print("FP", "{0:.2%}".format(fall), "NFP", "{0:.2%}".format(notFall))
                # SRS: 3.1.6, 3.1.22, CIR.1, CIR.2
                if fall > notFall:
                    detectStatus = "FALL DETECTED"
                    print(detectStatus)
                    batchCreate.batch = []
                    break

            # The current frame and the next frame are separate.
            batchCreate.batch = []

        # SRS: CIR.2
        if HUD:
            if detectStatus == "\nFALL DETECTED\n":
                cv2.putText(frame, "Status: {}".format(detectStatus),
                            (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1)
            else:
                cv2.putText(frame, "Status: {}".format(detectStatus),
                            (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 128, 0), 1)

        # (Default) Loads video frame window in grayscale.
        # SRS: UI.2
        cv2.imshow("Video Feed", frame)

        # (Optional) Loads video frame window using background subtraction.
        # SRS: UI.2
        # cv2.imshow("Background Subtraction", frameTransform.frame_transform)

        done = True  # Video has successfully loaded.

        # SRS: CIR.1, CIR.3
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nVIDEO FEED TERMINATED\n")
            camera.camera.release()
            cv2.destroyAllWindows()
            break

    # SRS: CIR.1
    except Exception as e:
        print(e)
        break
