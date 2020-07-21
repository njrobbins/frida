"""
Fall Rate Identification, Detection, & Analysis (FRIDA) program
For CSC 450 Course Project at Missouri State University
Contributors: Jonah Falk, Samuel Pete, Normandy River, Niko Robbins, Jacob Schmoll
"""

import cv2
import time
import caffe2.python.onnx.backend as backend
import numpy.random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.utils.extmath import softmax

print(cv2.__version__)


import onnx
import numpy as np
print(onnx.__version__)

import onnxruntime as rt
print("onnxruntime: ", rt.__version__)
#import onnxruntime as rt
from onnx import numpy_helper
import torch
import torchvision
import onnxruntime as ort
threshold=0.1
interval=2
duration=40

onnx_model = onnx.load('model.onnx')


model =cv2.dnn.readNetFromONNX('model.onnx')

'''dummy_input = torch.randn(10, 3, 224, 224, device='cpu')'''






'''print(sess)
print("The model expects input shape: ", sess.get_inputs()[0].shape)

input_name = sess.get_inputs()[0].name
print("input name", input_name)
input_shape = sess.get_inputs()[0].shape
print("input shape", input_shape)
input_type = sess.get_inputs()[0].type
print("input type", input_type)

output_name = sess.get_outputs()[0].name
print("output name", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)
output_type = sess.get_outputs()[0].type
print("output type", output_type)

x = numpy.random.random((32, 3, 256, 224))
x = x.astype(numpy.float32)
res = sess.run(None, {input_name: x})'''

'''print("predict", res[0][:])
print("predict_proba", res[1][:1])
print(res)
print("predict_proba", res[0][:1])
print(res)
print("predict_proba", res[0][:1])
outputTensor = res.values().next().value
predictions = outputTensor.data
maxPrediction = max(predictions)
print(maxPred)'''

sess = ort.InferenceSession('model.onnx')
input_name = sess.get_inputs()[0].name

batch_size = 32
batch = []


#session = rt.InferenceSession('model.onnx')
#input_name = session.get_inputs()[0].name
#print(input_name)
#output_name = session.get_outputs()[0].name
#print(output_name)

cameraPort = 0  # 0 is system's default webcam. 1 is an external webcam. -1 is auto-detection.
# cameraPort = input("What kind of webcam are you using?\nEnter 0 for a built-in webcam.\nEnter 1 for an external webcam.\n")
# print("Loading...")
camera = cv2.VideoCapture(cameraPort)  # Initiates video stream
videoBrightness = 150
camera.set(10, videoBrightness)
time.sleep(1)  # Gives the camera's auto-focus & auto-saturation time to load
fgbg = cv2.createBackgroundSubtractorMOG2()  # Initiates a background subtraction/foreground detection process
frameCount = 0  # Used to delay fall detection to prevent false positives



#if not camera.isOpened():
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

        #print("WEBCAM NOT DETECTED")

#else:
    # Returns confirmation to user if valid or error if invalid
    # if int(cameraPort) == 0 or int(cameraPort) == 1:
    #     print("Thank you. Webcam confirmed.")
    # else:
    #     print("ERROR. INPUT MUST BE 0 OR 1.")
while True:
    ret, frame = camera.read()
    # Reads video frames
    dim = (224, 256)
    # resize image
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    frames = np.expand_dims(resized, axis=0)
    frames = np.array(frames)
    frames = frames.astype(numpy.float32)
    #frames *= 255.0/frames.max()

    image = torch.from_numpy(frames)

    image = image.permute(0, 3, 1, 2)


    result = np.array(image)
    #shape = result.shape
    #result = minmax_scale(result.ravel(), feature_range=(0,1)).reshape(shape)





    #a = np.array(a)[indices.astype(int)]
    #a.reshape(a, (1,3,480,640))

    try:
        if( len(batch) != 32):
            batch.append(result)
        if(len(batch) > 32):
            batch = batch[:32]
            batch = np.array(batch)



        if(len(batch) == 32):
            result = np.concatenate(batch, axis=0)



            x = result

            #x = result.astype(numpy.float32)
            #x *= 255.0/x.max()
            #x /= x.max()/255.0
            #y =  numpy.linalg.norm(x, ord=3.5, axis=2, keepdims=True)
            #x = x/y
            res = sess.run(None, {input_name: x })
            #norm = softmax(res[0])
            #y =  numpy.linalg.norm(x, ord=3.5, axis=2, keepdims=True)
            norm = softmax(res[0])

            #norm =  norm/numpy.linalg.norm(norm, ord=2, axis=1, keepdims=True)



            print("predict", np.round(norm, decimals=3))
            batch = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fgmask = fgbg.apply(gray)
        #below is for opencv  4
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours



        #below is for opencv under 4
        #_, contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours

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
    except Exception as e:
        print(e)# Can be edited
        break