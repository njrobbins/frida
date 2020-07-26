"""
Fall Rate Identification, Detection, & Analysis (FRIDA) program
For CSC 450 Course Project at Missouri State University
Contributors: Jonah Falk, Samuel Pete, Normandy River, Niko Robbins, Jacob Schmoll
"""

import cv2
import time

import numpy.random

from sklearn.utils.extmath import softmax


print(cv2.__version__)
import tensorflow as tf
print("tf ", tf.__version__)


import onnx
import numpy as np
print(onnx.__version__)


import onnxruntime as rt
print("onnxruntime: ", rt.__version__)

import torch

import torchvision
print("torchvision ", torchvision.__version__)
import onnxruntime as ort


import os





onnx_model = onnx.load('model.onnx')


model =cv2.dnn.readNetFromONNX('model.onnx')


sess = ort.InferenceSession('model.onnx')
input_name = sess.get_inputs()[0].name

batch_size = 32
batch = []




cameraPort = 0  # 0 is system's default webcam. 1 is an external webcam. -1 is auto-detection.
# cameraPort = input("What kind of webcam are you using?\nEnter 0 for a built-in webcam.\nEnter 1 for an external webcam.\n")
# print("Loading...")
#use this for accessing live video feed
camera = cv2.VideoCapture(cameraPort)  # Initiates video stream and use to input mp4 to test

#use this when using video input and change cameraPort to name of video file and the extension so nameofvideofile.mp4 for example
#camera = cv2.VideoCapture(cameraPort)
camera.set(cv2.CAP_PROP_FPS, 32) #set frames per second
videoBrightness = 150
camera.set(10, videoBrightness)
time.sleep(1)  # Gives the camera's auto-focus & auto-saturation time to load

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



color = np.random.randint(0,255,(100,3))

count_frame = 0
frame_counter = 0
while True:
    ret, frame = camera.read()

    count_frame +=1








    frame_x = frame
    img2 = np.zeros_like(frame)
    #use for video input
    #frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY)
    #use for live video access
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    img2[:,:,0] = frame
    img2[:,:,1] = frame
    img2[:,:,2] = frame
    frame = img2
    dim = (224, 256)
    dims = (256, 224, 3)

    mhi_zeros = np.zeros(dims)


    if(count_frame == 1):

        prev_frame  = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA )
        prev_mhi = mhi_zeros

    else:
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA )
        diff = cv2.absdiff(prev_frame, resized)

        binary = (diff >= (.4 * 255)).astype(np.uint8)


        mhi = binary + (binary == 0) * np.maximum(mhi_zeros,
                                                      (prev_mhi-1/32))

        prev_frame = resized
        prev_mhi = mhi

        img2 = mhi







    img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
    frames = np.expand_dims(img2, axis=0)

    frames = np.array(frames)
    frames = frames.astype(numpy.float32)


    image = torch.from_numpy(frames)


    image = image.permute(0, 3, 1, 2)

    #This is for the append current frame to next frame
    #if(count_frame == 1):
     #   result = np.array(image)
    #elif(count_frame > 1 and count_frame %2 == 0):
     #   result = np.array(image)

     #this is for normal use current frame then empty and take next frame
    result = np.array(image)





    try:

        if( len(batch) != 32):
                batch.append(result)



        if(len(batch) > 32):
            batch = batch[:32]
            batch = np.array(batch)



        if(len(batch) == 32):



            result_x = np.concatenate(batch, axis=0)



            x = result_x



            res = sess.run(None, {input_name: x })

            norm = softmax(res[0])






            for x in norm:
                fall = x.item(0)
                not_fall = x.item(1)




                print("predict", np.round(fall, decimals=3), np.round(not_fall, decimals=3))
                if(fall > not_fall):
                    print("fall detected")
                    batch = []
                    break

                # here we are testing the appending of current frame to next frame
            #batch = batch[31:15:-1]
            #predict 0.516 0.484


            #here we are doing just the regular current frame and then next frame is its own frame as well
            batch = []
            #predict 0.52 0.48


        cv2.imshow("Video Feed", img2)




        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to terminate video feed
            print("VIDEO FEED TERMINATED")
            camera.release()
            cv2.destroyAllWindows()
            break
    except Exception as e:
        print(e)# Can be edited
        break