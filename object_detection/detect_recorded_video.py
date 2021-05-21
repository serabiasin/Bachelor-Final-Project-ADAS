import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from ssd_predict import detector
import cv2
import os
import datetime
from PIL import Image

video_input_path='/content/drive/MyDrive/TUGAS_AKHIR/Uji_Video/DrivinginToronto-SaturdayMorningDrive-February 2017-Front Dash Cam.mp4'

if tf.__version__ < '1.4.0':
    raise ImportError(
        'Please upgrade your tensorflow installation to v1.4.* or later!')

FILE_OUTPUT = '/content/drive/MyDrive/DrivinginToronto-SaturdayMorningDrive-February2017-FrontDashCam.avi'

# Checks and deletes the output file
# You cant have a existing file or it will through an error
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)

# Playing video from file
cap = cv2.VideoCapture(video_input_path)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'output.avi' file.
out = cv2.VideoWriter(FILE_OUTPUT, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                      20, (frame_width, frame_height))

sys.path.append("..")

# Object detection imports
# Here are the imports from the object detection module.

# Model preparation
NETWORK = "mobilenetv1" # Choose an option in `mobilenetv1`, `mobilenetv2` or `mobilenetv3`
WEIGHT_FILE = "/content/drive/MyDrive/SSD-MobilenetSSD-KITTI-336.h5" # The network weigth file path.
det = detector(weight_path=WEIGHT_FILE, network=NETWORK)

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

            #SSD MOBILENET

            #convert to PIL
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame=Image.fromarray(frame)
    frame = det.detect_image(frame)

    #revert back to opencv format image
    numpy_image=np.array(frame)  
    opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 

    if ret == True:
        # Saves for video
        out.write(opencv_image)

    else:
        break

    # When everything done, release the video capture and video write objects
cap.release()
out.release()
