# Driver-Fatigue-Detection-with-OpenCV-and-Deep-Learning 

## Research paper for the app : [Click here](https://github.com/rkf2778/Driver-Fatigue-Detection-with-OpenCV-and-Deep-Learning/blob/master/ICOEI_2019_paper_247.pdf)

## These are the following modules used in this project:

```python
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import playsound
import os
from threading import Thread
```


Requirements.txt 

```txt
imutils==0.5.4
numpy==1.21.5
opencv-python==4.5.4.68
scipy==1.7.3
dlib==19.22.0
playsound==1.3.0
```


## To install the module, use pip
```python
pip install <module-name>
```


## To run the app, enter the following code
```python
python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel --alarm alarm.wav
```

## Reference
Check out [this website](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)
