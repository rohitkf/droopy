# Driver-Fatigue-Detection-with-OpenCV-and-Deep-Learning 
A state of the art face detection system, that identifies and extracts the facial features from live webcam feed, to detect whether the driver is getting tired or not.

Research paper for the app : [Click here](https://github.com/rkf2778/Driver-Fatigue-Detection-with-OpenCV-and-Deep-Learning/blob/master/ICOEI_2019_paper_247.pdf)

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
CMake
opencv-python
imutils
dlib
scipy
playsound
numpy==1.21.5
```

## Getting Started

1. Install Python **[Click here](https://www.python.org/downloads/)**

2. Install Git **[Click here](https://git-scm.com/)**

3. Install Anaconda [Click here](https://www.anaconda.com/download)
   
4. Clone this repository
```Bash
git clone https://github.com/rkf2778/Driver-Fatigue-Detection-with-OpenCV-and-Deep-Learning/tree/master
cd Driver-Fatigue-Detection-with-OpenCV-and-Deep-Learning
```

5. Create an environment
```
conda create --name facedetect
conda activate facedetect
```

6. Install the modules
```python
pip install -r requirements.txt
```

If it causes error, then do the following :

- If facing issues with dlib, do either of the following
   -  Install CMake : ```pip install CMake``` 
   -  Visit [here](https://github.com/sachadee/Dlib) and install Dlib
   - Or else, try ```conda install conda-forge::dlib```

- Then install the following manually
```python
pip install opencv-python numpy scipy playsound
```

NOTE :
If you have more than one webcam, then you might need to change values for the following code inside `detect_faces_video.py` file to 0 or 1

```python
vs = cv2.VideoCapture(0) // For Inbuilt Camera
```
```python
vs = cv2.VideoCapture(1) // For External Camera
```

Also, you can control the threshold values in `detect_faces_video.py` file to control the threshold before alarm is activated
```python
EYE_AR_THRESH = 0.26
EYE_AR_CONSEC_FRAMES = 10
MOUTH_AR_THRESH = 0.4

SHOW_POINTS_FACE = True
SHOW_CONVEX_HULL_FACE = True
SHOW_INFO = True
```


6 To run the app, enter the following code
```python
python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel --alarm alarm.wav
```




## Reference
Check out [this website](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)
