# Driver-Fatigue-Detection-with-OpenCV-and-Deep-Learning

# These are the following modules used :

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

# To install the module, use pip
```python
pip install <module-name>
```


# To run the app, enter the following code
> python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel --alarm alarm.wav

# Reference
Check out [this website](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)
