# Driver-Fatigue-Detection-with-OpenCV-and-Deep-Learning

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

>python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel --alarm alarm.wav
