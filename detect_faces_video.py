#!/usr/bin/python
# -*- coding: utf-8 -*-

# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel --alarm alarm.wav

# import the necessary packages

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

# Program 2 EAR MAR functions

EYE_AR_THRESH = 0.26
EYE_AR_CONSEC_FRAMES = 10
MOUTH_AR_THRESH = 0.4

SHOW_POINTS_FACE = True
SHOW_CONVEX_HULL_FACE = True
SHOW_INFO = True

ear = 0
mar = 0

COUNTER_FRAMES_EYE = 0
COUNTER_FRAMES_MOUTH = 0
COUNTER_BLINK = 0
COUNTER_MOUTH = 0

def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[5], mouth[8])
    B = dist.euclidean(mouth[1], mouth[11])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)


# construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--prototxt', required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument('-m', '--model', required=True,
                help='path to Caffe pre-trained model')
ap.add_argument('-a', '--alarm', type=str, default='',
                help='path alarm .WAV file')
ap.add_argument('-c', '--confidence', type=float, default=0.5,
                help='minimum probability to filter weak detections')
args = vars(ap.parse_args())

# Assigning Initial Values to the counter and the alarm

COUNTER = 0
ALARM_ON = False

# load our serialized model from disk

print( '[INFO] loading model...')
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

# initialize the video stream and allow the cammera sensor to warmup

print ('[INFO] starting video stream...')

# vs = VideoStream(src=1).start()

vs = cv2.VideoCapture(1)
time.sleep(2.0)

# Program 2 to plot facial landmarks

(ret, frame) = vs.read()
size = frame.shape

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat'
                                 )
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

model_points = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
    ])

focal_length = size[1]
center = (size[1] / 2, size[0] / 2)

camera_matrix = np.array([[focal_length, 0, center[0]], [0,
                         focal_length, center[1]], [0, 0, 1]],
                         dtype='double')

dist_coeffs = np.zeros((4, 1))

t_end = time.time()

# loop over the frames from the video stream

while True:

    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels

    (ret, frame) = vs.read()

    # frame = imutils.resize(frame, width=400)

    # Program 2 to change to grayscale

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    # grab the frame dimensions and convert it to a blob

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions

    net.setInput(blob)
    detections = net.forward()

    # loop over the detections

    for i in range(0, detections.shape[2]):

        # extract the confidence (i.e., probability) associated with the
        # prediction

        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence

        if confidence < args['confidence']:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')

        # draw the bounding box of the face along with the associated
        # probability

        text = '{:.2f}%'.format(confidence * 100)
        y = (startY - 10 if startY - 10 > 10 else startY + 10)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0,
                      0xFF), 2)
        cv2.putText(
            frame,
            text,
            (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0xFF),
            2,
            )

         # Program 2 for EAR and MAR code

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            jaw = shape[48:61]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = mouth_aspect_ratio(jaw)

            image_points = np.array([
                (shape[30][0], shape[30][1]),
                (shape[8][0], shape[8][1]),
                (shape[36][0], shape[36][1]),
                (shape[45][0], shape[45][1]),
                (shape[48][0], shape[48][1]),
                (shape[54][0], shape[54][1]),
                ], dtype='double')

            (success, rotation_vector, translation_vector) = \
                cv2.solvePnP(model_points, image_points, camera_matrix,
                             dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            (nose_end_point2D, jacobian) = \
                cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]),
                                  rotation_vector, translation_vector,
                                  camera_matrix, dist_coeffs)

            if SHOW_POINTS_FACE:
                for p in image_points:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0,
                               0xFF), -1)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]),
                  int(nose_end_point2D[0][0][1]))

            if SHOW_CONVEX_HULL_FACE:
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                jawHull = cv2.convexHull(jaw)

                cv2.drawContours(frame, [leftEyeHull], 0, (0xFF, 0xFF,
                                 0xFF), 1)
                cv2.drawContours(frame, [rightEyeHull], 0, (0xFF, 0xFF,
                                 0xFF), 1)
                cv2.drawContours(frame, [jawHull], 0, (0xFF, 0xFF,
                                 0xFF), 1)
                cv2.line(frame, p1, p2, (0xFF, 0xFF, 0xFF), 2)

            if p2[1] > p1[1] * 1.5 or COUNTER_BLINK > 25 \
                or COUNTER_MOUTH > 2:
                cv2.putText(
                    frame,
                    'Sending Alert!',
                    (200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0xFF),
                    2,
                    )
                os.system('python os.py')

                COUNTER_MOUTH = 0
            if ear < EYE_AR_THRESH:
                COUNTER_FRAMES_EYE += 1

                if COUNTER_FRAMES_EYE >= EYE_AR_CONSEC_FRAMES:

                        # if the alarm is not on, turn it on

                    if not ALARM_ON:
                        ALARM_ON = True

                    # check to see if an alarm file was supplied,
                    # and if so, start a thread to have the alarm
                    # sound played in the background

                        if args['alarm'] != '':
                            t = Thread(target=sound_alarm,
                                    args=(args['alarm'], ))
                            t.deamon = True
                            t.start()
                    cv2.putText(
                        frame,
                        'Sleeping Driver!',
                        (200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0xFF),
                        2,
                        )
            else:

                #os.system("python sound.py")

                if COUNTER_FRAMES_EYE > 2:
                    COUNTER_BLINK += 1
                COUNTER_FRAMES_EYE = 0
                ALARM_ON = False

            if mar >= MOUTH_AR_THRESH:
                COUNTER_FRAMES_MOUTH += 1
            else:
                if COUNTER_FRAMES_MOUTH > 5:
                    COUNTER_MOUTH += 1

                COUNTER_FRAMES_MOUTH = 0

            if time.time() - t_end > 120:
                t_end = time.time()
                COUNTER_BLINK = 0
                COUNTER_MOUTH = 0

    if SHOW_INFO:
        cv2.putText(
            frame,
            'EAR: {:.2f}'.format(ear),
            (30, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0xFF, 0, 0),
            2,
            )
        cv2.putText(
            frame,
            'MAR: {:.2f}'.format(mar),
            (200, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0xFF, 0, 0),
            2,
            )
        cv2.putText(
            frame,
            'Blinks: {}'.format(COUNTER_BLINK),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0xFF, 0, 0),
            2,
            )
        cv2.putText(
            frame,
            'Mouths: {}'.format(COUNTER_MOUTH),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0xFF, 0, 0),
            2,
            )

    # show the output frame

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop

    if key == ord('q'):
        break
    if key == ord('p'):
        SHOW_POINTS_FACE = not SHOW_POINTS_FACE
    if key == ord('c'):
        SHOW_CONVEX_HULL_FACE = not SHOW_CONVEX_HULL_FACE
    if key == ord('i'):
        SHOW_INFO = not SHOW_INFO
    time.sleep(0.02)

# do a bit of cleanup

cv2.destroyAllWindows()
vs.stop()


			
			
