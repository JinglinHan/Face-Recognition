import cv2
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
import os


def faceDetect(path):
    font = cv2.FONT_HERSHEY_SIMPLEX

    cascPath = "/usr/local/lib/python3.7/dist-packages/cv2/data/haarcascade_frontalface_default.xml"
    eyePath = "/usr/local/lib/python3.7/dist-packages/cv2/data/haarcascade_eye.xml"
    smilePath = "/usr/local/lib/python3.7/dist-packages/cv2/data/haarcascade_smile.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    eyeCascade = cv2.CascadeClassifier(eyePath)
    smileCascade = cv2.CascadeClassifier(smilePath)
    x_dirs = os.listdir(path)
    for i in range(len(x_dirs)):
        gray = cv2.imread(path + x_dirs[i], 0)
        # Load the image
        face_detect = dlib.get_frontal_face_detector()
        rects = face_detect(gray, 0)
        for (j, rect) in enumerate(rects):
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            gray = gray[y:(y+h), x:(x+w)]
        cv2.imwrite(path + x_dirs[i], gray)



