import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

cap = cv.VideoCapture(os.getcwd().replace('air_challenge', '') + '/p2.mp4')
 
while cap.isOpened():
    ret, img = cap.read()

    cv.imshow('frame', img)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()