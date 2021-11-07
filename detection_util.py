import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt



cap = cv2.VideoCapture(0)

while(True) :
    ret, frame = cap.read()

    if(ret) :
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv.destroyAllWindows()