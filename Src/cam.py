import cv2
import numpy as np
cap = cv2.VideoCapture(2)
while True:
    ret,frame = cap.read()
    cv2.imshow('camera',frame)

    if cv2.waitKey(1)&0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindows()