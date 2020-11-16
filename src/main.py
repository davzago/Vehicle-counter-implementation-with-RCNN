from Detector import Detector
from Tracker import Tracker
from RCNN import detect
import cv2
import time
import numpy as np
import os


#detector = Detector("data/frames")
tracker = Tracker()
#detector.detect("data/diff")
result_path = "data/results"

"""for i in range(0,len(detector.rects)):
    centers = tracker.update(detector.rects[i])
    img = cv2.imread("data/frames/%d.jpg" %i)
    for j in range(0,len(detector.rects[i])):
        cv2.rectangle(img, detector.rects[i][j][0], detector.rects[i][j][1], (0, 255, 0), 1)
    for c_id, c in centers.items():
        text = "ID{}".format(c_id)
        cv2.putText(img, text, (c[0] - 10, c[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(img, (c[0],c[1]), 4, (0, 255, 0), -1)
        
    cv2.imwrite(result_path + '/%d.jpg' % i, img)"""


vheicles = detect()

for i in range(0,len(vheicles)):
    centers = tracker.update(vheicles[i])
    img = cv2.imread("data/frames/%d.jpg" %(419+i))
    for j in range(0,len(vheicles[i])):
        cv2.rectangle(img, (vheicles[i][j][0], vheicles[i][j][1]), (vheicles[i][j][0]+vheicles[i][j][2],vheicles[i][j][1]+vheicles[i][j][3]), (0, 255, 0), 1)
    for c_id, c in centers.items():
        text = "ID{}".format(c_id)
        cv2.putText(img, text, (c[0] - 10, c[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(img, (c[0],c[1]), 4, (0, 255, 0), -1)
        
    cv2.imwrite(result_path + '/%d.jpg' % i, img)
    

    

